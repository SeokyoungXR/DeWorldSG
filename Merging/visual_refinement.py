import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VJEPA_ROOT = os.path.join(CURRENT_DIR, "vjepa2_lib")

if VJEPA_ROOT not in sys.path:
    sys.path.insert(0, VJEPA_ROOT)

try:
    from vjepa_src.models.vision_transformer import vit_giant
except ImportError as e:
    print(f"VJEPARefiner: V-JEPA import failed: {e}")
    vit_giant = None


REL_CLASSES_3RSCAN = [
    "attached to",
    "build in",
    "connected to",
    "hanging on",
    "part of",
    "standing on",
    "supported by",
]


class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def infer_probe_input_dim(state_dict):
    for key, value in state_dict.items():
        if key.endswith("net.1.weight"):
            return value.shape[1]
    return None


class BaseProbeRefiner:
    name = "base"

    def __init__(
        self,
        device="cuda",
        model_path=None,
        probe_path=None,
        rel_classes=None,
        batch_size=None,
        **kwargs,
    ):
        self.device = device
        self.model_path = model_path
        self.probe_path = probe_path
        self.rel_classes = rel_classes or REL_CLASSES_3RSCAN
        self.batch_size = batch_size or 8
        self.model = None
        self.processor = None
        self.probe = None

    @property
    def enabled(self):
        return self.model is not None and self.probe is not None

    def load_probe(self, input_dim):
        if not self.probe_path:
            print(f"{self.name}: probe_path is not set; relation prior disabled.")
            return
        if not os.path.exists(self.probe_path):
            print(f"{self.name}: probe weights not found at {self.probe_path}; relation prior disabled.")
            return

        state_dict = torch.load(self.probe_path, map_location="cpu")
        inferred_dim = infer_probe_input_dim(state_dict)
        input_dim = inferred_dim or input_dim

        self.probe = MLPProbe(input_dim, len(self.rel_classes))
        self.probe.load_state_dict(state_dict)
        self.probe.to(self.device).eval().float()
        print(f"{self.name}: loaded probe from {self.probe_path} with input_dim={input_dim}")

    def extract_features(self, crop_sequences):
        raise NotImplementedError

    def predict_relations(self, crop_sequences):
        if not self.enabled or len(crop_sequences) == 0:
            return []

        results = []
        for start in range(0, len(crop_sequences), self.batch_size):
            chunk = crop_sequences[start : start + self.batch_size]
            with torch.no_grad():
                features = self.extract_features(chunk).float()
                logits = self.probe(features)
                probs = torch.softmax(logits, dim=1)

            for prob_vec in probs:
                rel_dict = {
                    label: score.item()
                    for label, score in zip(self.rel_classes, prob_vec)
                }
                results.append(rel_dict)

            del features, logits, probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


class VJEPARefiner(BaseProbeRefiner):
    name = "vjepa"

    def __init__(self, device="cuda", model_path=None, probe_path=None, **kwargs):
        batch_size = kwargs.pop("batch_size", None) or 4
        super().__init__(
            device=device,
            model_path=model_path,
            probe_path=probe_path,
            batch_size=batch_size,
            **kwargs,
        )

        if vit_giant is None:
            print("VJEPARefiner: V-JEPA module not found; visual refinement disabled.")
            return

        weights_dir = os.path.join(os.path.dirname(CURRENT_DIR), "weights")
        if self.model_path is None:
            self.model_path = os.path.join(weights_dir, "vitg-384.pt")
        if self.probe_path is None:
            self.probe_path = os.path.join(CURRENT_DIR, "probes", "vjepa_probe_mlp.pt")

        print("VJEPARefiner: initializing V-JEPA ViT-Giant 384px.")
        self.model = vit_giant(img_size=384, num_frames=16, tubelet_size=2)
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"VJEPARefiner: loaded backbone from {self.model_path}")
        else:
            print(f"VJEPARefiner: backbone weights not found at {self.model_path}")

        self.model.to(self.device).eval().half()
        self.load_probe(1408)
        self.transform = T.Compose(
            [
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_features(self, crop_sequences):
        batch_tensors = []
        for seq in crop_sequences:
            frames = [self.transform(img) for img in seq]
            batch_tensors.append(torch.stack(frames, dim=1))

        batch = torch.stack(batch_tensors, dim=0).to(self.device).half()
        features = self.model(batch)
        if isinstance(features, list):
            features = features[-1]
        if features.dim() == 3:
            features = features.mean(dim=1)
        return features


def build_visual_refiner(
    refiner_name,
    device="cuda",
    model_path=None,
    probe_path=None,
    rel_classes=None,
    batch_size=None,
):
    refiner_name = (refiner_name or "vjepa").lower()
    refiner_cls = {
        "vjepa": VJEPARefiner,
    }.get(refiner_name)

    if refiner_cls is None:
        raise ValueError(f"Unknown visual refiner: {refiner_name}")

    return refiner_cls(
        device=device,
        model_path=str(model_path) if model_path else None,
        probe_path=str(probe_path) if probe_path else None,
        rel_classes=rel_classes,
        batch_size=batch_size,
    )
