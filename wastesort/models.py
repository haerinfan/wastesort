"""Identical preprocessing for training, evaluation, and the upload app."""
import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.feature import hog

from .common import CLASSES, ROOT, read_json

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def read_upload(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Choose a JPEG or PNG file smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Choose an image with at most 20 million pixels.")
                if min(image.size) < 32:
                    raise ValueError("Choose an image at least 32 pixels wide and tall.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This file cannot be read as a safe JPEG or PNG image.") from exc


def hog_features(image):
    gray = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)


def transforms_for(kind, training=False):
    from torchvision import transforms as T
    steps = [T.Resize((160, 160))]
    if training:
        steps += [T.RandomHorizontalFlip(), T.RandomRotation(10, fill=255),
                  T.ColorJitter(brightness=0.1, contrast=0.1)]
    steps += [T.ToTensor()]
    if kind == "mobilenet":
        steps += [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(steps)


def build_network(kind, pretrained=False):
    from torch import nn
    if kind == "mobilenet":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, len(CLASSES)))
        return model
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # Fixed 160px input produces a 20x20 map: kernel 5 is exactly adaptive 4x4 averaging,
        # while permitting deterministic CUDA backward on the supported Torch version.
        nn.AvgPool2d(5), nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(64 * 4 * 4, len(CLASSES)),
    )


class Predictor:
    """Loads only the project's trusted, locally bundled deployment checkpoint."""
    def __init__(self):
        self.metadata = read_json(ROOT / "models/deployment.json")
        self.kind = self.metadata["kind"]
        path = ROOT / self.metadata["artifact"]
        from .common import digest
        if digest(path) != self.metadata["artifact_sha256"]:
            raise ValueError("Model checksum does not match deployment metadata.")
        if self.kind == "classical":
            import joblib
            self.model = joblib.load(path)
        else:
            import torch
            torch.set_num_threads(2)
            self.model = build_network(self.kind)
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.model.eval()
            self.transform = transforms_for(self.kind)

    def predict(self, image):
        if self.kind == "classical":
            x = hog_features(image)[None]
            predicted = int(self.model.predict(x)[0])
            # SVM margins are not probabilities. Only display a score when supplied by a probability model.
            probabilities = self.model.predict_proba(x)[0] if hasattr(self.model, "predict_proba") else None
        else:
            import torch
            with torch.inference_mode():
                probabilities = self.model(self.transform(image)[None]).softmax(1)[0].numpy()
            predicted = int(np.argmax(probabilities))
        return {"label": CLASSES[predicted],
                "scores": None if probabilities is None else dict(zip(CLASSES, map(float, probabilities))),
                "model": self.metadata["model_id"]}
