"""Read-only installation and real inference check; never downloads or trains."""
import importlib.metadata
import json
import sys

from .common import ROOT, CLASSES, digest


def check_artifact(root=ROOT):
    metadata = json.loads((root / "models/deployment.json").read_text(encoding="utf-8"))
    artifact = (root / metadata["artifact"]).resolve()
    if not artifact.is_relative_to(root.resolve()):
        raise ValueError("Deployment model must be inside the project folder.")
    if metadata["classes"] != CLASSES:
        raise ValueError("Deployment labels do not match the app's label order.")
    if digest(artifact) != metadata["artifact_sha256"]:
        raise ValueError("Model is incomplete or changed. Download the project again.")
    return metadata


def main():
    failures = []
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 is the supported version.")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            actual = importlib.metadata.version(name)
            # CUDA/CPU builds append a local version tag to the pinned release.
            if actual.split("+")[0] != expected:
                failures.append(f"{name}: installed {actual}; expected {expected}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{name} is missing")
    try:
        metadata = check_artifact()
        print(f"Model checksum and labels: OK ({metadata['model_id']})")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(f"Deployment: {exc}")
    if not failures:
        try:
            from PIL import Image
            from .models import Predictor
            result = Predictor().predict(Image.new("RGB", (160, 160), "white"))
            if result["label"] not in CLASSES:
                raise ValueError("Unexpected output label")
            print("CPU inference: OK (synthetic input; not an accuracy test)")
        except Exception as exc:
            failures.append(f"Inference could not run: {exc}")
    for failure in failures:
        print(f"CHECK: {failure}")
    if failures:
        print("Repair dependencies with: python -m pip install -r requirements.txt")
        return 1
    print("Ready to run WasteSort. No dataset download or retraining needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
