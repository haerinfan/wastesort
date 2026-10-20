import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("WASTESORT_ROOT", Path(__file__).resolve().parents[1])).resolve()
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SEED = 42


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
