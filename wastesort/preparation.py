"""Day 2: publish an inventory and an explicit preprocessing contract, without training."""
from pathlib import Path

import pandas as pd
from PIL import Image

from .common import CLASSES, ROOT, read_json, save_json


def build_inventory():
    manifest = pd.read_csv(ROOT / "data/splits.csv")
    excluded = pd.read_csv(ROOT / "reports/excluded.csv").fillna("")
    by_path = manifest.set_index("path")
    reasons = dict(zip(excluded.path, excluded.reason))
    encoding = {label: i for i, label in enumerate(CLASSES)}
    rows = []
    for label in CLASSES:
        for path in sorted((ROOT / "data/raw" / label).glob("*")):
            relative = path.relative_to(ROOT).as_posix()
            with Image.open(path) as image:
                image.load()
                row = {
                    "path": relative, "label": label, "label_id": encoding[label],
                    "format": image.format, "mode": image.mode, "width": image.width,
                    "height": image.height, "file_bytes": path.stat().st_size,
                    "included": relative in by_path.index,
                    "exclusion_reason": reasons.get(relative, ""),
                    "split": by_path.loc[relative, "split"] if relative in by_path.index else "excluded",
                    "group": by_path.loc[relative, "group"] if relative in by_path.index else "",
                }
            rows.append(row)
    inventory = pd.DataFrame(rows)
    audit = read_json(ROOT / "reports/data_audit.json")
    if len(inventory) != audit["source_images"] or int(inventory.included.sum()) != audit["usable_images"]:
        raise ValueError("Inventory does not match the frozen audit.")
    reports = ROOT / "reports"
    inventory.to_csv(reports / "dataset_inventory.csv", index=False, lineterminator="\n")
    save_json(reports / "label_encoding.json", encoding)
    save_json(reports / "preprocessing.json", {
        "source_images": len(inventory),
        "usable_images": int(inventory.included.sum()),
        "formats": inventory.format.value_counts().to_dict(),
        "modes": inventory["mode"].value_counts().to_dict(),
        "dimensions": inventory.groupby(["width", "height"]).size().reset_index(name="images").to_dict("records"),
        "label_encoding": encoding,
        "label_id_meaning": "Category identifiers only; not quantities or rankings.",
        "hog": {"size": [64, 64], "color": "grayscale", "pixel_range": [0, 1], "features": 1764},
        "cnn": {"size": [160, 160], "color": "RGB", "pixel_range": [0, 1], "tensor_shape": [3, 160, 160]},
        "mobilenet": {"size": [160, 160], "color": "RGB", "mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225], "tensor_shape": [3, 160, 160]},
        "augmentation": "Training only; see protocol.json. Validation/test transformations are deterministic.",
        "split_counts": audit["split_counts"],
        "manifest_sha256": audit["manifest_sha256"],
        "human_review": "Pending; the generated inventory and review worksheet do not constitute human approval.",
    })
    pairs = pd.read_csv(reports / "near_duplicate_candidates.csv")
    review = [{"kind": "near_duplicate_candidate", "left": r.left, "right": r.right,
               "automated_reason": f"pHash Hamming distance {r.hamming_distance}",
               "reviewer": "", "decision": "", "notes": ""} for r in pairs.itertuples()]
    review += [{"kind": "excluded_file", "left": r.path, "right": "",
                "automated_reason": r.reason, "reviewer": "", "decision": "", "notes": ""}
               for r in excluded.itertuples()]
    worksheet = reports / "manual_review_worksheet.csv"
    # Preserve any human annotations across repeated preparation runs.
    if not worksheet.exists():
        pd.DataFrame(review).to_csv(worksheet, index=False, lineterminator="\n")
    print(f"Day 2 inventory: {len(inventory)} source images; {int(inventory.included.sum())} included.")
    print(inventory.groupby(["label", "included"]).size().unstack(fill_value=0).to_string())
    return inventory


if __name__ == "__main__":
    build_inventory()
