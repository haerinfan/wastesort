"""Download only the official resized archive; audit and freeze grouped splits."""
import csv
import hashlib
import html
import io
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from .common import CLASSES, ROOT, SEED, digest, read_json, save_json

SOURCE = "https://huggingface.co/datasets/garythung/trashnet/resolve/94cd17e/dataset-resized.zip"
EXPECTED = {"cardboard": 403, "glass": 501, "metal": 410, "paper": 594, "plastic": 482, "trash": 137}


def download():
    archive = ROOT / "data/downloads/dataset-resized.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temp = archive.with_suffix(".part")
        print("Downloading original creators' resized TrashNet archive...", flush=True)
        with urllib.request.urlopen(SOURCE, timeout=120) as response, temp.open("wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block)
        temp.replace(archive)
    provenance = ROOT / "data/provenance.json"
    sha = digest(archive)
    if provenance.exists() and read_json(provenance)["archive_sha256"] != sha:
        raise ValueError("Archive differs from frozen provenance. Investigate before continuing.")
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if "__MACOSX" in parts or member.is_dir():
                continue
            if len(parts) == 3 and parts[0] == "dataset-resized" and parts[1] in CLASSES:
                name = parts[2]
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                target = raw / parts[1] / name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    target.write_bytes(z.read(member))
    if not provenance.exists():
        save_json(provenance, {
            "dataset": "TrashNet, resized archive only",
            "creators": "Gary Thung and Mindy Yang",
            "source_repository": "https://github.com/garythung/trashnet",
            "archive_url": SOURCE, "archive_sha256": sha,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "license": "MIT as identified by the creators' dataset card",
            "expected_counts": EXPECTED,
        })
    return raw


def group_hashes(hashes, distance=6):
    """Connected components of candidate near-duplicates, including transitive links."""
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i, a in enumerate(hashes):
        for j in range(i):
            d = (a ^ hashes[j]).bit_count()
            if d <= distance:
                parent[find(i)] = find(j)
                pairs.append((j, i, d))
    return [find(i) for i in range(len(hashes))], pairs


def make_split(frame):
    """20 stratified group folds combined into approximately 70/15/15 partitions."""
    result = frame.copy()
    fold = np.zeros(len(result), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for k, (_, indices) in enumerate(splitter.split(result.path, result.label, result.group)):
        fold[indices] = k
    result["split"] = np.where(fold < 14, "train", np.where(fold < 17, "validation", "test"))
    assert result.groupby("group").split.nunique().max() == 1
    for split in ["train", "validation", "test"]:
        if set(result[result.split == split].label) != set(CLASSES):
            raise ValueError(f"Missing class in {split}; do not silently change the seed.")
    return result


def prepare():
    raw = download()
    rows, rejected = [], []
    seen = {}
    for label in CLASSES:
        for path in sorted((raw / label).glob("*")):
            try:
                with Image.open(path) as source:
                    source.load()
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    pixel_hash = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()
                    row = {"path": path.relative_to(ROOT).as_posix(), "label": label,
                           "width": image.width, "height": image.height, "mode": source.mode,
                           "sha256": digest(path), "pixel_sha256": pixel_hash,
                           "phash": str(imagehash.phash(image))}
                rows.append(row)
            except (OSError, ValueError) as exc:
                rejected.append({"path": path.relative_to(ROOT).as_posix(),
                                 "reason": f"unreadable: {exc}", "retained": ""})
    for row in rows:
        seen.setdefault(row["pixel_sha256"], []).append(row)
    clean = []
    for identical in seen.values():
        if len({r["label"] for r in identical}) > 1:
            rejected.extend({"path": r["path"], "reason": "conflicting_labels_for_identical_pixels",
                             "retained": ""} for r in identical)
        else:
            clean.append(identical[0])
            rejected.extend({"path": r["path"], "reason": "exact_pixel_duplicate",
                             "retained": identical[0]["path"]} for r in identical[1:])
    rows = clean
    actual = {c: len(list((raw / c).glob("*"))) for c in CLASSES}
    if actual != EXPECTED:
        raise ValueError(f"Unexpected source counts: {actual}")
    groups, pairs = group_hashes([int(r["phash"], 16) for r in rows])
    for row, group in zip(rows, groups):
        row["group"] = f"group_{group:04d}"
    frame = make_split(pd.DataFrame(rows))
    manifest = ROOT / "data/splits.csv"
    serialized = frame.to_csv(index=False, lineterminator="\n")
    if manifest.exists() and manifest.read_text(encoding="utf-8") != serialized:
        raise ValueError("Frozen manifest would change. Review differences in a separate experiment.")
    manifest.write_text(serialized, encoding="utf-8")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame(rejected, columns=["path", "reason", "retained"]).to_csv(reports / "excluded.csv", index=False)
    pd.DataFrame([{"left": rows[a]["path"], "right": rows[b]["path"], "hamming_distance": d,
                   "same_label": rows[a]["label"] == rows[b]["label"]}
                  for a, b, d in pairs],
                 columns=["left", "right", "hamming_distance", "same_label"]).to_csv(
                     reports / "near_duplicate_candidates.csv", index=False)
    counts = pd.crosstab(frame.label, frame.split).reindex(CLASSES)
    counts.to_csv(reports / "split_counts.csv")
    save_json(reports / "data_audit.json", {
        "source_images": sum(actual.values()), "usable_images": len(frame),
        "source_class_counts": actual, "excluded_images": len(rejected),
        "near_duplicate_candidate_pairs": len(pairs),
        "near_duplicate_threshold": "64-bit perceptual hash Hamming distance <= 6; transitive groups",
        "group_count": len(set(groups)),
        "manual_review": "Candidate grouping is conservative screening, not proof all duplicate objects were found.",
        "split_counts": frame.split.value_counts().to_dict(),
        "split_method": "StratifiedGroupKFold(20, shuffle=True, random_state=42); folds 0-13/14-16/17-19",
        "manifest_sha256": digest(manifest),
    })
    # A local visual inventory includes every image and candidate pair for group review.
    import base64
    visual = reports / "audit_images"
    visual.mkdir(exist_ok=True)
    sections = []
    for i, row in enumerate(rows):
        image = Image.open(ROOT / row["path"]).convert("RGB")
        image.thumbnail((130, 100))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG")
        uri = base64.b64encode(buffer.getvalue()).decode()
        sections.append(f'<figure><img src="data:image/jpeg;base64,{uri}"><figcaption>'
                        f'{html.escape(row["path"])}<br>{frame.iloc[i]["split"]}</figcaption></figure>')
    (visual / "inventory.html").write_text(
        "<title>TrashNet visual inventory</title><style>body{font:12px Arial}main{display:flex;flex-wrap:wrap}"
        "figure{width:150px;margin:8px;overflow-wrap:anywhere}</style><h1>TrashNet visual inventory</h1>"
        "<p>Inspect labels, quality and near-duplicate candidates. Automated grouping is provisional.</p>"
        "<main>" + "".join(sections) + "</main>", encoding="utf-8")
    print(counts.to_string(), flush=True)
    print(f"Prepared {len(frame)} images; {len(pairs)} candidate pairs; manifest {digest(manifest)}", flush=True)


if __name__ == "__main__":
    prepare()
