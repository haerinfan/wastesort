"""Primary set (synthetic): prompt plan, quality control, deduplication and grouped splits.

Generation needs a GPU and runs in ``wastesort.generate_primary``. Everything in this
module is CPU-only, so the selection and split can be checked without regenerating.
"""
import base64
import hashlib
import html
import io
import math
import shutil
from datetime import datetime, timezone

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from .common import CLASSES, ROOT, SEED, digest, read_json, save_json
from .data import freeze_manifest, make_split

PROTOCOL = ROOT / "protocol_primary.json"
DATA = ROOT / "data/primary_set"
REPORTS = ROOT / "reports/primary_set"
DISCLOSURE = "AI-generated synthetic image; WasteSort primary set (synthetic). Not a photograph of a real object."


def load_protocol():
    protocol = read_json(PROTOCOL)
    if protocol["classes"] != CLASSES:
        raise ValueError("Primary-set classes must match the six TrashNet labels in order.")
    return protocol


def candidates_per_class(protocol, label):
    target = protocol["quality_control"]["target_counts"][label]
    return math.ceil(protocol["generator"]["candidate_multiplier"] * target)


def build_prompts(protocol):
    """Deterministic candidate plan: every item x setting pair is cycled before any repeats."""
    rows = []
    for class_index, label in enumerate(CLASSES):
        rng = np.random.default_rng(SEED + class_index)
        items = protocol["items"][label]
        settings = protocol["settings"]
        pairs = [(i, s) for s in range(len(settings)) for i in range(len(items))]
        # Prefix-stable: raising the multiplier later only appends new candidates.
        for n in range(candidates_per_class(protocol, label)):
            item_index, setting_index = pairs[n % len(pairs)]
            condition = protocol["conditions"][label][rng.integers(len(protocol["conditions"][label]))]
            lighting = protocol["lighting"][rng.integers(len(protocol["lighting"]))]
            angle = protocol["angles"][rng.integers(len(protocol["angles"]))]
            item = items[item_index]
            # "a clean a glass jar" -> drop the item's own article after the condition's article.
            noun = item.split(" ", 1)[1] if item.split(" ", 1)[0] in {"a", "an"} else item
            prompt = protocol["prompt_template"].format(
                condition=condition, item=noun, setting=settings[setting_index],
                lighting=lighting, angle=angle)
            candidate_id = f"ps_{label}_{n:04d}"
            rows.append({"candidate_id": candidate_id, "label": label, "item": item,
                         "condition": condition, "setting": settings[setting_index],
                         "lighting": lighting, "angle": angle, "prompt": prompt,
                         "seed": SEED * 100_000 + class_index * 10_000 + n,
                         "group": f"{label}|item{item_index}|setting{setting_index}",
                         "candidate_path": f"data/primary_set/candidates/{label}/{candidate_id}.jpg"})
    return rows


def accept_candidate(probabilities, label, qc):
    """Return None to keep a generated candidate, or a short rejection reason.

    probabilities: dict class -> CLIP zero-shot probability (sums to 1 over CLASSES).
    label: the class the prompt asked for.
    qc: protocol_primary.json["quality_control"] (min_label_probability, min_margin).
    """
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    if ranked[0] != label:
        return "clip_top_class_mismatch"
    if probabilities[label] < qc["min_label_probability"]:
        return "clip_low_confidence"
    # A narrow lead means CLIP finds the image ambiguous between two materials.
    if probabilities[label] - probabilities[ranked[1]] < qc["min_margin"]:
        return "clip_ambiguous"
    return None


def select_to_targets(frame, targets):
    """Round-robin over items in candidate order, so no single item dominates a class."""
    chosen = []
    for label in CLASSES:
        target = targets[label]
        rows = frame[frame.label == label]
        queues = [list(g.index) for _, g in rows.groupby("item", sort=False)]
        picked = []
        while len(picked) < target and any(queues):
            for queue in queues:
                if queue and len(picked) < target:
                    picked.append(queue.pop(0))
        chosen.extend(picked)
    return frame.loc[chosen]


def image_record(path):
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
    return {"width": image.width, "height": image.height, "sha256": digest(path),
            "pixel_sha256": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
            "phash": str(imagehash.phash(image))}


def screen(candidates, protocol, known_phashes=()):
    """Apply QC, exact and near-duplicate checks. Returns kept rows and rejected rows."""
    qc = protocol["quality_control"]
    limit = qc["near_duplicate_hamming"]
    known = [int(h, 16) for h in known_phashes]
    kept, rejected, seen_pixels, kept_hashes = [], [], set(), []
    for row in candidates:
        path = ROOT / row["candidate_path"]
        reason = None
        if not path.exists():
            reason = "not_generated"
        else:
            try:
                row.update(image_record(path))
            except (OSError, ValueError) as exc:
                reason = f"unreadable: {exc}"
        if reason is None:
            probabilities = {c: float(row[f"clip_{c}"]) for c in CLASSES}
            row["clip_label_prob"] = probabilities[row["label"]]
            reason = accept_candidate(probabilities, row["label"], qc)
        if reason is None:
            value = int(row["phash"], 16)
            if row["pixel_sha256"] in seen_pixels:
                reason = "exact_pixel_duplicate"
            elif any((value ^ other).bit_count() <= limit for other in kept_hashes):
                reason = "near_duplicate_within_primary_set"
            elif any((value ^ other).bit_count() <= limit for other in known):
                reason = "near_duplicate_of_trashnet_image"
        if reason:
            rejected.append({"candidate_id": row["candidate_id"], "label": row["label"],
                             "reason": reason})
            continue
        seen_pixels.add(row["pixel_sha256"])
        kept_hashes.append(int(row["phash"], 16))
        kept.append(row)
    return kept, rejected


def contact_sheet(frame, target):
    cards = []
    for row in frame.itertuples():
        image = Image.open(ROOT / row.path).convert("RGB")
        image.thumbnail((150, 112))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=80)
        uri = base64.b64encode(buffer.getvalue()).decode()
        cards.append(f'<figure><img src="data:image/jpeg;base64,{uri}" alt=""><figcaption>'
                     f'<b>{html.escape(row.label)}</b> · {html.escape(row.split)}<br>'
                     f'{html.escape(row.item)}<br>CLIP {row.clip_label_prob:.2f}</figcaption></figure>')
    target.write_text(
        "<!doctype html><meta charset=utf-8><title>Primary set (synthetic) contact sheet</title>"
        "<style>body{font:12px Arial;margin:20px}main{display:flex;flex-wrap:wrap}"
        "figure{width:160px;margin:6px}img{width:150px}</style>"
        "<h1>WasteSort primary set (synthetic)</h1><p>Every image is AI-generated. Review each label; "
        "record wrong or unclear images in reports/primary_set/manual_review.csv.</p>"
        "<main>" + "".join(cards) + "</main>", encoding="utf-8")


def prepare():
    protocol = load_protocol()
    qc = protocol["quality_control"]
    log = DATA / "candidates.csv"
    if not log.exists():
        raise SystemExit("Run python -m wastesort.generate_primary first (GPU).")
    candidates = pd.read_csv(log).to_dict("records")
    trashnet = ROOT / "data/splits.csv"
    known = pd.read_csv(trashnet).phash.tolist() if trashnet.exists() else []
    kept, rejected = screen(candidates, protocol, known)
    targets = qc["target_counts"]
    frame = select_to_targets(pd.DataFrame(kept), targets)
    counts = frame.label.value_counts().reindex(CLASSES, fill_value=0)
    minimum = {c: math.ceil(qc["min_fraction_of_target"] * targets[c]) for c in CLASSES}
    short = {c: int(counts[c]) for c in CLASSES if counts[c] < minimum[c]}
    if short:
        raise ValueError(f"Too few accepted images {short}; minimum {minimum}. Raise "
                         "generator.candidate_multiplier, rerun generation (it resumes) and record why. "
                         "Do not lower the quality rule after looking at results.")
    surplus = set(pd.DataFrame(kept).candidate_id) - set(frame.candidate_id)
    rejected += [{"candidate_id": c, "label": c.split("_")[1], "reason": "surplus_above_target_count"}
                 for c in sorted(surplus)]
    images = DATA / "images"
    paths = []
    for row in frame.itertuples():
        target = images / row.label / f"{row.candidate_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT / row.candidate_path, target)
        paths.append(target.relative_to(ROOT).as_posix())
    frame = frame.assign(path=paths)
    columns = ["path", "label", "item", "condition", "setting", "lighting", "angle", "prompt", "seed",
               "clip_label_prob", "width", "height", "sha256", "pixel_sha256", "phash", "group"]
    frame = make_split(frame[columns].reset_index(drop=True))
    manifest = DATA / "splits.csv"
    freeze_manifest(frame, manifest)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected, columns=["candidate_id", "label", "reason"]).to_csv(
        REPORTS / "excluded.csv", index=False, lineterminator="\n")
    split_counts = pd.crosstab(frame.label, frame.split).reindex(CLASSES)
    split_counts.to_csv(REPORTS / "split_counts.csv", lineterminator="\n")
    environment = DATA / "generation_environment.json"
    save_json(DATA / "provenance.json", {
        "dataset": protocol["dataset_name"], "nature": protocol["nature"],
        "generator": protocol["generator"], "quality_control_model": qc["checker"],
        "quality_control_revision": qc["checker_revision"],
        "protocol_sha256": digest(PROTOCOL), "candidates_sha256": digest(log),
        "manifest_sha256": digest(manifest),
        "generation_environment": read_json(environment) if environment.exists() else None,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": DISCLOSURE,
    })
    reasons = pd.Series([r["reason"].split(":")[0] for r in rejected]).value_counts().to_dict()
    save_json(REPORTS / "audit.json", {
        "candidates_planned": len(candidates), "accepted_images": len(frame),
        "class_counts": counts.to_dict(), "target_counts": targets,
        "target_counts_source": qc["target_counts_source"], "rejected_by_reason": reasons,
        "split_counts": frame.split.value_counts().to_dict(),
        "group_count": int(frame.group.nunique()),
        "quality_rule": {k: qc[k] for k in ["min_label_probability", "min_margin",
                                            "near_duplicate_hamming"]},
        "selection_bias_note": "CLIP filtering keeps images CLIP recognises. Accepted images may be easier "
                               "than all generated images; report the rejection counts with any result.",
        "manifest_sha256": digest(manifest),
    })
    contact_sheet(frame, REPORTS / "contact_sheet.html")
    print(split_counts.to_string(), flush=True)
    print(f"Primary set (synthetic): {len(frame)} images, {len(rejected)} candidates excluded", flush=True)


if __name__ == "__main__":
    prepare()
