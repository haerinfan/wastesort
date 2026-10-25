"""Classify local photos without a browser; no dataset changes or training."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from wastesort.models import MAX_BYTES, Predictor, read_upload


def classify(paths, predictor):
    rows = []
    for path in paths:
        row = {"file": str(path), "label": "", "model": "", "top_score": "", "error": ""}
        try:
            with path.open("rb") as source:
                image = read_upload(source.read(MAX_BYTES + 1))
            result = predictor.predict(image)
            row.update(label=result["label"], model=result["model"])
            if result["scores"] is not None:
                row["top_score"] = result["scores"][result["label"]]
        except (OSError, ValueError, RuntimeError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Classify JPEG/PNG photos with the bundled WasteSort model")
    parser.add_argument("photos", nargs="+", type=Path, help="One or more photo paths; quote paths with spaces")
    parser.add_argument("--output", type=Path, required=True, help="New .csv or .json report")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".json", ".csv"}:
        parser.error("Output must end in .csv or .json")
    if args.output.exists():
        parser.error("Output already exists. Choose another filename to preserve it.")
    try:
        predictor = Predictor()
        rows = classify(args.photos, predictor)
        # Exclusive creation also protects against an output appearing during inference.
        with args.output.open("x", encoding="utf-8", newline="") as target:
            if args.output.suffix.lower() == ".csv":
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                # Prevent a user-supplied filename from becoming a spreadsheet formula.
                for row in rows:
                    writer.writerow({key: ("'" + value if isinstance(value, str) and
                                           value.startswith(("=", "+", "-", "@", "\t", "\r")) else value)
                                     for key, value in row.items()})
            else:
                json.dump({"created_utc": datetime.now(timezone.utc).isoformat(),
                           "model_sha256": predictor.metadata["artifact_sha256"],
                           "note": "Uncalibrated model scores; predictions do not establish accuracy or disposal rules.",
                           "predictions": rows}, target, indent=2)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Could not complete prediction: {exc}", file=sys.stderr)
        return 1
    failures = sum(bool(row["error"]) for row in rows)
    print(f"Saved {len(rows) - failures} predictions and {failures} errors to {args.output}")
    print("Scores are uncalibrated. Check the item and local disposal rules yourself.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
