"""Evaluate a separately labeled photo set without modifying the published study."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from wastesort.common import CLASSES, ROOT, digest
from wastesort.models import MAX_BYTES, Predictor, read_upload


def evaluate(manifest, predictor, known_pixels):
    manifest = Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'item_id', 'photo', 'label'} <= set(reader.fieldnames or []):
            raise ValueError('CSV needs item_id, photo and label columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('Add labeled photos before evaluating; the template is empty.')
    seen_items, seen_paths, seen_pixels = set(), set(), set()
    predictions = []
    for number, row in enumerate(rows, 2):
        item, photo, label = ((row.get(key) or '').strip() for key in ('item_id', 'photo', 'label'))
        if not item or not photo or label not in CLASSES:
            raise ValueError(f'Row {number}: item_id/photo required and label must be one of {CLASSES}.')
        path = (manifest.parent / photo).resolve()
        if item in seen_items or path in seen_paths:
            raise ValueError(f'Row {number}: use one photo per distinct item; duplicate item or path.')
        with path.open('rb') as source:
            payload = source.read(MAX_BYTES + 1)
        image = read_upload(payload)
        pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if pixels in known_pixels:
            raise ValueError(f'Row {number}: photo matches the recorded study; use a new photo.')
        if pixels in seen_pixels:
            raise ValueError(f'Row {number}: duplicate image pixels in this evaluation.')
        seen_items.add(item)
        seen_paths.add(path)
        seen_pixels.add(pixels)
        result = predictor.predict(image)
        if result['label'] not in CLASSES:
            raise ValueError('Predictor returned an unknown label.')
        predictions.append({'item_id': item, 'actual': label, 'predicted': result['label'],
                            'correct': label == result['label'],
                            'photo_sha256': hashlib.sha256(payload).hexdigest(),
                            'pixel_sha256': pixels})
    actual = [r['actual'] for r in predictions]
    predicted = [r['predicted'] for r in predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0)
    covered = [i for i, n in enumerate(support) if n]
    complete = len(covered) == len(CLASSES)
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'evaluation_type': 'separate_labeled_photos', 'n': len(rows),
            'manifest_sha256': digest(manifest),
            'model_sha256': predictor.metadata['artifact_sha256'],
            'model': predictor.metadata['model_id'], 'classes': CLASSES,
            'accuracy': accuracy_score(actual, predicted),
            'six_class_macro_f1': float(f1.mean()) if complete else None,
            'macro_f1_present_classes': float(f1[covered].mean()),
            'missing_classes': [CLASSES[i] for i, n in enumerate(support) if not n],
            'per_class': {name: {'precision': float(precision[i]),
                                'recall': float(recall[i]) if support[i] else None,
                                'f1': float(f1[i]) if support[i] else None,
                                'support': int(support[i])} for i, name in enumerate(CLASSES)},
            'confusion_matrix': confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
            'notes': ['Results apply only to the supplied labels and photos.',
                      'Six-class macro F1 is withheld if a class has no actual examples.',
                      'Exact pixel checks do not detect all crops, near-duplicates or repeated objects.',
                      'The tool cannot verify campus origin, label quality or collection independence.',
                      'No original benchmark reports or model weights were changed.'],
            'predictions': predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='CSV with item_id,photo,label; photo paths relative to CSV')
    parser.add_argument('--output', required=True, type=Path, help='New JSON report outside published reports/')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix.lower() != '.json' or output.exists():
        parser.error('Choose a new .json output file; existing reports are never overwritten.')
    if output.is_relative_to(ROOT / 'reports'):
        parser.error('Save separate evaluation outside published reports/, for example local_evaluation/result.json.')
    try:
        known = set()
        for name in ['data/splits.csv', 'reports/dataset_inventory.csv']:
            path = ROOT / name
            if path.exists():
                with path.open(encoding='utf-8-sig', newline='') as source:
                    known.update(r.get('pixel_sha256') for r in csv.DictReader(source))
        report = evaluate(args.manifest, Predictor(), known)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as target:
            json.dump(report, target, indent=2, allow_nan=False)
            target.write('\n')
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Evaluation not saved: {exc}', file=sys.stderr)
        return 1
    print(f'Saved separate evaluation of {report["n"]} photos to {output}')
    if report['missing_classes']:
        print('Incomplete class coverage:', ', '.join(report['missing_classes']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
