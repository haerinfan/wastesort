"""Export Wednesday validation predictions from the frozen introductory models."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from .common import ROOT, CLASSES, digest, save_json
from .models import hog_features


def main():
    manifest = pd.read_csv(ROOT / "data/splits.csv")
    validation = manifest[manifest.split == "validation"].copy()
    features = []
    for path in validation.path:
        with Image.open(ROOT / path) as image:
            features.append(hog_features(image.convert("RGB")))
    features = np.stack(features)
    truth = np.array([CLASSES.index(label) for label in validation.label])
    rows, metrics = [], {}
    for name in ["knn_exercise", "decision_tree_exercise"]:
        record = json.loads((ROOT / f"reports/{name}_training.json").read_text())
        artifact = ROOT / record["artifact"]
        if digest(artifact) != record["artifact_sha256"]:
            raise ValueError(f"Changed model: {name}")
        predicted = joblib.load(artifact).predict(features)
        score = f1_score(truth, predicted, average="macro")
        if not np.isclose(score, record["validation_macro_f1"], atol=1e-12):
            raise ValueError(f"Validation result does not reproduce for {name}")
        for path, actual, guess in zip(validation.path, truth, predicted):
            rows.append(dict(model=name, split="validation", path=path,
                             actual=CLASSES[actual], predicted=CLASSES[guess], correct=bool(actual == guess)))
        matrix = confusion_matrix(truth, predicted, labels=range(len(CLASSES)))
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap="Greens", colorbar=False)
        ax.set_title(f"{name.replace('_', ' ').title()} — validation (n={len(truth)})")
        fig.tight_layout()
        fig.savefig(ROOT / f"reports/{name}_validation_confusion.png", dpi=160)
        plt.close(fig)
        metrics[name] = dict(n=len(truth), accuracy=accuracy_score(truth, predicted), macro_f1=score,
                             artifact_sha256=record["artifact_sha256"], confusion_matrix=matrix.tolist())
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "reports/intro_validation_predictions.csv", index=False)
    # First two paths per true class, selected without filtering on correctness.
    sample_paths = set(validation.groupby("label", sort=True).head(2).path)
    table[table.path.isin(sample_paths)].to_csv(ROOT / "reports/intro_sample_predictions.csv", index=False)
    save_json(ROOT / "reports/intro_validation_metrics.json", metrics)
    print(f"Exported {len(table)} predictions and two validation confusion matrices; frozen metrics reproduced.")


if __name__ == "__main__":
    main()
