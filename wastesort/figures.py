"""Create descriptive dataset and error-review figures from saved evidence."""
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import CLASSES, ROOT, read_json


def main():
    manifest = pd.read_csv(ROOT / "data/splits.csv")
    counts = pd.crosstab(manifest.label, manifest.split).reindex(CLASSES)
    ax = counts[["train", "validation", "test"]].plot.bar(
        stacked=True, color=["#2A795B", "#81AC98", "#D6E6DC"], figsize=(8, 4.5))
    ax.set(title="Usable TrashNet images by material and partition", xlabel="Material", ylabel="Images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ROOT / "reports/class_distribution.png", dpi=150)
    plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for label, ax in zip(CLASSES, axes.flat):
        row = manifest[(manifest.label == label) & (manifest.split == "train")].iloc[0]
        ax.imshow(Image.open(ROOT / row.path))
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Training examples | TrashNet, Thung and Yang")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/training_examples.png", dpi=120)
    plt.close(fig)
    if (ROOT / "reports/test_predictions.csv").exists():
        predictions = pd.read_csv(ROOT / "reports/test_predictions.csv")
        selected = read_json(ROOT / "models/deployment.json")["model_id"]
        wrong = predictions[predictions.label != predictions[selected]].copy()
        wrong.to_csv(ROOT / "reports/selected_model_errors.csv", index=False)
        fig, axes = plt.subplots(3, 4, figsize=(10, 7))
        for ax, (_, row) in zip(axes.flat, wrong.head(12).iterrows()):
            ax.imshow(Image.open(ROOT / row.path))
            ax.set_title(f"{row.label} → {row[selected]}", fontsize=10)
            ax.axis("off")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle("First 12 errors by manifest order | final test, no retuning")
        fig.tight_layout()
        fig.savefig(ROOT / "reports/error_examples.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
