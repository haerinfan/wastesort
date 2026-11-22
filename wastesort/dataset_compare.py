"""Compare the primary set (synthetic) with the secondary TrashNet dataset.

Run: python -m wastesort.dataset_compare  (needs downloaded TrashNet images in data/raw)
"""
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import sobel

from .common import CLASSES, ROOT, SEED, save_json

REPORTS = ROOT / "reports/primary_set"
NAMES = {"secondary": "Secondary: TrashNet", "primary": "Primary set (synthetic)"}
COLORS = {"secondary": "#2a6f97", "primary": "#e07a1f"}


def image_statistics(image):
    """Simple, explainable appearance measures on a 256 x 192 copy."""
    small = np.asarray(image.convert("RGB").resize((256, 192), Image.Resampling.BILINEAR), dtype=np.float32)
    gray = small.mean(axis=2) / 255
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    high, low = small.max(axis=2), small.min(axis=2)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1), 0)
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()),
            "saturation": float(saturation.mean()),
            "white_background_fraction": float((low > 215).mean()),
            "colorfulness": float(colorfulness / 255),
            "edge_density": float((sobel(gray) > 0.05).mean())}


def load_frames():
    secondary = pd.read_csv(ROOT / "data/splits.csv").assign(dataset="secondary")
    primary_manifest = ROOT / "data/primary_set/splits.csv"
    if not primary_manifest.exists():
        raise SystemExit("Prepare the primary set first: python -m wastesort.primary_set")
    primary = pd.read_csv(primary_manifest).assign(dataset="primary")
    missing = [p for p in secondary.path.head(5) if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("TrashNet images are missing. Download them first (see notebook step 4).")
    return pd.concat([secondary, primary], ignore_index=True)


def embeddings(paths):
    """ImageNet MobileNetV2 features; a fixed, untrained-on-waste view of image content."""
    import torch
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    from .models import transforms_for
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).eval()
    transform = transforms_for("mobilenet")
    features = []
    with torch.inference_mode():
        for start in range(0, len(paths), 64):
            batch = torch.stack([transform(Image.open(ROOT / p).convert("RGB")) for p in paths[start:start + 64]])
            pooled = model.features(batch).mean(dim=(2, 3))
            features.append(pooled.numpy())
    return np.concatenate(features)


def domain_separability(features, domain):
    """Cross-validated accuracy of telling the datasets apart; 50% means indistinguishable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    primary = np.flatnonzero(domain == "primary")
    secondary = rng.choice(np.flatnonzero(domain == "secondary"), size=len(primary), replace=False)
    index = np.concatenate([primary, secondary])
    scores = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
                             features[index], domain[index] == "primary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return {"balanced_sample_per_dataset": int(len(primary)), "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_folds": scores.tolist(),
            "interpretation": "Near 50% means the feature distributions overlap; near 100% means a "
                              "simple classifier can tell synthetic from TrashNet images (a domain gap)."}


def plots(frame, stats, features, domain):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Class balance as proportions, since the datasets differ in size.
    share = pd.crosstab(frame.label, frame.dataset, normalize="columns").reindex(CLASSES)
    counts = pd.crosstab(frame.label, frame.dataset).reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASSES))
    for offset, key in [(-0.2, "secondary"), (0.2, "primary")]:
        bars = ax.bar(x + offset, share[key] * 100, 0.38, label=NAMES[key], color=COLORS[key])
        ax.bar_label(bars, labels=counts[key].tolist(), fontsize=8, padding=2)
    ax.set(xticks=x, xticklabels=CLASSES, ylabel="Share of dataset (%)",
           title="Class balance (bar labels are image counts)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "class_distribution_comparison.png", dpi=150)
    plt.close(fig)
    measures = ["brightness", "contrast", "saturation", "white_background_fraction", "colorfulness", "edge_density"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, measure in zip(axes.flat, measures):
        for key in ["secondary", "primary"]:
            ax.hist(stats.loc[stats.dataset == key, measure], bins=30, density=True, alpha=0.55,
                    color=COLORS[key], label=NAMES[key])
        ax.set_title(measure.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Image appearance: primary set (synthetic) vs TrashNet")
    fig.tight_layout()
    fig.savefig(REPORTS / "image_statistics_comparison.png", dpi=150)
    plt.close(fig)
    points = PCA(2, random_state=SEED).fit_transform(features)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key in ["secondary", "primary"]:
        mask = domain == key
        ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.5, color=COLORS[key], label=NAMES[key])
    ax.set(title="MobileNetV2 ImageNet features, 2-D PCA", xlabel="PC 1", ylabel="PC 2")
    ax.legend(frameon=False, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "embedding_pca.png", dpi=150)
    plt.close(fig)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(CLASSES), 8, figsize=(13, 10))
    for row, label in enumerate(CLASSES):
        for half, key in enumerate(["secondary", "primary"]):
            paths = frame[(frame.dataset == key) & (frame.label == label)].path.to_numpy()
            for k, path in enumerate(rng.choice(paths, 4, replace=False)):
                ax = axes[row, half * 4 + k]
                ax.imshow(Image.open(ROOT / path).convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel(label, fontsize=10)
                if row == 0 and k == 0:
                    ax.set_title(NAMES[key], loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "sample_grid.png", dpi=110)
    plt.close(fig)


def main():
    frame = load_frames()
    print(f"Measuring {len(frame)} images...", flush=True)
    stats = pd.DataFrame([image_statistics(Image.open(ROOT / p)) for p in frame.path])
    stats.insert(0, "dataset", frame.dataset.to_numpy())
    stats.insert(1, "label", frame.label.to_numpy())
    summary = stats.drop(columns="label").groupby("dataset").agg(["mean", "std", "median"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reindex(["secondary", "primary"])
    summary.insert(0, "images", frame.dataset.value_counts().reindex(summary.index).to_numpy())
    summary.insert(0, "dataset_name", [NAMES[k] for k in summary.index])
    summary.to_csv(REPORTS / "dataset_statistics.csv", lineterminator="\n")
    print("Extracting MobileNetV2 features...", flush=True)
    domain = frame.dataset.to_numpy()
    features = embeddings(frame.path.tolist())
    separability = domain_separability(features, domain)
    centroid_gap = float(np.linalg.norm(features[domain == "primary"].mean(0) - features[domain == "secondary"].mean(0)))
    spread = float(np.linalg.norm(features[domain == "secondary"].std(0)))
    plots(frame, stats, features, domain)
    counts = pd.crosstab(frame.label, frame.dataset).reindex(CLASSES)
    secondary_counts, primary_counts = counts["secondary"], counts["primary"]
    save_json(REPORTS / "dataset_comparison.json", {
        "datasets": {k: {"name": NAMES[k], "images": int((domain == k).sum()),
                         "class_counts": counts[k].astype(int).to_dict(),
                         "imbalance_ratio_max_to_min": float(counts[k].max() / counts[k].min()),
                         "origin": "photographs by the TrashNet creators" if k == "secondary"
                                   else "AI-generated (synthetic) images, see docs/PRIMARY_DATASET.md"}
                     for k in ["secondary", "primary"]},
        "appearance_means": {k: {m: float(stats.loc[stats.dataset == k, m].mean())
                                 for m in stats.columns[2:]} for k in ["secondary", "primary"]},
        "domain_separability": separability,
        "feature_centroid_distance": centroid_gap,
        "feature_spread_secondary": spread,
        "trash_share": {"secondary": float(secondary_counts["trash"] / secondary_counts.sum()),
                        "primary": float(primary_counts["trash"] / primary_counts.sum())},
        "notes": ["Appearance measures use a 256 x 192 copy of each image.",
                  "Features come from ImageNet MobileNetV2 without waste training, so they describe "
                  "general visual content, not the deployed classifier's view.",
                  "Statistics describe the two collections; they do not show which is more realistic."],
    })
    print(summary.to_string(), flush=True)
    print(f"Domain separability (50% = indistinguishable): {separability['cv_accuracy_mean']:.1%}", flush=True)


if __name__ == "__main__":
    main()
