"""One final evaluation after all model and deployment decisions have been frozen."""
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import CLASSES, ROOT, SEED, digest, read_json, save_json
from .models import build_network, hog_features, transforms_for
from .train import ImageDataset, get_hog, metrics, predict_network


def grouped_bootstrap(y, predictions, group_ids, n=2000):
    from sklearn.metrics import f1_score
    groups = np.unique(group_ids)
    positions = [np.flatnonzero(group_ids == group) for group in groups]
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name in predictions}
    recall_draws = {name: [] for name in predictions}
    for _ in range(n):
        idx = np.concatenate([positions[i] for i in rng.integers(0, len(groups), len(groups))])
        for name, pred in predictions.items():
            draws[name].append(f1_score(y[idx], pred[idx], labels=list(range(6)),
                                       average="macro", zero_division=0))
            trash = idx[y[idx] == 5]
            recall_draws[name].append(float(np.mean(pred[trash] == 5)) if len(trash) else float("nan"))
    def interval(values):
        return np.nanpercentile(values, [2.5, 97.5]).tolist()
    comparisons = {}
    for a, b in [("mobilenet", "cnn"), ("cnn_weighted", "cnn")]:
        comparisons[f"{a}_minus_{b}_macro_f1"] = interval(np.array(draws[a]) - np.array(draws[b]))
    comparisons["cnn_weighted_minus_cnn_trash_recall"] = interval(
        np.array(recall_draws["cnn_weighted"]) - np.array(recall_draws["cnn"]))
    return {"resamples": n, "unit": "perceptual-hash candidate group",
            "macro_f1_95_percent_intervals": {k: interval(v) for k, v in draws.items()},
            "paired_difference_95_percent_intervals": comparisons,
            "interpretation": "Descriptive conditional intervals for one fitted run; no formal significance claims."}


def main():
    result_file = ROOT / "reports/test_evaluation.json"
    if result_file.exists():
        raise SystemExit("Final test report already exists; refusing repeated test evaluation.")
    deployment = read_json(ROOT / "models/deployment.json")
    for key, path in [("protocol_sha256", ROOT / "protocol.json"), ("manifest_sha256", ROOT / "data/splits.csv")]:
        if deployment[key] != digest(path):
            raise ValueError(f"Frozen {key} no longer matches.")
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(4)
    frame = pd.read_csv(ROOT / "data/splits.csv")
    test = frame[frame.split == "test"].copy()
    y = test.label.map(CLASSES.index).to_numpy()
    x = get_hog(test)
    predictions, results, rows = {}, {}, []
    names = ["svm", "random_forest", "cnn", "mobilenet", "cnn_weighted"]
    for name in names:
        record = read_json(ROOT / f"reports/{name}_training.json")
        path = ROOT / record["artifact"]
        if record["artifact_sha256"] != digest(path):
            raise ValueError(f"Checkpoint changed: {name}")
        if record["kind"] == "classical":
            model = joblib.load(path)
            pred = model.predict(x)
            infer = lambda im: model.predict(hog_features(im)[None])
        else:
            model = build_network(record["kind"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            dataset = ImageDataset(test, record["kind"])
            _, pred, _ = predict_network(model, DataLoader(dataset, batch_size=64), "cpu")
            transform = transforms_for(record["kind"])
            def infer(im):
                with torch.inference_mode():
                    return model(transform(im)[None]).argmax(1)
        # Same CPU, decoded RGB input, preprocessing included, one sample per call.
        samples = [ImageOps.exif_transpose(Image.open(ROOT / p)).convert("RGB") for p in test.path.iloc[:30]]
        for im in samples[:3]:
            infer(im)
        timings = []
        for im in samples:
            start = time.perf_counter()
            infer(im)
            timings.append((time.perf_counter() - start) * 1000)
        report = metrics(y, pred)
        report.update({"model_size_mb": path.stat().st_size / 1024**2,
                       "inference_ms_mean": float(np.mean(timings)),
                       "inference_ms_median": float(np.median(timings)),
                       "training_seconds": record["training_seconds"],
                       "validation_macro_f1": record["validation_macro_f1"]})
        predictions[name] = pred
        results[name] = report
        rows.append({"model": name, "role": "ablation" if name == "cnn_weighted" else "primary",
                     "validation_macro_f1": record["validation_macro_f1"],
                     "test_accuracy": report["accuracy"], "test_macro_f1": report["macro_f1"],
                     "trash_recall": report["report"]["trash"]["recall"],
                     "training_seconds": report["training_seconds"],
                     "model_size_mb": report["model_size_mb"],
                     "inference_ms_mean": report["inference_ms_mean"]})
        print(f"{name}: test accuracy={report['accuracy']:.4f}, macro F1={report['macro_f1']:.4f}", flush=True)
    out = test[["path", "label", "group"]].copy()
    for name, pred in predictions.items():
        out[name] = [CLASSES[i] for i in pred]
    out.to_csv(ROOT / "reports/test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "reports/comparison.csv", index=False)
    save_json(result_file, {"evaluated_utc": datetime.now(timezone.utc).isoformat(),
                           "deployment_selected_before_test": deployment["model_id"],
                           "deployment_frozen_utc": deployment["frozen_utc"],
                           "test_images": len(test), "models": results,
                           "latency_protocol": "CPU, 4 Torch threads, 30 single-image calls, 3 warmups, preprocessing included; file decode excluded",
                           "protocol_sha256": deployment["protocol_sha256"],
                           "manifest_sha256": deployment["manifest_sha256"]})
    # Bootstrap uses saved final predictions; it does not retrain or select models.
    save_json(ROOT / "reports/uncertainty.json",
              grouped_bootstrap(y, predictions, test.group.to_numpy()))
    plots(results)


def plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, report in results.items():
        fig, ax = plt.subplots(figsize=(6.3, 5.2))
        matrix = np.array(report["confusion_matrix"])
        ax.imshow(matrix, cmap="Greens")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted material", ylabel="Actual material", title=f"{name} | held-out test")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max()/2 else "black")
        fig.tight_layout()
        fig.savefig(ROOT / f"reports/{name}_confusion.png", dpi=150)
        plt.close(fig)
    for name in ["cnn", "cnn_weighted", "mobilenet"]:
        history = pd.DataFrame(read_json(ROOT / f"reports/{name}_training.json")["history"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot(history.epoch, history.train_loss)
        axes[0].set(title=f"{name}: training loss", xlabel="Epoch", ylabel="Cross entropy")
        axes[1].plot(history.epoch, history.validation_macro_f1)
        axes[1].set(title="Validation macro F1", xlabel="Epoch", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(ROOT / f"reports/{name}_learning.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
