"""Experiments with the primary set (synthetic). The published TrashNet study is not changed.

A. domain_shift: frozen TrashNet checkpoints evaluated on the primary set (CPU is enough).
B. fine_tune: MobileNetV2 trained on TrashNet + primary-set training images, evaluated once.

Run: python -m wastesort.primary_experiments --stage all
"""
import argparse
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .common import CLASSES, ROOT, digest, read_json, save_json
from .train import ImageDataset, get_hog, metrics, predict_network, seed_everything

REPORTS = ROOT / "reports/primary_set"
RESEARCH = ["svm", "random_forest", "cnn", "mobilenet", "cnn_weighted"]


def primary_frame():
    manifest = ROOT / "data/primary_set/splits.csv"
    if not manifest.exists():
        raise SystemExit("Prepare the primary set first: python -m wastesort.primary_set")
    if digest(manifest) != read_json(ROOT / "data/primary_set/provenance.json")["manifest_sha256"]:
        raise ValueError("Primary-set manifest changed since preparation.")
    return pd.read_csv(manifest)


def load_checkpoint(name):
    import torch
    from .models import build_network
    record = read_json(ROOT / f"reports/{name}_training.json") if name != "mobilenet_primary" \
        else read_json(ROOT / "models/primary_finetuned.json")
    path = ROOT / record["artifact"]
    if digest(path) != record["artifact_sha256"]:
        raise ValueError(f"Checkpoint changed: {name}")
    if record["kind"] == "classical":
        return record["kind"], joblib.load(path)
    model = build_network(record["kind"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return record["kind"], model.eval()


def predict(kind, model, frame):
    from torch.utils.data import DataLoader
    if kind == "classical":
        return model.predict(get_hog(frame))
    _, pred, _ = predict_network(model, DataLoader(ImageDataset(frame, kind), batch_size=64), "cpu")
    return pred


def confusion_plot(matrix, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.imshow(matrix, cmap="Oranges")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted material", ylabel="Actual material", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def domain_shift():
    """How well do models trained only on TrashNet photos recognise synthetic campus scenes?"""
    frame = primary_frame()
    test = frame.split.to_numpy() == "test"
    y = frame.label.map(CLASSES.index).to_numpy()
    published = pd.read_csv(ROOT / "reports/comparison.csv").set_index("model")
    rows, reports, out = [], {}, frame[["path", "label", "split", "item", "setting"]].copy()
    for name in RESEARCH:
        kind, model = load_checkpoint(name)
        pred = np.asarray(predict(kind, model, frame))
        everything, held_out = metrics(y, pred), metrics(y[test], pred[test])
        reports[name] = {"all_primary_images": everything, "primary_test_split": held_out}
        out[name] = [CLASSES[i] for i in pred]
        rows.append({"model": name, "role": published.loc[name, "role"],
                     "trashnet_test_macro_f1": published.loc[name, "test_macro_f1"],
                     "primary_accuracy": everything["accuracy"],
                     "primary_macro_f1": everything["macro_f1"],
                     "primary_trash_recall": everything["report"]["trash"]["recall"],
                     "primary_test_macro_f1": held_out["macro_f1"],
                     "macro_f1_change": everything["macro_f1"] - published.loc[name, "test_macro_f1"]})
        print(f"{name}: TrashNet test F1={published.loc[name, 'test_macro_f1']:.3f} -> "
              f"primary set F1={everything['macro_f1']:.3f}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS / "domain_shift.csv", index=False, lineterminator="\n")
    out.to_csv(REPORTS / "domain_shift_predictions.csv", index=False, lineterminator="\n")
    # Accuracy by scene setting shows which conditions the TrashNet model finds hardest.
    by_setting = (out.assign(correct=out.mobilenet == out.label).groupby("setting").correct.mean()
                  .sort_values().to_dict())
    save_json(REPORTS / "domain_shift.json", {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "images": int(len(frame)), "test_images": int(test.sum()),
        "models_unchanged": "Frozen TrashNet checkpoints; no training or selection on primary-set images.",
        "mobilenet_accuracy_by_setting": by_setting, "models": reports,
        "manifest_sha256": digest(ROOT / "data/primary_set/splits.csv"),
        "claim_limit": "Synthetic scenes only; not a measure of real campus accuracy."})
    confusion_plot(reports["mobilenet"]["all_primary_images"]["confusion_matrix"],
                   "MobileNetV2 (TrashNet only) | primary set (synthetic)",
                   REPORTS / "mobilenet_domain_shift_confusion.png")


def train_mobilenet(train, validation, protocol, device):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from .models import build_network
    seed_everything()
    model = build_network("mobilenet", pretrained=True).to(device)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(ImageDataset(train, "mobilenet", True), batch_size=protocol["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    held = DataLoader(ImageDataset(validation, "mobilenet"), batch_size=64, num_workers=0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    history, best, stale, best_epoch, state = [], -1, 0, 0, None
    for epoch in range(1, protocol["epochs"] + 1):
        model.train()
        model.features.eval()  # Frozen batchnorm statistics, as in the original study.
        total, n = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            n += len(images)
        ys, pred, _ = predict_network(model, held, device)
        score = metrics(ys, pred)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / n, "validation_macro_f1": score})
        print(f"mobilenet_primary epoch {epoch:02}: loss={total / n:.4f}, val F1={score:.4f}", flush=True)
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= protocol["patience"]:
            break
    return state, best, best_epoch, history


def fine_tune():
    import torch
    result = REPORTS / "finetune_evaluation.json"
    if result.exists():
        raise SystemExit("Fine-tune test results already exist; refusing repeated test evaluation.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    protocol = read_json(ROOT / "protocol.json")
    secondary = pd.read_csv(ROOT / "data/splits.csv").assign(dataset="secondary")
    if not (ROOT / secondary.path.iloc[0]).exists():
        raise SystemExit("TrashNet images are missing. Download them first (see notebook step 4).")
    primary = primary_frame().assign(dataset="primary")
    combined = pd.concat([secondary[["path", "label", "split", "dataset"]],
                          primary[["path", "label", "split", "dataset"]]], ignore_index=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    state, best, best_epoch, history = train_mobilenet(
        combined[combined.split == "train"], combined[combined.split == "validation"], protocol, device)
    file = ROOT / "models/mobilenet_primary.pt"
    torch.save(state, file)
    metadata = {"model_id": "mobilenet_primary", "kind": "mobilenet",
                "artifact": file.relative_to(ROOT).as_posix(), "artifact_sha256": digest(file),
                "classes": CLASSES, "validation_macro_f1": best, "best_epoch": best_epoch,
                "trained_on": "TrashNet train + primary set (synthetic) train",
                "selection_rule": "epoch with highest macro F1 on combined validation; test unseen",
                "training_seconds": time.perf_counter() - started, "device": device,
                "history": history, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": digest(ROOT / "protocol.json"),
                "primary_protocol_sha256": digest(ROOT / "protocol_primary.json")}
    save_json(ROOT / "models/primary_finetuned.json", metadata)
    print(f"Frozen before test: epoch {best_epoch}, combined validation F1 {best:.4f}", flush=True)
    results, rows = {}, []
    for name in ["mobilenet", "mobilenet_primary"]:
        kind, model = load_checkpoint(name)
        for key, frame in [("trashnet_test", secondary[secondary.split == "test"]),
                           ("primary_test", primary[primary.split == "test"])]:
            y = frame.label.map(CLASSES.index).to_numpy()
            report = metrics(y, np.asarray(predict(kind, model, frame)))
            results[f"{name}|{key}"] = report
            rows.append({"model": name, "test_set": key, "images": len(frame),
                         "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
                         "trash_recall": report["report"]["trash"]["recall"]})
            confusion_plot(report["confusion_matrix"], f"{name} | {key.replace('_', ' ')}",
                           REPORTS / f"confusion_{name}_on_{key}.png")
    pd.DataFrame(rows).to_csv(REPORTS / "finetune_comparison.csv", index=False, lineterminator="\n")
    save_json(result, {"evaluated_utc": datetime.now(timezone.utc).isoformat(),
                       "selected_before_test": metadata["frozen_utc"], "results": results,
                       "note": "The TrashNet test split was already used once for the original study. "
                               "This is a single new evaluation of a new model; no settings were "
                               "changed after seeing either test split.",
                       "claim_limit": "One seed; synthetic primary images; not campus accuracy."})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "domain_shift", "fine_tune"], default="all")
    args = parser.parse_args()
    if args.stage in {"all", "domain_shift"}:
        domain_shift()
    if args.stage in {"all", "fine_tune"}:
        fine_tune()


if __name__ == "__main__":
    main()
