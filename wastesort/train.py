"""Train on the frozen manifest and choose deployment using validation only."""
import argparse
import copy
import importlib.metadata
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .common import CLASSES, ROOT, SEED, digest, read_json, save_json
from .models import build_network, hog_features, transforms_for


def metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(range(6)), average="macro", zero_division=0)),
            "report": classification_report(y, pred, labels=list(range(6)), target_names=CLASSES,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(6))).tolist()}


def seed_everything():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_hog(frame):
    # Feature extraction is per image and has no fitted state. Scaling is fitted on train only.
    return np.stack([hog_features(ImageOps.exif_transpose(Image.open(ROOT / p)).convert("RGB"))
                     for p in frame.path])


def train_classical(frame, protocol):
    train = frame[frame.split == "train"]
    val = frame[frame.split == "validation"]
    started = time.perf_counter()
    x, vx = get_hog(train), get_hog(val)
    feature_seconds = time.perf_counter() - started
    y, vy = train.label.map(CLASSES.index).to_numpy(), val.label.map(CLASSES.index).to_numpy()
    candidates = {
        "svm": [make_pipeline(StandardScaler(), SVC(C=c, class_weight="balanced", random_state=SEED))
                for c in protocol["svm_C"]],
        "random_forest": [RandomForestClassifier(n_estimators=protocol["rf_trees"], max_depth=d,
                              class_weight="balanced", random_state=SEED, n_jobs=4)
                          for d in protocol["rf_depth"]],
        "knn_exercise": [make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))],
        "decision_tree_exercise": [DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=SEED)],
    }
    summary = []
    for name, estimators in candidates.items():
        best, best_score = None, -1
        trials, total = [], 0
        for estimator in estimators:
            tick = time.perf_counter()
            estimator.fit(x, y)
            seconds = time.perf_counter() - tick
            total += seconds
            report = metrics(vy, estimator.predict(vx))
            trials.append({"parameters": str(estimator), "training_seconds": seconds, "validation": report})
            if report["macro_f1"] > best_score:
                best, best_score = estimator, report["macro_f1"]
        file = ROOT / f"models/{name}.joblib"
        joblib.dump(best, file, compress=3)
        record = {"model_id": name, "kind": "classical", "artifact": file.relative_to(ROOT).as_posix(),
                  "artifact_sha256": digest(file), "validation_macro_f1": best_score,
                  "training_seconds": total, "shared_hog_extraction_seconds": feature_seconds,
                  "trials": trials, "research_candidate": not name.endswith("_exercise")}
        save_json(ROOT / f"reports/{name}_training.json", record)
        summary.append(record)
        print(f"{name}: validation macro F1={best_score:.4f}", flush=True)
    return summary


class ImageDataset:
    def __init__(self, frame, kind, training=False):
        self.rows = frame.to_dict("records")
        self.transform = transforms_for(kind, training)
        # Cache RGB images at the input size to avoid repeated JPEG decode.
        self.images = []
        for row in self.rows:
            with Image.open(ROOT / row["path"]) as image:
                self.images.append(ImageOps.exif_transpose(image).convert("RGB").resize(
                    (160, 160), Image.Resampling.BILINEAR))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.transform(self.images[index]), CLASSES.index(self.rows[index]["label"])


def predict_network(model, loader, device):
    import torch
    ys, ps, probabilities = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            scores = model(images.to(device)).softmax(1).cpu().numpy()
            ys.extend(labels.numpy().tolist())
            ps.extend(scores.argmax(1).tolist())
            probabilities.extend(scores.tolist())
    return np.array(ys), np.array(ps), np.array(probabilities)


def train_neural(frame, protocol):
    import torch
    from torch.utils.data import DataLoader
    from torch import nn
    summary = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, kind, weighted in [("cnn", "cnn", False), ("cnn_weighted", "cnn", True),
                                  ("mobilenet", "mobilenet", False)]:
        seed_everything()
        model = build_network(kind, pretrained=kind == "mobilenet").to(device)
        train_data = ImageDataset(frame[frame.split == "train"], kind, True)
        val_data = ImageDataset(frame[frame.split == "validation"], kind)
        # A dedicated generator gives identical order across the CNN ablation.
        generator = torch.Generator().manual_seed(SEED)
        loader = DataLoader(train_data, batch_size=protocol["batch_size"], shuffle=True,
                            generator=generator, num_workers=0)
        validation = DataLoader(val_data, batch_size=64, num_workers=0)
        counts = frame[frame.split == "train"].label.value_counts().reindex(CLASSES).to_numpy()
        weights = torch.tensor(counts.sum() / (6 * counts), dtype=torch.float32, device=device) if weighted else None
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
        history, best_score, stale, best_epoch, best_state = [], -1, 0, 0, None
        started = time.perf_counter()
        for epoch in range(1, protocol["epochs"] + 1):
            model.train()
            if kind == "mobilenet":
                model.features.eval()  # Frozen batchnorm statistics must not drift.
            loss_total, n = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(images)
                n += len(images)
            ys, pred, _ = predict_network(model, validation, device)
            report = metrics(ys, pred)
            score = report["macro_f1"]
            history.append({"epoch": epoch, "train_loss": loss_total / n,
                            "validation_macro_f1": score, "validation_accuracy": report["accuracy"]})
            print(f"{name} epoch {epoch:02}: loss={loss_total/n:.4f}, val F1={score:.4f}", flush=True)
            if score > best_score:
                best_score, stale, best_epoch = score, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= protocol["patience"]:
                break
        elapsed = time.perf_counter() - started
        file = ROOT / f"models/{name}.pt"
        torch.save(best_state, file)
        model.load_state_dict(best_state)
        ys, pred, _ = predict_network(model, validation, device)
        record = {"model_id": name, "kind": kind, "artifact": file.relative_to(ROOT).as_posix(),
                  "artifact_sha256": digest(file), "validation_macro_f1": best_score,
                  "training_seconds": elapsed, "best_epoch": best_epoch, "class_weighted": weighted,
                  "device": device, "history": history, "validation": metrics(ys, pred),
                  "research_candidate": name != "cnn_weighted"}
        save_json(ROOT / f"reports/{name}_training.json", record)
        summary.append(record)
        del model, train_data, val_data
        if device == "cuda":
            torch.cuda.empty_cache()
    return summary


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "classical", "neural"], default="all")
    args = parser.parse_args()
    if (ROOT / "reports/test_evaluation.json").exists():
        raise SystemExit("Test results already exist. Start a separately documented study before retraining.")
    protocol = read_json(ROOT / "protocol.json")
    frame = pd.read_csv(ROOT / "data/splits.csv")
    if digest(ROOT / "data/splits.csv") != read_json(ROOT / "reports/data_audit.json")["manifest_sha256"]:
        raise ValueError("Manifest changed since audit.")
    (ROOT / "models").mkdir(exist_ok=True)
    binding = {"protocol_sha256": digest(ROOT / "protocol.json"), "manifest_sha256": digest(ROOT / "data/splits.csv")}
    freeze = ROOT / "reports/training_binding.json"
    if freeze.exists() and read_json(freeze) != binding:
        raise ValueError("Protocol or split changed since training began.")
    save_json(freeze, binding)
    if args.stage in {"all", "classical"}:
        train_classical(frame, protocol)
    if args.stage in {"all", "neural"}:
        train_neural(frame, protocol)
    expected = ["svm", "random_forest", "cnn", "mobilenet", "cnn_weighted", "knn_exercise", "decision_tree_exercise"]
    records = [read_json(ROOT / f"reports/{n}_training.json") for n in expected
               if (ROOT / f"reports/{n}_training.json").exists()]
    if len(records) != len(expected):
        print("Partial training complete; deployment freezes only after all planned experiments finish.")
        return
    # Weighting is an auxiliary ablation; deployment competition contains exactly the four main approaches.
    choice = sorted([r for r in records if r["research_candidate"]],
                    key=lambda r: (-r["validation_macro_f1"], r["model_id"]))[0]
    destination = ROOT / ("models/deployed" + (".joblib" if choice["kind"] == "classical" else ".pt"))
    shutil.copyfile(ROOT / choice["artifact"], destination)
    save_json(ROOT / "models/deployment.json", {
        "model_id": choice["model_id"], "kind": choice["kind"],
        "artifact": destination.relative_to(ROOT).as_posix(), "artifact_sha256": digest(destination),
        "classes": CLASSES, "validation_macro_f1": choice["validation_macro_f1"],
        "selection_rule": "highest validation macro F1 among four primary models; test unseen",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), **binding,
    })
    import torch
    save_json(ROOT / "reports/environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_cuda": torch.version.cuda, "seed": SEED,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchvision", "scikit-learn", "scikit-image", "numpy", "pandas", "pillow"]},
    })
    print(f"Deployment frozen from validation: {choice['model_id']}", flush=True)


if __name__ == "__main__":
    main()
