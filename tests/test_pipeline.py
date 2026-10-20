import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from wastesort.common import CLASSES, ROOT, read_json, digest
from wastesort.data import group_hashes, make_split
from wastesort.models import hog_features, read_upload


def png(size=(80, 90), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, "PNG")
    return buf.getvalue()


def test_upload_validation_and_rgb():
    assert read_upload(png(mode="RGBA")).mode == "RGB"
    for payload in [b"not an image", b"", png((10, 10)), b"x" * (10 * 1024 * 1024 + 1)]:
        with pytest.raises(ValueError):
            read_upload(payload)
    gif = io.BytesIO()
    Image.new("RGB", (80, 80)).save(gif, "GIF")
    with pytest.raises(ValueError):
        read_upload(gif.getvalue())


def test_hog_matches_training_shape():
    features = hog_features(read_upload(png()))
    assert features.shape == (1764,)
    assert np.isfinite(features).all()


def test_transitive_near_duplicates():
    # 000 -> 001 -> 011 must remain together even if the endpoints differ by two bits.
    groups, _ = group_hashes([0, 1, 3, 255], distance=1)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_frozen_manifest_no_leakage():
    path = ROOT / "data/splits.csv"
    if not path.exists():
        pytest.skip("Dataset not prepared")
    frame = pd.read_csv(path)
    assert frame.path.is_unique
    assert frame.groupby("group").split.nunique().max() == 1
    assert frame.groupby("pixel_sha256").split.nunique().max() == 1
    for split in ["train", "validation", "test"]:
        assert set(frame[frame.split == split].label) == set(CLASSES)
    assert digest(path) == read_json(ROOT / "reports/data_audit.json")["manifest_sha256"]
    pd.testing.assert_frame_equal(make_split(frame.drop(columns="split")), frame)


def test_deployment_has_frozen_selection():
    path = ROOT / "models/deployment.json"
    if not path.exists():
        pytest.skip("Training not finished")
    deployment = read_json(path)
    assert digest(ROOT / deployment["artifact"]) == deployment["artifact_sha256"]
    primary = [read_json(ROOT / f"reports/{name}_training.json")
               for name in ["svm", "random_forest", "cnn", "mobilenet"]]
    winner = sorted(primary, key=lambda r: (-r["validation_macro_f1"], r["model_id"]))[0]
    assert deployment["model_id"] == winner["model_id"]
    assert deployment["protocol_sha256"] == digest(ROOT / "protocol.json")


def test_neural_input_preprocessing_consistent():
    from wastesort.models import transforms_for
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (180, 240, 3), dtype=np.uint8))
    for kind in ["cnn", "mobilenet"]:
        direct = transforms_for(kind)(image).numpy()
        cached = transforms_for(kind)(image.resize((160, 160), Image.Resampling.BILINEAR)).numpy()
        np.testing.assert_array_equal(direct, cached)
