from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from wastesort.common import ROOT, CLASSES
from wastesort.models import Predictor


def test_app_starts_without_exception():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=60)
    assert not app.exception
    assert len(app.tabs) == 3
    assert "WasteSort" in app.caption[0].value or "WASTESORT" in app.caption[0].value


def test_deployed_model_accepts_rgb_without_dataset():
    if not (ROOT / "models/deployment.json").exists():
        import pytest
        pytest.skip("No deployment model")
    # Synthetic input checks execution only; it is not an accuracy evaluation.
    output = Predictor().predict(Image.new("RGB", (256, 192), "#eeeeee"))
    assert output["label"] in CLASSES
    if output["scores"] is not None:
        assert set(output["scores"]) == set(CLASSES)
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in output["scores"].values())
        assert abs(sum(output["scores"].values()) - 1) < 1e-5
