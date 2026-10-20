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


def test_result_survives_rerun_and_clears_when_photo_changes(monkeypatch):
    import io
    import streamlit as st
    import wastesort.models as models

    def upload(color):
        buffer = io.BytesIO()
        Image.new("RGB", (80, 90), color).save(buffer, "PNG")
        return buffer

    current = [upload("red")]
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: current[0])

    class StubPredictor:
        def predict(self, image):
            return {"label": "plastic", "scores": None, "model": "test"}

    monkeypatch.setattr(models, "Predictor", StubPredictor)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=60)
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state["prediction_result"]["label"] == "plastic"
        # A widget rerun must retain the current image's result.
        app.selectbox[0].set_value("cnn").run()
        assert not app.exception
        assert app.session_state["prediction_result"]["label"] == "plastic"
        assert any(m.value == "## Plastic" for m in app.markdown)
        current[0] = upload("blue")
        app.run()
        assert not app.exception
        assert "prediction_result" not in app.session_state
        assert not any(m.value == "## Plastic" for m in app.markdown)
        current[0] = None
        app.run()
        assert not app.exception
        assert app.session_state["prediction_upload"] is None
    finally:
        st.cache_resource.clear()
