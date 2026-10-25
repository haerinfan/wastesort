from PIL import Image

from predict import classify


def test_batch_keeps_valid_predictions_when_another_file_is_invalid(tmp_path):
    valid = tmp_path / "item.png"
    invalid = tmp_path / "broken.png"
    Image.new("RGB", (64, 64), "red").save(valid)
    invalid.write_bytes(b"not an image")

    class Stub:
        def predict(self, image):
            return {"label": "paper", "model": "test", "scores": {"paper": 0.6}}

    rows = classify([invalid, valid, tmp_path / "missing.png"], Stub())
    assert rows[0]["error"]
    assert rows[1]["label"] == "paper"
    assert rows[1]["top_score"] == 0.6
    assert not rows[1]["error"]
    assert rows[2]["error"]
