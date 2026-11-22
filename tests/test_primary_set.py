import numpy as np
import pandas as pd
import pytest
from PIL import Image

import wastesort.primary_set as primary_set
from wastesort.common import CLASSES, ROOT
from wastesort.data import make_split
from wastesort.dataset_compare import image_statistics
from wastesort.generate_primary import crop_to_frame
from wastesort.models import Predictor


@pytest.fixture(scope="module")
def protocol():
    return primary_set.load_protocol()


def probabilities(label, value):
    rest = (1 - value) / (len(CLASSES) - 1)
    return {c: (value if c == label else rest) for c in CLASSES}


def test_prompt_plan_is_deterministic_and_covers_every_item_and_setting(protocol):
    first, second = primary_set.build_prompts(protocol), primary_set.build_prompts(protocol)
    assert first == second
    frame = pd.DataFrame(first)
    for label in CLASSES:
        # Enough candidates that quality control can reject freely and still hit the target.
        assert (frame.label == label).sum() >= protocol["quality_control"]["target_counts"][label] * 1.5
    assert frame.candidate_id.is_unique and frame.seed.is_unique
    for label in CLASSES:
        rows = frame[frame.label == label]
        assert set(rows.item) == set(protocol["items"][label])
        assert set(rows.setting) == set(protocol["settings"])
    # Articles from the condition and item must not double up ("a clean a jar").
    assert not frame.prompt.str.contains(r"\b(?:a|an) (?:a|an)\b").any()
    # Condition words must not repeat inside the item ("a crumpled crumpled wrapper").
    assert not frame.prompt.str.contains(r"\b(\w+) \1\b").any()


def test_target_counts_mirror_the_secondary_dataset(protocol):
    trashnet = pd.read_csv(ROOT / "data/splits.csv").label.value_counts().to_dict()
    assert protocol["quality_control"]["target_counts"] == trashnet


def test_prompt_plan_is_prefix_stable_when_more_candidates_are_needed(protocol):
    bigger = {**protocol, "generator": {**protocol["generator"], "candidate_multiplier": 2.5}}
    small, large = primary_set.build_prompts(protocol), primary_set.build_prompts(bigger)
    large_by_id = {r["candidate_id"]: r for r in large}
    assert all(large_by_id[r["candidate_id"]] == r for r in small)


def test_generated_images_use_the_trashnet_frame():
    framed = crop_to_frame(Image.new("RGB", (512, 512), "white"), (512, 384))
    assert framed.size == (512, 384)
    assert crop_to_frame(Image.new("RGB", (800, 400)), (512, 384)).size == (512, 384)


def test_quality_rule_keeps_clear_matches_and_rejects_mismatches(protocol):
    qc = protocol["quality_control"]
    assert primary_set.accept_candidate(probabilities("glass", 0.9), "glass", qc) is None
    wrong = probabilities("plastic", 0.8)
    reason = primary_set.accept_candidate(wrong, "glass", qc)
    assert isinstance(reason, str) and reason
    unsure = probabilities("glass", 1 / len(CLASSES))
    assert primary_set.accept_candidate(unsure, "glass", qc) is not None


def test_selection_hits_each_class_target_and_round_robins_items():
    frame = pd.DataFrame([{"label": label, "item": f"item{i % 3}", "candidate_id": f"{label}{i}"}
                          for label in CLASSES for i in range(30)])
    targets = {c: 3 * (k + 1) for k, c in enumerate(CLASSES)}
    chosen = primary_set.select_to_targets(frame, targets)
    assert chosen.label.value_counts().to_dict() == targets
    per_item = chosen.groupby(["label", "item"]).size()
    assert (per_item.groupby("label").max() - per_item.groupby("label").min()).max() == 0


def test_screen_rejects_duplicates_and_low_quality(tmp_path, monkeypatch, protocol):
    monkeypatch.setattr(primary_set, "ROOT", tmp_path)
    rows = []
    # Random-noise images have unrelated perceptual hashes; "a" and "b" share one seed.
    for name, seed, label, score in [("a", 1, "glass", 0.9), ("b", 1, "glass", 0.9),
                                     ("c", 2, "metal", 0.05), ("d", 3, "paper", 0.9)]:
        path = tmp_path / f"{name}.png"
        pixels = np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        row = {"candidate_id": f"ps_{label}_{name}", "label": label, "candidate_path": path.name}
        row.update({f"clip_{c}": v for c, v in probabilities(label if score > 0.5 else "glass", 0.9).items()})
        rows.append(row)
    rows.append({"candidate_id": "ps_paper_missing", "label": "paper", "candidate_path": "none.jpg",
                 **{f"clip_{c}": 1 / 6 for c in CLASSES}})
    kept, rejected = primary_set.screen(rows, protocol)
    reasons = {r["candidate_id"]: r["reason"] for r in rejected}
    assert [r["candidate_id"] for r in kept] == ["ps_glass_a", "ps_paper_d"]
    assert reasons["ps_glass_b"] == "exact_pixel_duplicate"
    assert reasons["ps_paper_missing"] == "not_generated"
    assert "ps_metal_c" in reasons


def test_primary_like_groups_never_cross_splits():
    rows = [{"path": f"{label}/{g}_{k}.jpg", "label": label, "group": f"{label}|item{g % 6}|setting{g // 6}"}
            for label in CLASSES for g in range(48) for k in range(2)]
    frame = make_split(pd.DataFrame(rows))
    assert frame.groupby("group").split.nunique().max() == 1
    for split in ["train", "validation", "test"]:
        assert set(frame[frame.split == split].label) == set(CLASSES)


def test_image_statistics_flag_white_backgrounds():
    white = image_statistics(Image.new("RGB", (512, 384), "white"))
    dark = image_statistics(Image.new("RGB", (512, 384), (40, 60, 30)))
    assert white["white_background_fraction"] == 1.0
    assert dark["white_background_fraction"] == 0.0
    assert white["brightness"] > dark["brightness"]


def test_predictor_only_loads_known_metadata():
    with pytest.raises(ValueError):
        Predictor("../secrets.json")
