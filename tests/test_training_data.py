import json

import numpy as np
import pytest

from audit_engine.training.data import (
    FEATURE_NAMES,
    build_dataset,
    load_dataset,
    validate_split_integrity,
)
from audit_engine.training.metrics import best_balanced_threshold, binary_metrics


def write_features(path, source_index: int) -> None:
    rows = []
    for index in range(64):
        pulse = float(index % 8 == source_index % 8)
        rows.append({
            "time": index * 0.25,
            "audio_energy_z": pulse * 3 - 0.2,
            "speech_ratio": 1.0,
            "pitch_hz": 120.0 + pulse * 10,
            "vocal_pulses": int(pulse),
            "body_motion_z": pulse * 2 - 0.1,
            "gesture_motion_z": pulse * 3 - 0.2,
            "camera_facing": 0.9,
            "torso_lean": 0.0,
            "shoulder_tilt": 0.0,
            "hand_count": 2.0,
        })
    path.parent.mkdir()
    path.write_text(json.dumps({
        "source": {"source": f"independent-source-{source_index}"},
        "timeline_resolution_seconds": 0.25,
        "fusion": {"timeline": rows},
    }), encoding="utf-8")


def test_dataset_is_balanced_and_group_split_is_leakage_safe(tmp_path):
    inputs = []
    for index in range(8):
        path = tmp_path / f"source-{index}" / "features.json"
        write_features(path, index)
        inputs.append(path)
    output = tmp_path / "dataset.npz"
    manifest = build_dataset(
        inputs,
        output,
        window_seconds=4.0,
        stride_seconds=1.0,
        offsets_seconds=(-1.0, -0.5, 0.5, 1.0),
        seed=11,
    )
    dataset = load_dataset(output)
    assert dataset["x"].shape[-1] == len(FEATURE_NAMES)
    assert manifest["positive_samples"] == manifest["negative_samples"]
    assert manifest["minimum_alignment_gain"] == 0.10
    assert "alignment_gain" in dataset
    assert np.all(dataset["alignment_gain"] >= 0.10 - 1e-6)
    assert np.array_equal(dataset["alignment_gain"][::2], dataset["alignment_gain"][1::2])
    assert set(manifest["split_offset_counts"]) == {"train", "val", "test"}
    assert len(manifest["split_groups"]["test"]) == 4
    assert validate_split_integrity(dataset)["passed"]
    assert set(dataset["split"].tolist()) == {"train", "val", "test"}


def test_explicit_split_plan_is_recorded_and_enforced(tmp_path):
    inputs = []
    for index in range(8):
        path = tmp_path / f"source-{index}" / "features.json"
        write_features(path, index)
        inputs.append(path)
    split_plan = {
        **{f"independent-source-{index}": "train" for index in range(4)},
        **{f"independent-source-{index}": "val" for index in range(4, 6)},
        **{f"independent-source-{index}": "test" for index in range(6, 8)},
    }

    manifest = build_dataset(
        inputs,
        tmp_path / "planned.npz",
        window_seconds=4.0,
        stride_seconds=1.0,
        offsets_seconds=(-1.0, -0.5, 0.5, 1.0),
        split_plan=split_plan,
    )

    assert manifest["split_strategy"] == "explicit_plan"
    assert {name: len(groups) for name, groups in manifest["split_groups"].items()} == {
        "train": 4,
        "val": 2,
        "test": 2,
    }
    with pytest.raises(ValueError, match="assign every input group exactly once"):
        build_dataset(inputs, tmp_path / "incomplete.npz", split_plan={"independent-source-0": "train"})

    wildcard_manifest = build_dataset(
        inputs,
        tmp_path / "wildcard.npz",
        window_seconds=4.0,
        stride_seconds=1.0,
        offsets_seconds=(-1.0, -0.5, 0.5, 1.0),
        split_plan={
            "*": "train",
            "independent-source-6": "val",
            "independent-source-7": "test",
        },
    )
    assert {name: len(groups) for name, groups in wildcard_manifest["split_groups"].items()} == {
        "train": 6,
        "val": 1,
        "test": 1,
    }


def test_binary_metrics_and_threshold_selection():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_balanced_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
