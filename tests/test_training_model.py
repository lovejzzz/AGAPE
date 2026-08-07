import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from audit_engine.training.inference import score_features
from audit_engine.training.judge import holdout_consumption_path, reserve_holdout_evaluation
from audit_engine.training.metrics import best_group_macro_threshold, group_macro_metrics
from audit_engine.training.model import ModelConfig, TemporalSynchronyNet
from audit_engine.training.runner import group_balanced_pair_weights, validation_selection_score


def test_temporal_model_returns_one_logit_per_window():
    values = torch.randn(5, 32, 20)
    with_correlations = TemporalSynchronyNet(
        ModelConfig(input_features=20, hidden_size=16, blocks=2)
    )
    without_correlations = TemporalSynchronyNet(
        ModelConfig(input_features=20, hidden_size=16, blocks=2, explicit_correlations=False)
    )
    assert with_correlations(values).shape == (5,)
    assert without_correlations(values).shape == (5,)


def test_group_macro_threshold_and_pair_weights_treat_groups_equally():
    labels = np.array([1, 0, 1, 0, 1, 0])
    probabilities = np.array([0.9, 0.2, 0.8, 0.7, 0.6, 0.4])
    groups = np.array(["large", "large", "large", "large", "small", "small"])
    threshold = best_group_macro_threshold(labels, probabilities, groups)
    metrics = group_macro_metrics(labels, probabilities, groups, threshold)
    pair_groups = np.array(["large", "large", "large", "large", "small", "small"])

    assert metrics["groups"] == 2
    assert metrics["balanced_accuracy"] >= 0.75
    assert np.allclose(group_balanced_pair_weights(pair_groups), [0.5, 0.5, 1.0])


def test_gate_ready_validation_epoch_outranks_unseparated_epoch():
    ready = {
        "balanced_accuracy": 0.76,
        "auroc": 0.80,
        "mean_positive_score": 0.62,
        "mean_negative_score": 0.45,
    }
    unseparated = {
        "balanced_accuracy": 0.84,
        "auroc": 0.90,
        "mean_positive_score": 0.52,
        "mean_negative_score": 0.49,
    }
    assert validation_selection_score(ready)[0] > validation_selection_score(unseparated)[0]


def test_final_holdout_can_only_be_reserved_once(tmp_path):
    dataset = tmp_path / "fresh-final.npz"
    first_checkpoint = tmp_path / "candidate-a.pt"
    second_checkpoint = tmp_path / "candidate-b.pt"
    dataset.write_bytes(b"immutable final holdout")
    first_checkpoint.write_bytes(b"frozen candidate a")
    second_checkpoint.write_bytes(b"frozen candidate b")

    registry, record = reserve_holdout_evaluation(first_checkpoint, dataset)

    assert registry == holdout_consumption_path(dataset)
    assert record["evaluation_status"] == "started"
    assert json.loads(registry.read_text(encoding="utf-8"))["checkpoint"] == str(first_checkpoint)
    with pytest.raises(ValueError, match="already been consumed"):
        reserve_holdout_evaluation(second_checkpoint, dataset)


def test_promote_report_only_authorizes_its_exact_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"different checkpoint")
    (tmp_path / "judge.json").write_text(json.dumps({
        "verdict": "PROMOTE",
        "checkpoint": str(tmp_path / "last.pt"),
        "checkpoint_sha256": "not-this-checkpoint",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="has not received a PROMOTE verdict"):
        score_features(checkpoint, tmp_path / "features.json")
