from __future__ import annotations

import math

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (predicted == 1)))
    tn = int(np.sum((y_true == 0) & (predicted == 0)))
    fp = int(np.sum((y_true == 0) & (predicted == 1)))
    fn = int(np.sum((y_true == 1) & (predicted == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "samples": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "auroc": roc_auc(y_true, probabilities),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "mean_positive_score": float(np.mean(probabilities[y_true == 1])) if np.any(y_true == 1) else float("nan"),
        "mean_negative_score": float(np.mean(probabilities[y_true == 0])) if np.any(y_true == 0) else float("nan"),
    }


def best_balanced_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([
        np.linspace(0.1, 0.9, 81),
        np.asarray(probabilities, dtype=float),
    ]))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        score = binary_metrics(y_true, probabilities, float(threshold))["balanced_accuracy"]
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def group_macro_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray,
    threshold: float,
) -> dict:
    groups = sorted(set(np.asarray(group_ids).tolist()))
    per_group = {
        str(group): binary_metrics(
            np.asarray(y_true)[np.asarray(group_ids) == group],
            np.asarray(probabilities)[np.asarray(group_ids) == group],
            threshold,
        )
        for group in groups
    }
    balanced = [metrics["balanced_accuracy"] for metrics in per_group.values()]
    aurocs = [metrics["auroc"] for metrics in per_group.values() if math.isfinite(metrics["auroc"])]
    return {
        "groups": len(groups),
        "balanced_accuracy": float(np.mean(balanced)),
        "worst_group_balanced_accuracy": float(np.min(balanced)),
        "auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "per_group": per_group,
    }


def best_group_macro_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray,
) -> float:
    candidates = np.unique(np.concatenate([
        np.linspace(0.1, 0.9, 81),
        np.asarray(probabilities, dtype=float),
    ]))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        score = group_macro_metrics(
            y_true, probabilities, group_ids, float(threshold)
        )["balanced_accuracy"]
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
