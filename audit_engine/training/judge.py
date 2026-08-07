from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .data import load_dataset, sha256_file, validate_split_integrity
from .runner import _json_safe, evaluate_checkpoint


def _gate(name: str, passed: bool, observed, requirement: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def holdout_consumption_path(dataset_path: Path) -> Path:
    return dataset_path.expanduser().resolve().with_suffix(".holdout-consumption.json")


def reserve_holdout_evaluation(checkpoint_path: Path, dataset_path: Path) -> tuple[Path, dict]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    registry_path = holdout_consumption_path(dataset_path)
    record = {
        "kind": "agape_final_holdout_consumption",
        "schema_version": 1,
        "status": "consumed",
        "consumed_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_status": "started",
        "policy": "A final holdout is consumed when judging starts, even if evaluation is interrupted or fails.",
    }
    try:
        with registry_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    except FileExistsError as exc:
        try:
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {"registry": str(registry_path)}
        raise ValueError(
            "final holdout has already been consumed; use a newly recorded holdout dataset "
            f"instead of judging another checkpoint: {existing}"
        ) from exc
    return registry_path, record


def update_holdout_consumption(registry_path: Path, record: dict, **updates) -> None:
    record.update(updates)
    registry_path.write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")


def judge_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    output_directory: Path | None = None,
    requested_device: str = "auto",
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    consumption_path, consumption = reserve_holdout_evaluation(checkpoint_path, dataset_path)
    dataset = load_dataset(dataset_path)
    integrity = validate_split_integrity(dataset)
    metrics, probabilities, labels, checkpoint, selected = evaluate_checkpoint(
        checkpoint_path,
        dataset_path,
        "test",
        requested_device=requested_device,
    )
    dataset_digest = sha256_file(dataset_path)
    groups_total = len(set(dataset["group_id"].tolist()))
    test_groups = len(set(selected["group_id"].tolist()))
    contains_demo = bool(np.any(dataset.get("is_demo", np.zeros(len(dataset["y"]), dtype=bool))))
    positive_score = metrics["mean_positive_score"]
    negative_score = metrics["mean_negative_score"]
    margin = positive_score - negative_score
    validation_balanced = float(checkpoint["validation_metrics"]["balanced_accuracy"])
    gap = validation_balanced - metrics["balanced_accuracy"]

    delay_scores: dict[str, dict] = {}
    negative_delay_means: list[float] = []
    expected_delays = sorted({
        float(value) for value in dataset["offset_seconds"] if abs(float(value)) > 1e-6
    })
    for delay in sorted(set(float(value) for value in selected["offset_seconds"] if abs(value) > 1e-6)):
        mask = np.isclose(selected["offset_seconds"], delay)
        mean_score = float(np.mean(probabilities[mask]))
        negative_delay_means.append(mean_score)
        delay_scores[f"{delay:+.2f}"] = {
            "samples": int(np.sum(mask)),
            "mean_intact_probability": mean_score,
        }
    every_delay_lower = bool(negative_delay_means) and all(
        value <= positive_score - 0.05 for value in negative_delay_means
    )
    delay_coverage = all(
        int(np.sum(np.isclose(selected["offset_seconds"], delay))) >= 5
        for delay in expected_delays
    )

    gates = [
        _gate(
            "dataset_provenance",
            checkpoint.get("dataset_sha256") == dataset_digest,
            {"checkpoint": checkpoint.get("dataset_sha256"), "evaluated": dataset_digest},
            "Judge exactly the immutable dataset used for training.",
        ),
        _gate(
            "split_integrity",
            integrity["passed"],
            integrity,
            "No source group may cross train, validation, or test splits.",
        ),
        _gate(
            "real_data",
            not contains_demo,
            {"contains_demo_data": contains_demo},
            "Promotion requires locally analyzed real recordings, not generated smoke-test features.",
        ),
        _gate(
            "source_diversity",
            groups_total >= 8 and test_groups >= 4,
            {"all_sources": groups_total, "held_out_sources": test_groups},
            "At least 8 independent sources overall and 4 held-out sources.",
        ),
        _gate(
            "held_out_sample_size",
            metrics["samples"] >= 80 and min(metrics["positives"], metrics["negatives"]) >= 30,
            {"samples": metrics["samples"], "positives": metrics["positives"], "negatives": metrics["negatives"]},
            "At least 80 held-out windows with at least 30 examples per class.",
        ),
        _gate(
            "balanced_accuracy",
            metrics["balanced_accuracy"] >= 0.70,
            metrics["balanced_accuracy"],
            "Held-out balanced accuracy >= 0.70.",
        ),
        _gate(
            "ranking_quality",
            metrics["auroc"] >= 0.75,
            metrics["auroc"],
            "Held-out AUROC >= 0.75.",
        ),
        _gate(
            "score_separation",
            margin >= 0.12,
            margin,
            "Mean intact score exceeds mean shifted score by >= 0.12.",
        ),
        _gate(
            "generalization_gap",
            gap <= 0.15,
            gap,
            "Validation minus held-out balanced accuracy <= 0.15.",
        ),
        _gate(
            "delay_coverage",
            delay_coverage,
            {
                "expected_shifts": expected_delays,
                "held_out_counts": {
                    key: value["samples"] for key, value in delay_scores.items()
                },
            },
            "Every dataset shift must have at least 5 held-out examples.",
        ),
        _gate(
            "delay_response",
            every_delay_lower,
            {"intact_mean": positive_score, "by_shift": delay_scores},
            "Every represented shift must score at least 0.05 below intact windows on average.",
        ),
    ]
    passed = all(gate["passed"] for gate in gates)
    report = {
        "kind": "agape_checkpoint_judgment",
        "verdict": "PROMOTE" if passed else "DO_NOT_PROMOTE",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset": str(dataset_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "target_boundary": checkpoint["target_boundary"],
        "held_out_metrics": metrics,
        "validation_balanced_accuracy": validation_balanced,
        "validation_to_test_gap": gap,
        "intact_shifted_score_margin": margin,
        "delay_scores": delay_scores,
        "gates": gates,
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "holdout_consumption_record": str(consumption_path),
        "interpretation": (
            "This checkpoint cleared the offline synchrony promotion gates. It still does not measure personality, emotion, honesty, diagnosis, protected traits, or overall speaking quality."
            if passed else
            "Keep this checkpoint experimental. Fix the failed gates and rerun training; do not wire its scores into coaching output."
        ),
    }
    update_holdout_consumption(
        consumption_path,
        consumption,
        evaluation_status="completed",
        completed_at=datetime.now(UTC).isoformat(),
        verdict=report["verdict"],
        failed_gates=report["failed_gates"],
    )
    output_directory = (output_directory or checkpoint_path.parent).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "judge.json"
    markdown_path = output_directory / "judge.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    lines = [
        "# AGAPE checkpoint judgment",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        "",
        f"Held-out balanced accuracy: {metrics['balanced_accuracy']:.3f}",
        "",
        f"Held-out AUROC: {metrics['auroc']:.3f}",
        "",
        f"Intact/shifted score margin: {margin:.3f}",
        "",
        "## Gates",
        "",
    ]
    for gate in gates:
        mark = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- **{mark} — {gate['name']}**: {gate['requirement']}")
    lines.extend(["", report["interpretation"], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(markdown_path)
    return report
