from __future__ import annotations

import math


def value(summary: dict, key: str) -> float:
    try:
        result = float(summary.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def delta(before: float, after: float) -> float:
    return after - before if math.isfinite(before) and math.isfinite(after) else float("nan")


def diagnostic_metrics(baseline: dict, variant: dict) -> dict:
    first = baseline["fusion"]["summary"]
    second = variant["fusion"]["summary"]
    return {
        "alignment_before": value(first, "emphasis_motion_alignment_ratio"),
        "alignment_after": value(second, "emphasis_motion_alignment_ratio"),
        "alignment_delta": delta(
            value(first, "emphasis_motion_alignment_ratio"),
            value(second, "emphasis_motion_alignment_ratio"),
        ),
        "matched_lag_before": value(first, "median_absolute_gesture_lag_seconds"),
        "matched_lag_after": value(second, "median_absolute_gesture_lag_seconds"),
        "matched_lag_delta": delta(
            value(first, "median_absolute_gesture_lag_seconds"),
            value(second, "median_absolute_gesture_lag_seconds"),
        ),
        "correlation_before": value(first, "crossmodal_energy_motion_correlation"),
        "correlation_after": value(second, "crossmodal_energy_motion_correlation"),
        "correlation_delta": delta(
            value(first, "crossmodal_energy_motion_correlation"),
            value(second, "crossmodal_energy_motion_correlation"),
        ),
        "facing_before": value(first, "camera_facing_median"),
        "facing_after": value(second, "camera_facing_median"),
        "facing_delta": delta(
            value(first, "camera_facing_median"),
            value(second, "camera_facing_median"),
        ),
    }


def score_control(baseline: dict, repeat: dict) -> dict:
    metrics = diagnostic_metrics(baseline, repeat)
    checks = {
        "alignment_repeatable": math.isfinite(metrics["alignment_delta"]) and abs(metrics["alignment_delta"]) <= 0.05,
        "lag_repeatable": (
            not math.isfinite(metrics["matched_lag_delta"])
            or abs(metrics["matched_lag_delta"]) <= 0.05
        ),
        "facing_repeatable": (
            not math.isfinite(metrics["facing_delta"])
            or abs(metrics["facing_delta"]) <= 0.02
        ),
    }
    return {
        "kind": "repeatability_control",
        "expected_delay_seconds": 0.0,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "explanation": "The same raw sample should produce materially equivalent measurements.",
    }


def score_delay(baseline: dict, variant: dict, expected_delay_seconds: float) -> dict:
    metrics = diagnostic_metrics(baseline, variant)
    signals = {
        "alignment_weakened": (
            math.isfinite(metrics["alignment_delta"]) and metrics["alignment_delta"] <= -0.08
        ),
        "matched_lag_increased": (
            math.isfinite(metrics["matched_lag_delta"]) and metrics["matched_lag_delta"] >= 0.08
        ),
        "energy_motion_correlation_weakened": (
            math.isfinite(metrics["correlation_delta"]) and metrics["correlation_delta"] <= -0.08
        ),
    }
    evidence_count = sum(signals.values())
    return {
        "kind": "visual_delay",
        "expected_delay_seconds": float(expected_delay_seconds),
        "metrics": metrics,
        "signals": signals,
        "evidence_count": evidence_count,
        "passed": evidence_count >= 1,
        "explanation": "A delayed visible performance should weaken at least one synchronized voice–movement signal.",
    }


def summarize_calibration(control: dict, delay_tests: list[dict]) -> dict:
    passed_delays = sum(item["passed"] for item in delay_tests)
    sensitivity = passed_delays / max(len(delay_tests), 1)
    overall = bool(control["passed"] and sensitivity + 1e-9 >= 2 / 3)
    return {
        "repeatability_passed": bool(control["passed"]),
        "delay_tests_passed": passed_delays,
        "delay_tests_total": len(delay_tests),
        "directional_sensitivity": sensitivity,
        "overall_passed": overall,
        "criterion": "repeatability control passes and at least two-thirds of known delays are detected",
    }
