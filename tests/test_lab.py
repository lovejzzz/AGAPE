from audit_engine.lab import score_control, score_delay, summarize_calibration


def sample(alignment: float, lag: float, correlation: float = 0.3, facing: float = 0.8) -> dict:
    return {
        "fusion": {
            "summary": {
                "emphasis_motion_alignment_ratio": alignment,
                "median_absolute_gesture_lag_seconds": lag,
                "crossmodal_energy_motion_correlation": correlation,
                "camera_facing_median": facing,
            }
        }
    }


def test_repeatability_control_passes_equivalent_measurements():
    baseline = sample(0.5, 0.1)
    repeat = sample(0.5, 0.1)
    assert score_control(baseline, repeat)["passed"]


def test_known_delay_passes_when_alignment_weakens():
    baseline = sample(0.6, 0.05)
    delayed = sample(0.4, 0.15)
    result = score_delay(baseline, delayed, 0.5)
    assert result["passed"]
    assert result["signals"]["alignment_weakened"]


def test_calibration_requires_repeatability_and_two_thirds_sensitivity():
    control = {"passed": True}
    delays = [{"passed": True}, {"passed": True}, {"passed": False}]
    result = summarize_calibration(control, delays)
    assert result["overall_passed"]
    assert result["directional_sensitivity"] == 2 / 3
