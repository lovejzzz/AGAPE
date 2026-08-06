from copy import deepcopy

from audit_engine.comparison import compare_takes


def take(alignment: float, coverage: float = 0.95) -> dict:
    return {
        "title": "Practice take",
        "job": "/tmp/example",
        "coaching_context": "camera",
        "audio": {
            "summary": {
                "pause_ratio": 0.08,
                "pitch_variation_cents": 180.0,
            }
        },
        "vision": {
            "summary": {
                "face_coverage": coverage,
                "pose_coverage": coverage,
            }
        },
        "fusion": {
            "events": {
                "vocal_emphasis": [{"time": 2.0}, {"time": 5.0}, {"time": 8.0}],
                "gesture_phases": [{"stroke_time": 2.0}],
            },
            "summary": {
                "duration_seconds": 30.0,
                "vocal_emphasis_events": 5,
                "aligned_emphasis_events": 3,
                "emphasis_motion_alignment_ratio": alignment,
                "median_absolute_gesture_lag_seconds": 0.25,
                "camera_facing_median": 0.75,
                "competing_motion_events": 2,
            },
        },
    }


def test_comparison_marks_alignment_as_candidate_not_proven_improvement():
    first = take(0.25)
    second = take(0.75)
    result = compare_takes(first, second, "Paired practice")
    finding = next(item for item in result["findings"] if "landed together" in item["title"])
    assert finding["kind"] == "candidate_gain"
    assert "candidate" in finding["interpretation"].lower()
    assert result["quality"]["confidence"] == "high"


def test_low_coverage_never_claims_visual_candidate_gain():
    first = take(0.1, coverage=0.2)
    second = deepcopy(first)
    second["fusion"]["summary"]["emphasis_motion_alignment_ratio"] = 0.9
    result = compare_takes(first, second, "Low coverage")
    assert result["quality"]["confidence"] == "low"
    assert result["findings"][0]["kind"] == "quality"
    assert not any(item["kind"] == "candidate_gain" for item in result["findings"])
