from audit_engine.analysis import fuse


def test_joint_timeline_finds_alignment():
    audio = {
        "timeline": [
            {"time": float(i), "audio_energy": float(5 if i in {2, 5, 8} else 1), "speech_ratio": 1.0, "pitch_hz": 120.0, "vocal_pulses": 2}
            for i in range(10)
        ],
        "pauses": [],
        "summary": {"pause_ratio": 0.0, "pitch_variation_cents": 130.0},
    }
    vision = {
        "timeline": [
            {"time": float(i), "motion": float(3 if i in {2, 5, 8} else 0.1), "gesture_motion": float(3 if i in {2, 5, 8} else 0.1), "camera_facing": 0.9, "torso_lean": 0.0, "shoulder_tilt": 0.0, "hand_count": 2.0}
            for i in range(10)
        ],
        "summary": {"face_coverage": 1.0, "pose_coverage": 1.0, "hand_coverage": 1.0},
    }
    result = fuse(audio, vision, 10.0)
    assert result["summary"]["vocal_emphasis_events"] == 3
    assert result["summary"]["emphasis_motion_alignment_ratio"] == 1.0
    assert any(item["kind"] == "strength" for item in result["findings"])


def test_low_visual_coverage_suppresses_crossmodal_coaching():
    audio = {
        "timeline": [
            {"time": float(i), "audio_energy": float(5 if i in {8, 12, 16} else 1), "speech_ratio": 1.0, "pitch_hz": 120.0, "vocal_pulses": 2}
            for i in range(25)
        ],
        "pauses": [],
        "summary": {"pause_ratio": 0.0, "pitch_variation_cents": 130.0},
    }
    vision = {
        "timeline": [
            {"time": float(i), "motion": 0.1, "gesture_motion": 0.1, "camera_facing": 0.1, "torso_lean": 0.0, "shoulder_tilt": 0.0, "hand_count": 0.0}
            for i in range(25)
        ],
        "summary": {"face_coverage": 0.15, "pose_coverage": 0.2, "hand_coverage": 0.0},
    }
    result = fuse(audio, vision, 25.0)
    titles = {item["title"] for item in result["findings"]}
    assert "Reframe before interpreting body language" in titles
    assert "Reconnect before the important beat" not in titles
    assert "Give emphasis one visible landing point" not in titles
