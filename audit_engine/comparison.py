from __future__ import annotations

import math


def number(value, fallback: float = float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def percent_change(before: float, after: float) -> float:
    if not math.isfinite(before) or abs(before) < 1e-9 or not math.isfinite(after):
        return float("nan")
    return (after - before) / abs(before)


def quality(data: dict) -> float:
    vision = data["vision"]["summary"]
    return min(number(vision["face_coverage"], 0.0), number(vision["pose_coverage"], 0.0))


def rate(count: float, duration: float) -> float:
    return number(count, 0.0) / max(number(duration, 0.0), 1e-6) * 60.0


def metric(key: str, label: str, before: float, after: float, unit: str, reliable: bool) -> dict:
    before = number(before)
    after = number(after)
    return {
        "key": key,
        "label": label,
        "take_1": before,
        "take_2": after,
        "delta": after - before if math.isfinite(before) and math.isfinite(after) else float("nan"),
        "unit": unit,
        "reliable": bool(reliable and math.isfinite(before) and math.isfinite(after)),
    }


def event_labels(data: dict, key: str, limit: int = 3) -> list[str]:
    events = data.get("fusion", {}).get("events", {}).get(key, [])[:limit]
    labels = []
    for item in events:
        seconds = number(item.get("time", item.get("stroke_time")))
        if not math.isfinite(seconds):
            continue
        minutes = int(seconds // 60)
        labels.append(f"{minutes}:{seconds - minutes * 60:05.2f}")
    return labels


def compare_takes(take_1: dict, take_2: dict, title: str) -> dict:
    first = take_1["fusion"]["summary"]
    second = take_2["fusion"]["summary"]
    audio_1 = take_1["audio"]["summary"]
    audio_2 = take_2["audio"]["summary"]
    duration_1 = number(first["duration_seconds"], 0.0)
    duration_2 = number(second["duration_seconds"], 0.0)
    duration_ratio = min(duration_1, duration_2) / max(duration_1, duration_2, 1e-6)
    visual_quality = min(quality(take_1), quality(take_2))
    content_comparability = duration_ratio >= 0.65
    visual_reliable = visual_quality >= 0.55
    if visual_quality >= 0.8 and duration_ratio >= 0.8:
        confidence = "high"
    elif visual_reliable and content_comparability:
        confidence = "medium"
    else:
        confidence = "low"

    emphasis_reliable = (
        visual_reliable
        and first["vocal_emphasis_events"] >= 3
        and second["vocal_emphasis_events"] >= 3
    )
    facing_reliable = (
        take_1.get("coaching_context") == "camera"
        and take_2.get("coaching_context") == "camera"
        and min(
            take_1["vision"]["summary"]["face_coverage"],
            take_2["vision"]["summary"]["face_coverage"],
        ) >= 0.55
    )

    metrics = [
        metric(
            "alignment",
            "Voice–gesture event alignment",
            first["emphasis_motion_alignment_ratio"],
            second["emphasis_motion_alignment_ratio"],
            "ratio",
            emphasis_reliable,
        ),
        metric(
            "gesture_lag",
            "Median absolute voice–gesture lag",
            first.get("median_absolute_gesture_lag_seconds"),
            second.get("median_absolute_gesture_lag_seconds"),
            "seconds",
            emphasis_reliable and first["aligned_emphasis_events"] >= 2 and second["aligned_emphasis_events"] >= 2,
        ),
        metric(
            "pause_ratio",
            "Long-pause share",
            audio_1["pause_ratio"],
            audio_2["pause_ratio"],
            "ratio",
            content_comparability,
        ),
        metric(
            "pitch_variation",
            "Pitch variation",
            audio_1["pitch_variation_cents"],
            audio_2["pitch_variation_cents"],
            "cents",
            content_comparability,
        ),
        metric(
            "camera_facing",
            "Camera-facing head-direction proxy",
            first.get("camera_facing_median"),
            second.get("camera_facing_median"),
            "ratio",
            facing_reliable,
        ),
        metric(
            "competing_motion_rate",
            "Competing body-motion events per minute",
            rate(first["competing_motion_events"], duration_1),
            rate(second["competing_motion_events"], duration_2),
            "events/min",
            visual_reliable and content_comparability,
        ),
    ]
    by_key = {item["key"]: item for item in metrics}
    findings: list[dict] = []

    if not visual_reliable:
        findings.append({
            "kind": "quality",
            "title": "The visual comparison is constrained",
            "observation": f"Joint face-and-body coverage fell to {visual_quality:.0%} in at least one take.",
            "interpretation": "Audio changes can still be compared, but visible delivery changes should not be treated as reliable.",
            "experiment": "Record both takes with the same stable camera position and keep face, shoulders, and hands visible.",
            "confidence": "high",
        })
    if not content_comparability:
        findings.append({
            "kind": "quality",
            "title": "The takes may not contain equivalent material",
            "observation": f"The shorter take is only {duration_ratio:.0%} of the longer take.",
            "interpretation": "Some changes may reflect different content rather than delivery.",
            "experiment": "Compare the same words and approximately the same duration in both takes.",
            "confidence": "high",
        })

    alignment = by_key["alignment"]
    if alignment["reliable"] and abs(alignment["delta"]) >= 0.12:
        up = alignment["delta"] > 0
        before_marks = ", ".join(event_labels(take_1, "vocal_emphasis")) or "Take 1 emphasis events"
        after_marks = ", ".join(event_labels(take_2, "vocal_emphasis")) or "Take 2 emphasis events"
        findings.append({
            "kind": "candidate_gain" if up else "candidate_tradeoff",
            "title": "Voice and gesture landed together more often" if up else "Voice and gesture separated more often",
            "observation": f"Event alignment changed from {alignment['take_1']:.0%} to {alignment['take_2']:.0%}.",
            "interpretation": "This is a candidate gain in visible emphasis, not a universal quality score." if up else "This may make emphasis less visibly explicit, though restraint can be intentional.",
            "experiment": f"Replay Take 1 near {before_marks} and Take 2 near {after_marks}; keep the version whose important beats feel more intentional.",
            "confidence": confidence,
        })

    lag = by_key["gesture_lag"]
    if lag["reliable"] and abs(lag["delta"]) >= 0.08:
        closer = lag["delta"] < 0
        findings.append({
            "kind": "candidate_gain" if closer else "change",
            "title": "Gesture timing moved closer to the vocal beat" if closer else "Gesture timing moved farther from the vocal beat",
            "observation": f"Median absolute lag changed from {lag['take_1']:.2f}s to {lag['take_2']:.2f}s.",
            "interpretation": "Closer timing usually makes a deliberately marked beat easier to perceive, but anticipatory gestures can also be purposeful.",
            "experiment": "Try one more take with the gesture stroke deliberately on the stressed syllable, then compare the measured lag and your own perception.",
            "confidence": confidence,
        })

    pauses = by_key["pause_ratio"]
    if pauses["reliable"] and abs(pauses["delta"]) >= 0.02:
        direction = "increased" if pauses["delta"] > 0 else "decreased"
        findings.append({
            "kind": "change",
            "title": f"Long-pause space {direction}",
            "observation": f"Detected long-pause share changed from {pauses['take_1']:.1%} to {pauses['take_2']:.1%}.",
            "interpretation": "This is a pacing change, not automatically an improvement; its value depends on whether transitions became easier to follow.",
            "experiment": "Listen without watching and choose which take gives the clearest boundaries between ideas.",
            "confidence": confidence,
        })

    facing = by_key["camera_facing"]
    if facing["reliable"] and abs(facing["delta"]) >= 0.05:
        up = facing["delta"] > 0
        findings.append({
            "kind": "candidate_gain" if up else "change",
            "title": "Head direction stayed closer to the camera zone" if up else "Head direction moved farther from the camera zone",
            "observation": f"The camera-facing proxy changed from {facing['take_1']:.0%} to {facing['take_2']:.0%}.",
            "interpretation": "This may change perceived directness, but it is a head-direction proxy—not verified eye contact.",
            "experiment": "Watch both takes muted and decide which head orientation feels appropriately direct for the intended audience.",
            "confidence": confidence,
        })

    pitch = by_key["pitch_variation"]
    pitch_delta = percent_change(pitch["take_1"], pitch["take_2"])
    if pitch["reliable"] and math.isfinite(pitch_delta) and abs(pitch_delta) >= 0.15:
        direction = "widened" if pitch_delta > 0 else "narrowed"
        findings.append({
            "kind": "change",
            "title": f"Vocal pitch contrast {direction}",
            "observation": f"Speaker-relative pitch variation changed by {abs(pitch_delta):.0%}.",
            "interpretation": "More variation is not always better; the useful question is whether contrast supports the intended delivery.",
            "experiment": "Listen to both takes with the video hidden and choose which vocal contour makes the key beats easier to hear.",
            "confidence": confidence,
        })

    competing = by_key["competing_motion_rate"]
    if competing["reliable"] and abs(competing["delta"]) >= 1.0:
        lower = competing["delta"] < 0
        findings.append({
            "kind": "candidate_gain" if lower else "candidate_tradeoff",
            "title": "Unmatched body motion decreased" if lower else "Unmatched body motion increased",
            "observation": f"Low-vocal-energy motion changed from {competing['take_1']:.1f} to {competing['take_2']:.1f} events per minute.",
            "interpretation": "Lower rates can make deliberate emphasis easier to see, while higher rates may fit an intentionally animated style.",
            "experiment": "Watch both takes muted and choose whether the movement pattern clarifies or competes with the intended emphasis.",
            "confidence": confidence,
        })

    if not findings:
        findings.append({
            "kind": "stable",
            "title": "No material delivery change cleared the confidence gates",
            "observation": "The measured differences were small or insufficiently supported.",
            "interpretation": "The second take is behaviorally similar under AGAPE's current observable signals.",
            "experiment": "Change only one variable—pause placement, gesture timing, or camera orientation—and record a third take.",
            "confidence": confidence,
        })

    return {
        "engine": "AGAPE",
        "engine_full_name": "Audiovisual Gesture And Prosody Engine",
        "engine_version": "0.2.0",
        "title": title,
        "transcript_used": False,
        "comparison_boundary": "Same speaker and substantially the same content are assumed. Candidate gains require human confirmation.",
        "quality": {
            "confidence": confidence,
            "joint_visual_coverage": visual_quality,
            "duration_similarity": duration_ratio,
            "visual_reliable": visual_reliable,
            "content_comparable": content_comparability,
        },
        "take_1": {"title": take_1["title"], "job": take_1.get("job")},
        "take_2": {"title": take_2["title"], "job": take_2.get("job")},
        "metrics": metrics,
        "findings": findings[:6],
    }
