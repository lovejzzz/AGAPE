from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import librosa
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

from .config import AuditConfig


def finite(value: float | np.floating | None, fallback: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return fallback
    return float(value)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.zeros_like(values)
    if valid.sum() < 2:
        return result
    std = values[valid].std()
    if std < 1e-8:
        return result
    result[valid] = (values[valid] - values[valid].mean()) / std
    return result


def robust_baseline(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": float("nan"), "mad": float("nan")}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {"median": median, "mad": mad}


def contiguous_regions(mask: np.ndarray, times: np.ndarray, minimum_seconds: float) -> list[dict]:
    if not len(mask):
        return []
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    regions = []
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    for start, end in zip(starts, ends, strict=False):
        start_time = float(times[min(start, len(times) - 1)])
        end_time = float(times[min(end - 1, len(times) - 1)] + step)
        if end_time - start_time >= minimum_seconds:
            regions.append({"start": start_time, "end": end_time, "duration": end_time - start_time})
    return regions


def analyze_audio(path: Path, config: AuditConfig, duration: float) -> dict:
    y, sr = librosa.load(path, sr=config.audio_sample_rate, mono=True)
    frame_length = 1024
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    smoothed = np.convolve(rms, np.ones(7) / 7, mode="same")
    p20, p90 = np.percentile(smoothed, [20, 90])
    threshold = max(float(p20 + 0.22 * (p90 - p20)), 1e-4)
    speech = smoothed > threshold
    pauses = contiguous_regions(~speech, times, minimum_seconds=0.45)

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=65, fmax=420, sr=sr, frame_length=2048, hop_length=hop,
        )
    except Exception:
        f0 = np.full(len(rms), np.nan)
        voiced_flag = np.zeros(len(rms), dtype=bool)
    if len(f0) != len(rms):
        f0 = np.interp(times, np.linspace(0, duration, len(f0)), np.nan_to_num(f0, nan=0.0))
        f0[f0 <= 0] = np.nan

    peak_distance = max(1, int(0.14 * sr / hop))
    prominence = max(float(np.std(smoothed) * 0.28), 1e-5)
    peaks, _ = find_peaks(smoothed, distance=peak_distance, prominence=prominence)
    peak_times = times[peaks]

    window = config.timeline_window_seconds
    bin_starts = np.arange(0.0, max(duration, window), window)
    rows = []
    for start in bin_starts:
        end = min(start + window, duration)
        mask = (times >= start) & (times < end)
        pitch_values = f0[mask] if len(f0) == len(times) else np.array([])
        rows.append({
            "time": float(start),
            "audio_energy": finite(np.mean(smoothed[mask]) if mask.any() else 0.0),
            "speech_ratio": finite(np.mean(speech[mask]) if mask.any() else 0.0),
            "pitch_hz": finite(np.nanmedian(pitch_values) if np.isfinite(pitch_values).any() else None, fallback=float("nan")),
            "vocal_pulses": int(np.sum((peak_times >= start) & (peak_times < end))),
        })

    voiced_pitch = f0[np.isfinite(f0)]
    pitch_cents_std = 0.0
    if len(voiced_pitch) > 4:
        pitch_cents_std = finite(np.std(1200 * np.log2(voiced_pitch / np.median(voiced_pitch))))
    pause_seconds = sum(item["duration"] for item in pauses)
    active_seconds = max(duration - pause_seconds, 1e-6)
    return {
        "timeline": rows,
        "pauses": pauses,
        "summary": {
            "speech_ratio": finite(np.mean(speech)),
            "pause_ratio": finite(pause_seconds / max(duration, 1e-6)),
            "pause_count": len(pauses),
            "median_pitch_hz": finite(np.median(voiced_pitch) if len(voiced_pitch) else None, fallback=float("nan")),
            "pitch_variation_cents": pitch_cents_std,
            "vocal_pulses_per_minute_active": finite(len(peaks) / active_seconds * 60.0),
            "energy_dynamic_ratio": finite(p90 / max(p20, 1e-5)),
        },
    }


@dataclass
class VisionRow:
    time: float
    face_detected: float
    pose_detected: float
    hand_count: float
    head_yaw_proxy: float
    head_pitch_proxy: float
    camera_facing: float
    shoulder_tilt: float
    torso_lean: float
    motion: float
    gesture_motion: float
    mouth_open: float


def point(landmarks: list, index: int) -> np.ndarray | None:
    if len(landmarks) <= index:
        return None
    item = landmarks[index]
    return np.array([float(item.x), float(item.y), float(getattr(item, "z", 0.0))], dtype=float)


def analyze_vision(path: Path, config: AuditConfig, duration: float) -> dict:
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.45,
        min_face_landmarks_confidence=0.45,
        min_pose_detection_confidence=0.45,
        min_pose_landmarks_confidence=0.45,
        min_hand_landmarks_confidence=0.4,
    )
    cap = cv2.VideoCapture(str(path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or config.proxy_fps
    sample_every = max(1, int(round(source_fps / config.analysis_fps)))
    frame_index = 0
    rows: list[VisionRow] = []
    previous_vector: np.ndarray | None = None
    previous_wrists: np.ndarray | None = None

    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every:
                frame_index += 1
                continue
            timestamp = frame_index / source_fps
            if timestamp > duration:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(round(timestamp * 1000)))
            face = list(result.face_landmarks or [])
            pose = list(result.pose_landmarks or [])
            left_hand = list(result.left_hand_landmarks or [])
            right_hand = list(result.right_hand_landmarks or [])

            face_detected = float(len(face) >= 153)
            pose_detected = float(len(pose) >= 25)
            head_yaw = float("nan")
            head_pitch = float("nan")
            camera_facing = float("nan")
            mouth_open = float("nan")
            if face_detected:
                left_eye, right_eye, nose, chin, brow = (point(face, i) for i in (33, 263, 1, 152, 10))
                upper_lip, lower_lip = point(face, 13), point(face, 14)
                eye_mid = (left_eye + right_eye) / 2
                eye_distance = max(float(np.linalg.norm(left_eye[:2] - right_eye[:2])), 1e-6)
                face_height = max(float(np.linalg.norm(brow[:2] - chin[:2])), 1e-6)
                head_yaw = finite((nose[0] - eye_mid[0]) / eye_distance, fallback=float("nan"))
                head_pitch = finite((nose[1] - eye_mid[1]) / face_height, fallback=float("nan"))
                yaw_score = 1.0 - min(abs(head_yaw) / 0.62, 1.0)
                pitch_score = 1.0 - min(abs(head_pitch - 0.38) / 0.45, 1.0)
                camera_facing = finite(0.72 * yaw_score + 0.28 * pitch_score, fallback=float("nan"))
                mouth_open = finite(abs(lower_lip[1] - upper_lip[1]) / face_height, fallback=float("nan"))

            shoulder_tilt = float("nan")
            torso_lean = float("nan")
            movement_vector = None
            wrist_vector = None
            if pose_detected:
                ls, rs, lh, rh, lw, rw = (point(pose, i) for i in (11, 12, 23, 24, 15, 16))
                shoulder_width = max(float(np.linalg.norm(ls[:2] - rs[:2])), 1e-6)
                shoulder_tilt = finite(math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])), fallback=float("nan"))
                shoulder_mid = (ls + rs) / 2
                hip_mid = (lh + rh) / 2
                torso_lean = finite((shoulder_mid[0] - hip_mid[0]) / shoulder_width, fallback=float("nan"))
                movement_vector = np.concatenate([ls[:2], rs[:2], lh[:2], rh[:2]]) / shoulder_width
                wrist_vector = np.concatenate([lw[:2], rw[:2]]) / shoulder_width

            motion = 0.0
            gesture_motion = 0.0
            if movement_vector is not None and previous_vector is not None:
                motion = finite(np.linalg.norm(movement_vector - previous_vector))
            if wrist_vector is not None and previous_wrists is not None:
                gesture_motion = finite(np.linalg.norm(wrist_vector - previous_wrists))
            if movement_vector is not None:
                previous_vector = movement_vector
            if wrist_vector is not None:
                previous_wrists = wrist_vector
            hand_count = float(bool(left_hand)) + float(bool(right_hand))

            rows.append(VisionRow(
                time=timestamp,
                face_detected=face_detected,
                pose_detected=pose_detected,
                hand_count=hand_count,
                head_yaw_proxy=head_yaw,
                head_pitch_proxy=head_pitch,
                camera_facing=camera_facing,
                shoulder_tilt=shoulder_tilt,
                torso_lean=torso_lean,
                motion=motion,
                gesture_motion=gesture_motion,
                mouth_open=mouth_open,
            ))
            frame_index += 1
    cap.release()

    data = [asdict(row) for row in rows]
    face_coverage = finite(np.mean([row.face_detected for row in rows]) if rows else 0.0)
    pose_coverage = finite(np.mean([row.pose_detected for row in rows]) if rows else 0.0)
    hand_coverage = finite(np.mean([row.hand_count > 0 for row in rows]) if rows else 0.0)
    return {
        "timeline": data,
        "summary": {
            "frames_analyzed": len(rows),
            "face_coverage": face_coverage,
            "pose_coverage": pose_coverage,
            "hand_coverage": hand_coverage,
        },
    }


def interpolate_vision(vision_rows: list[dict], times: np.ndarray, key: str) -> np.ndarray:
    if not vision_rows:
        return np.full(len(times), np.nan)
    source_times = np.array([row["time"] for row in vision_rows], dtype=float)
    values = np.array([row[key] for row in vision_rows], dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.full(len(times), np.nan)
    if valid.sum() == 1:
        return np.full(len(times), values[valid][0])
    return np.interp(times, source_times[valid], values[valid], left=np.nan, right=np.nan)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:05.2f}"


def peak_event_mask(values: np.ndarray, active: np.ndarray, step: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.where(active, values, -10.0)
    distance = max(1, int(round(0.5 / max(step, 1e-6))))
    indexes, _ = find_peaks(candidate, height=height, distance=distance, prominence=0.25)
    mask = np.zeros(len(values), dtype=bool)
    mask[indexes] = True
    return indexes, mask


def gesture_phases(times: np.ndarray, gesture_z: np.ndarray, valid: np.ndarray) -> list[dict]:
    if not len(times):
        return []
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.25
    indexes, _ = peak_event_mask(gesture_z, valid, step, height=0.75)
    events = []
    max_steps = max(1, int(round(1.0 / max(step, 1e-6))))
    for index in indexes:
        start = index
        while start > 0 and index - start < max_steps and gesture_z[start - 1] > 0:
            start -= 1
        end = index
        while end + 1 < len(times) and end - index < max_steps and gesture_z[end + 1] > 0:
            end += 1
        events.append({
            "prepare_start": finite(times[start]),
            "stroke_time": finite(times[index]),
            "recovery_end": finite(times[end] + step),
            "peak_strength_z": finite(gesture_z[index]),
        })
    return events


def evidence(times: np.ndarray, mask: np.ndarray, limit: int = 3) -> list[dict]:
    indexes = np.where(mask)[0][:limit]
    return [{"time": finite(times[index]), "label": format_time(finite(times[index]))} for index in indexes]


def fuse(audio: dict, vision: dict, duration: float, coaching_context: str = "camera") -> dict:
    timeline = audio["timeline"]
    times = np.array([row["time"] for row in timeline], dtype=float)
    energy = np.array([row["audio_energy"] for row in timeline], dtype=float)
    speech = np.array([row["speech_ratio"] for row in timeline], dtype=float)
    motion = interpolate_vision(vision["timeline"], times, "motion")
    gesture = interpolate_vision(vision["timeline"], times, "gesture_motion")
    facing = interpolate_vision(vision["timeline"], times, "camera_facing")
    lean = interpolate_vision(vision["timeline"], times, "torso_lean")
    tilt = interpolate_vision(vision["timeline"], times, "shoulder_tilt")
    hand_count = interpolate_vision(vision["timeline"], times, "hand_count")

    energy_z = zscore(energy)
    motion_z = zscore(np.nan_to_num(motion, nan=np.nanmedian(motion) if np.isfinite(motion).any() else 0.0))
    gesture_z = zscore(np.nan_to_num(gesture, nan=np.nanmedian(gesture) if np.isfinite(gesture).any() else 0.0))
    active = speech >= 0.45
    usable = active & np.isfinite(gesture)
    corr = float("nan")
    if usable.sum() >= 5 and np.std(energy_z[usable]) > 0 and np.std(gesture_z[usable]) > 0:
        corr = finite(np.corrcoef(energy_z[usable], gesture_z[usable])[0, 1], fallback=float("nan"))

    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.25
    emphasis_indexes, emphasis = peak_event_mask(energy_z, active, step, height=0.75)
    gesture_event_list = gesture_phases(times, gesture_z, np.isfinite(gesture) & (speech >= 0.1))
    gesture_times = np.array([item["stroke_time"] for item in gesture_event_list], dtype=float)
    aligned = np.zeros(len(times), dtype=bool)
    lags: list[float] = []
    for index in emphasis_indexes:
        if not len(gesture_times):
            continue
        nearest = int(np.argmin(np.abs(gesture_times - times[index])))
        lag = float(gesture_times[nearest] - times[index])
        if abs(lag) <= 0.35:
            aligned[index] = True
            lags.append(lag)
    alignment_ratio = finite(aligned.sum() / max(len(emphasis_indexes), 1))
    away_on_emphasis = emphasis & np.isfinite(facing) & (facing < 0.52)
    competing_motion = active & (energy_z < -0.25) & (motion_z > 1.0)
    interior = (times >= 5.0) & (times <= max(duration - 5.0, 5.0))
    still_pause = (~active) & (motion_z < -0.15) & interior

    rows = []
    for index, base in enumerate(timeline):
        rows.append({
            **base,
            "audio_energy_z": finite(energy_z[index]),
            "body_motion": finite(motion[index], fallback=float("nan")),
            "body_motion_z": finite(motion_z[index]),
            "gesture_motion": finite(gesture[index], fallback=float("nan")),
            "gesture_motion_z": finite(gesture_z[index]),
            "camera_facing": finite(facing[index], fallback=float("nan")),
            "torso_lean": finite(lean[index], fallback=float("nan")),
            "shoulder_tilt": finite(tilt[index], fallback=float("nan")),
            "hand_count": finite(hand_count[index], fallback=float("nan")),
        })

    findings: list[dict] = []
    face_coverage = vision["summary"]["face_coverage"]
    pose_coverage = vision["summary"]["pose_coverage"]
    coverage = min(face_coverage, pose_coverage)
    if coverage < 0.55:
        findings.append({
            "priority": 0,
            "kind": "quality",
            "title": "Reframe before interpreting body language",
            "observation": f"The face and body trackers had only {coverage:.0%} jointly usable coverage, so conclusions that combine head direction with movement are intentionally limited.",
            "experiment": "Use a stable camera, keep the face and shoulders visible, and avoid strong backlight before recording the next take.",
            "confidence": "high",
            "evidence": [],
        })

    if coaching_context == "camera" and face_coverage >= 0.55 and away_on_emphasis.sum() >= 2:
        findings.append({
            "priority": 1,
            "kind": "improve",
            "title": "Reconnect before the important beat",
            "observation": f"At {away_on_emphasis.sum()} high-energy moments, head direction moved away from the camera-facing zone.",
            "experiment": "Before the key sentence, return your face toward the lens, hold for one beat, then deliver the emphasis without adding another movement.",
            "confidence": "medium" if vision["summary"]["face_coverage"] < 0.8 else "high",
            "evidence": evidence(times, away_on_emphasis),
        })

    if emphasis.sum() >= 3 and alignment_ratio < 0.38 and vision["summary"]["pose_coverage"] >= 0.55:
        findings.append({
            "priority": 2,
            "kind": "improve",
            "title": "Give emphasis one visible landing point",
            "observation": f"Only {alignment_ratio:.0%} of strong vocal-energy beats coincided with a visible hand or wrist motion peak.",
            "experiment": "Choose one sentence to mark. Prepare the hand before it, place the gesture stroke on the emphasized phrase, then let the hand settle.",
            "confidence": "medium",
            "evidence": evidence(times, emphasis & ~aligned),
        })
    elif emphasis.sum() >= 3 and alignment_ratio >= 0.58 and pose_coverage >= 0.55:
        findings.append({
            "priority": 6,
            "kind": "strength",
            "title": "Your visible emphasis often lands with the voice",
            "observation": f"About {alignment_ratio:.0%} of high-energy vocal beats were accompanied by a movement peak on the same shared timeline.",
            "experiment": "Preserve that timing, but use it selectively so the strongest idea keeps the clearest gesture.",
            "confidence": "medium",
            "evidence": evidence(times, aligned),
        })

    if competing_motion.sum() >= 3 and vision["summary"]["pose_coverage"] >= 0.55:
        findings.append({
            "priority": 3,
            "kind": "improve",
            "title": "Quiet the body during lower-energy phrases",
            "observation": f"The tracker found {competing_motion.sum()} moments where body motion rose while vocal energy was comparatively low.",
            "experiment": "Record one version with deliberate stillness between emphasis points. Compare whether the important gestures become easier to read.",
            "confidence": "medium",
            "evidence": evidence(times, competing_motion),
        })

    pause_ratio = audio["summary"]["pause_ratio"]
    if pause_ratio < 0.055 and duration >= 30:
        findings.append({
            "priority": 2,
            "kind": "improve",
            "title": "Make more room around transitions",
            "observation": f"Detected pauses longer than 0.45 seconds occupied only {pause_ratio:.1%} of this sample.",
            "experiment": "Add one silent beat after the opening claim and one before the conclusion. Keep your body still through each pause.",
            "confidence": "medium",
            "evidence": [],
        })
    elif pause_ratio >= 0.08 and still_pause.sum() >= 2 and pose_coverage >= 0.55:
        findings.append({
            "priority": 7,
            "kind": "strength",
            "title": "Some pauses create a clean reset",
            "observation": "Several vocal pauses coincided with lower body motion, producing a readable boundary in the performance.",
            "experiment": "Keep the strongest reset and shorten any pause that does not separate two distinct ideas.",
            "confidence": "medium",
            "evidence": evidence(times, still_pause),
        })

    if audio["summary"]["pitch_variation_cents"] < 95 and duration >= 30:
        findings.append({
            "priority": 4,
            "kind": "improve",
            "title": "Test a wider vocal contrast",
            "observation": "Pitch movement stayed within a relatively narrow range for most voiced frames.",
            "experiment": "Pick one contrast in the content. Deliver the setup lower and calmer, then allow the key phrase a visibly different pitch and energy contour.",
            "confidence": "low",
            "evidence": [],
        })

    if not findings:
        findings.append({
            "priority": 5,
            "kind": "reflection",
            "title": "The basic signals are stable; test intentional contrast",
            "observation": "This sample did not trigger a strong, high-confidence timing conflict under the current rules.",
            "experiment": "Record two versions of the same 30 seconds—one deliberately restrained and one with explicit pause and gesture anchors—then compare them.",
            "confidence": "low",
            "evidence": [],
        })

    findings.sort(key=lambda item: item["priority"])
    facing_valid = facing[np.isfinite(facing)]
    lean_valid = lean[np.isfinite(lean)]
    return {
        "timeline": rows,
        "events": {
            "vocal_emphasis": [
                {"time": finite(times[index]), "energy_z": finite(energy_z[index])}
                for index in emphasis_indexes
            ],
            "gesture_phases": gesture_event_list,
            "aligned_voice_gesture": [
                {"time": finite(times[index]), "label": format_time(finite(times[index]))}
                for index in np.where(aligned)[0]
            ],
        },
        "speaker_baseline": {
            "audio_energy": robust_baseline(energy),
            "body_motion": robust_baseline(motion),
            "gesture_motion": robust_baseline(gesture),
            "camera_facing": robust_baseline(facing),
            "torso_lean": robust_baseline(lean),
        },
        "summary": {
            "duration_seconds": duration,
            "crossmodal_energy_motion_correlation": corr,
            "vocal_emphasis_events": int(len(emphasis_indexes)),
            "aligned_emphasis_events": int(aligned.sum()),
            "emphasis_motion_alignment_ratio": alignment_ratio,
            "median_gesture_lag_seconds": finite(np.median(lags) if lags else None, fallback=float("nan")),
            "median_absolute_gesture_lag_seconds": finite(np.median(np.abs(lags)) if lags else None, fallback=float("nan")),
            "gesture_events": len(gesture_event_list),
            "camera_facing_median": finite(np.median(facing_valid) if len(facing_valid) else None, fallback=float("nan")),
            "camera_facing_stability": finite(1.0 - np.std(facing_valid) if len(facing_valid) else None, fallback=float("nan")),
            "torso_lean_variation": finite(np.std(lean_valid) if len(lean_valid) else None, fallback=float("nan")),
            "away_during_emphasis_events": int(away_on_emphasis.sum()),
            "competing_motion_events": int(competing_motion.sum()),
        },
        "findings": findings[:6],
    }
