from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import DATASET_SCHEMA_VERSION


BASE_FEATURE_NAMES = (
    "audio_energy_z",
    "speech_ratio",
    "pitch_relative",
    "vocal_pulses_log1p",
    "body_motion_z",
    "gesture_motion_z",
    "camera_facing",
    "torso_lean",
    "shoulder_tilt_scaled",
    "hand_count_scaled",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(f"{name}_valid" for name in BASE_FEATURE_NAMES)
VISUAL_BASE_INDEXES = tuple(range(4, len(BASE_FEATURE_NAMES)))
VISUAL_INPUT_INDEXES = VISUAL_BASE_INDEXES + tuple(
    len(BASE_FEATURE_NAMES) + index for index in VISUAL_BASE_INDEXES
)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Clip:
    path: Path
    group_id: str
    clip_id: str
    step_seconds: float
    matrix: np.ndarray
    is_demo: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def resolve_feature_files(inputs: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("features.json"))
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no features.json files were found")
    return sorted(files)


def _raw_feature_matrix(rows: list[dict]) -> np.ndarray:
    pitch = np.array([finite_number(row.get("pitch_hz")) for row in rows], dtype=np.float32)
    valid_pitch = np.isfinite(pitch) & (pitch > 0)
    pitch_relative = np.full(len(rows), np.nan, dtype=np.float32)
    if valid_pitch.any():
        median = float(np.median(pitch[valid_pitch]))
        pitch_relative[valid_pitch] = np.log2(pitch[valid_pitch] / max(median, 1e-6))

    columns = [
        np.array([finite_number(row.get("audio_energy_z")) for row in rows]),
        np.array([finite_number(row.get("speech_ratio")) for row in rows]),
        pitch_relative,
        np.log1p(np.maximum(0.0, np.array([
            finite_number(row.get("vocal_pulses")) for row in rows
        ]))),
        np.array([finite_number(row.get("body_motion_z")) for row in rows]),
        np.array([finite_number(row.get("gesture_motion_z")) for row in rows]),
        np.array([finite_number(row.get("camera_facing")) for row in rows]),
        np.array([finite_number(row.get("torso_lean")) for row in rows]),
        np.array([finite_number(row.get("shoulder_tilt")) / 45.0 for row in rows]),
        np.array([finite_number(row.get("hand_count")) / 2.0 for row in rows]),
    ]
    base = np.column_stack(columns).astype(np.float32)
    valid = np.isfinite(base).astype(np.float32)
    base = np.nan_to_num(base, nan=0.0, posinf=8.0, neginf=-8.0)
    base = np.clip(base, -8.0, 8.0)
    return np.concatenate([base, valid], axis=1).astype(np.float32)


def load_clip(path: Path, group_override: str | None = None) -> Clip:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("fusion", {}).get("timeline")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{path} is not a single-take AGAPE features file with a fusion timeline")
    times = np.array([finite_number(row.get("time")) for row in rows], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path} has an invalid or non-monotonic timeline")
    step = float(np.median(np.diff(times)))
    declared_step = finite_number(payload.get("timeline_resolution_seconds"))
    if math.isfinite(declared_step) and abs(step - declared_step) > max(0.02, declared_step * 0.1):
        raise ValueError(f"{path} timeline resolution does not match its declared value")

    source = payload.get("source", {})
    source_label = str(source.get("source") or payload.get("job") or path.parent)
    group_label = str(group_override or payload.get("training_group") or source_label)
    clip_label = f"{group_label}:{path}:{sha256_file(path)}"
    return Clip(
        path=path,
        group_id=stable_id(group_label),
        clip_id=stable_id(clip_label),
        step_seconds=step,
        matrix=_raw_feature_matrix(rows),
        is_demo=bool(payload.get("training_demo", False)),
    )


def assign_group_splits(group_ids: Iterable[str], seed: int) -> dict[str, str]:
    groups = sorted(set(group_ids))
    if len(groups) < 3:
        raise ValueError(
            "at least three independent sources are required so train, validation, and test data do not overlap"
        )
    random.Random(seed).shuffle(groups)
    test_count = max(4 if len(groups) >= 8 else 1, int(round(len(groups) * 0.2)))
    val_count = max(1, int(round(len(groups) * 0.15)))
    if test_count + val_count >= len(groups):
        test_count = 1
        val_count = 1
    train_end = len(groups) - val_count - test_count
    mapping = {group: "train" for group in groups[:train_end]}
    mapping.update({group: "val" for group in groups[train_end:train_end + val_count]})
    mapping.update({group: "test" for group in groups[train_end + val_count:]})
    return mapping


def _starts(length: int, window: int, stride: int, offset: int = 0) -> list[int]:
    minimum = max(0, offset)
    maximum = min(length - window, length - window + offset)
    if maximum < minimum:
        return []
    first = minimum + ((stride - minimum % stride) % stride)
    return list(range(first, maximum + 1, stride))


def _shift_visual(matrix: np.ndarray, offset_steps: int) -> np.ndarray:
    result = matrix.copy()
    target_indexes = np.arange(len(matrix))
    source_indexes = target_indexes - offset_steps
    valid = (source_indexes >= 0) & (source_indexes < len(matrix))
    valid_targets = target_indexes[valid]
    result[np.ix_(valid_targets, VISUAL_INPUT_INDEXES)] = matrix[
        np.ix_(source_indexes[valid], VISUAL_INPUT_INDEXES)
    ]
    return result


def _masked_correlation(window: np.ndarray, first: int, second: int) -> float:
    mask_offset = len(BASE_FEATURE_NAMES)
    valid = (window[:, mask_offset + first] > 0.5) & (window[:, mask_offset + second] > 0.5)
    if int(valid.sum()) < 8:
        return 0.0
    left = window[valid, first].astype(float)
    right = window[valid, second].astype(float)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator > 1e-8 else 0.0


def window_alignment_evidence(window: np.ndarray) -> float:
    return max(
        _masked_correlation(window, 0, 4),
        _masked_correlation(window, 0, 5),
        _masked_correlation(window, 1, 4),
        _masked_correlation(window, 1, 5),
    )


def windows_for_clip(
    clip: Clip,
    *,
    window_seconds: float,
    stride_seconds: float,
    offsets_seconds: tuple[float, ...],
    seed: int,
    minimum_alignment_gain: float = 0.10,
) -> list[tuple[np.ndarray, int, float, float, float]]:
    window = max(4, int(round(window_seconds / clip.step_seconds)))
    stride = max(1, int(round(stride_seconds / clip.step_seconds)))
    if len(clip.matrix) < window:
        return []

    shifted_by_offset: dict[float, tuple[int, np.ndarray]] = {}
    for requested_offset in offsets_seconds:
        offset_steps = int(round(requested_offset / clip.step_seconds))
        if offset_steps == 0:
            continue
        actual_offset = offset_steps * clip.step_seconds
        shifted_by_offset[actual_offset] = (offset_steps, _shift_visual(clip.matrix, offset_steps))
    if not shifted_by_offset:
        raise ValueError(f"no shifted negative windows could be made from {clip.path}")
    results: list[tuple[np.ndarray, int, float, float, float]] = []
    offsets = sorted(shifted_by_offset)
    starts = _starts(len(clip.matrix), window, stride)
    rotation = (seed ^ int(clip.clip_id[:8], 16)) % len(offsets)
    for window_index, start in enumerate(starts):
        positive = clip.matrix[start:start + window]
        positive_score = window_alignment_evidence(positive)
        eligible: list[tuple[float, float, np.ndarray]] = []
        for actual_offset, (offset_steps, shifted) in shifted_by_offset.items():
            if start < max(0, offset_steps) or start > min(
                len(clip.matrix) - window, len(clip.matrix) - window + offset_steps
            ):
                continue
            negative = shifted[start:start + window]
            gain = positive_score - window_alignment_evidence(negative)
            if gain + 1e-9 >= minimum_alignment_gain:
                eligible.append((gain, actual_offset, negative))
        if not eligible:
            continue
        desired_offset = offsets[(window_index + rotation) % len(offsets)]
        desired = [candidate for candidate in eligible if candidate[1] == desired_offset]
        gain, actual_offset, negative = max(desired or eligible, key=lambda item: item[0])
        start_seconds = start * clip.step_seconds
        results.extend([
            (positive, 1, 0.0, start_seconds, gain),
            (negative, 0, actual_offset, start_seconds, gain),
        ])
    return results


def intact_windows_for_clip(
    clip: Clip, *, window_steps: int, stride_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(clip.matrix) < window_steps:
        raise ValueError(
            f"{clip.path} is shorter than the checkpoint's {window_steps * clip.step_seconds:.2f}s window"
        )
    starts = _starts(len(clip.matrix), window_steps, stride_steps)
    windows = np.stack([clip.matrix[start:start + window_steps] for start in starts]).astype(np.float32)
    start_seconds = np.asarray([start * clip.step_seconds for start in starts], dtype=np.float32)
    return windows, start_seconds


def parse_offsets(raw: str) -> tuple[float, ...]:
    try:
        values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("offsets must be comma-separated seconds") from exc
    if not values or any(abs(value) < 0.45 or abs(value) > 3.0 for value in values):
        raise ValueError("provide offsets from -3.0 to 3.0 seconds, excluding -0.45 to 0.45")
    if not any(value < 0 for value in values) or not any(value > 0 for value in values):
        raise ValueError("offsets must include both negative and positive time shifts")
    return tuple(values)


def build_dataset(
    inputs: Iterable[str | Path],
    output: Path,
    *,
    window_seconds: float = 8.0,
    stride_seconds: float = 2.0,
    offsets_seconds: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
    seed: int = 17,
    group_map: dict[str, str] | None = None,
    group_map_only: bool = False,
    split_plan: dict[str, str] | None = None,
    minimum_alignment_gain: float = 0.10,
) -> dict:
    if window_seconds < 2.0 or stride_seconds <= 0 or stride_seconds > window_seconds:
        raise ValueError("window must be at least 2 seconds and stride must be within the window")
    if minimum_alignment_gain < 0.0 or minimum_alignment_gain > 1.5:
        raise ValueError("minimum alignment gain must be between 0 and 1.5")
    files = resolve_feature_files(inputs)
    normalized_group_map = {
        str(Path(key).expanduser().resolve()): value for key, value in (group_map or {}).items()
    }
    unknown_group_paths = sorted(set(normalized_group_map) - {str(path) for path in files})
    if unknown_group_paths:
        raise ValueError(f"group map contains paths that are not dataset inputs: {unknown_group_paths}")
    if group_map_only:
        if not normalized_group_map:
            raise ValueError("group-map-only requires at least one group map")
        files = [path for path in files if str(path) in normalized_group_map]
    clips = [load_clip(path, normalized_group_map.get(str(path))) for path in files]
    resolutions = {round(clip.step_seconds, 6) for clip in clips}
    if len(resolutions) != 1:
        raise ValueError("all input timelines must use the same resolution")
    clip_groups = {clip.group_id for clip in clips}
    if split_plan is None:
        split_map = assign_group_splits(clip_groups, seed)
    else:
        invalid_splits = sorted(set(split_plan.values()) - set(SPLIT_NAMES))
        if invalid_splits:
            raise ValueError(f"split plan contains invalid splits: {invalid_splits}")
        default_split = split_plan.get("*")
        split_map = {
            stable_id(group): split for group, split in split_plan.items() if group != "*"
        }
        missing_groups = sorted(clip_groups - set(split_map))
        unknown_groups = sorted(set(split_map) - clip_groups)
        if default_split:
            split_map.update({group: default_split for group in missing_groups})
            missing_groups = []
        if missing_groups or unknown_groups:
            raise ValueError(
                f"split plan must assign every input group exactly once; missing={missing_groups}, unknown={unknown_groups}"
            )
        if set(split_map.values()) != set(SPLIT_NAMES):
            raise ValueError("split plan must contain train, val, and test groups")

    samples: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    groups: list[str] = []
    clip_ids: list[str] = []
    offsets: list[float] = []
    starts: list[float] = []
    demo_flags: list[bool] = []
    alignment_gains: list[float] = []
    skipped: list[str] = []
    for clip in clips:
        clip_windows = windows_for_clip(
            clip,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            offsets_seconds=offsets_seconds,
            seed=seed,
            minimum_alignment_gain=minimum_alignment_gain,
        )
        if not clip_windows:
            skipped.append(str(clip.path))
            continue
        for matrix, label, offset, start, alignment_gain in clip_windows:
            samples.append(matrix)
            labels.append(label)
            splits.append(split_map[clip.group_id])
            groups.append(clip.group_id)
            clip_ids.append(clip.clip_id)
            offsets.append(offset)
            starts.append(start)
            demo_flags.append(clip.is_demo)
            alignment_gains.append(alignment_gain)
    if not samples:
        raise ValueError("no training windows were created")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=np.stack(samples).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        split=np.asarray(splits, dtype="U5"),
        group_id=np.asarray(groups, dtype="U32"),
        clip_id=np.asarray(clip_ids, dtype="U32"),
        offset_seconds=np.asarray(offsets, dtype=np.float32),
        start_seconds=np.asarray(starts, dtype=np.float32),
        is_demo=np.asarray(demo_flags, dtype=np.bool_),
        alignment_gain=np.asarray(alignment_gains, dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U40"),
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
        timeline_step_seconds=np.asarray(next(iter(resolutions)), dtype=np.float32),
    )
    split_counts = {
        name: int(sum(item == name for item in splits)) for name in SPLIT_NAMES
    }
    split_groups = {
        name: sorted(group for group, split in split_map.items() if split == name)
        for name in SPLIT_NAMES
    }
    split_offset_counts = {
        name: {
            f"{requested:+.2f}": int(sum(
                split == name and abs(offset - requested) < 1e-6
                for split, offset in zip(splits, offsets)
            ))
            for requested in offsets_seconds
        }
        for name in SPLIT_NAMES
    }
    test_delay_shortfalls = {
        delay: count for delay, count in split_offset_counts["test"].items() if count < 5
    }
    manifest = {
        "kind": "agape_synchrony_dataset",
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset": str(output),
        "dataset_sha256": sha256_file(output),
        "feature_names": list(FEATURE_NAMES),
        "sample_shape": list(samples[0].shape),
        "samples": len(samples),
        "positive_samples": int(sum(labels)),
        "negative_samples": int(len(labels) - sum(labels)),
        "source_groups": len(split_map),
        "contains_demo_data": bool(any(demo_flags)),
        "split_counts": split_counts,
        "split_groups": split_groups,
        "split_offset_counts": split_offset_counts,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "offsets_seconds": list(offsets_seconds),
        "minimum_alignment_gain": minimum_alignment_gain,
        "seed": seed,
        "split_strategy": "explicit_plan" if split_plan is not None else "seeded_group_shuffle",
        "explicit_group_map_entries": len(normalized_group_map),
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in files
        ],
        "skipped_short_inputs": skipped,
        "promotion_preflight_warnings": ([
            "Untouched test delay coverage is below five examples for: "
            + ", ".join(f"{delay} ({count})" for delay, count in test_delay_shortfalls.items())
        ] if test_delay_shortfalls else []),
        "label_contract": {
            "1": "intact shared-timeline window with measurable cross-modal evidence",
            "0": "the same source and time window with visual features shifted enough to weaken that evidence",
            "boundary": "This paired target measures detectable audiovisual synchrony, not speaking quality or human worth.",
        },
        "leakage_control": "Group split is assigned before window extraction; one source group appears in exactly one split.",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key] for key in payload.files}
    version = int(np.asarray(result.get("schema_version", -1)).item())
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema {version}")
    if tuple(result["feature_names"].tolist()) != FEATURE_NAMES:
        raise ValueError("dataset feature contract does not match this AGAPE version")
    if result["x"].ndim != 3 or result["x"].shape[-1] != len(FEATURE_NAMES):
        raise ValueError("dataset tensor has the wrong shape")
    return result


def validate_split_integrity(dataset: dict[str, np.ndarray]) -> dict:
    group_sets = {
        name: set(dataset["group_id"][dataset["split"] == name].tolist())
        for name in SPLIT_NAMES
    }
    overlaps = {
        "train_val": sorted(group_sets["train"] & group_sets["val"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "val_test": sorted(group_sets["val"] & group_sets["test"]),
    }
    return {
        "passed": not any(overlaps.values()) and all(group_sets.values()),
        "groups": {name: len(groups) for name, groups in group_sets.items()},
        "overlaps": overlaps,
    }
