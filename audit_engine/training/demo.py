from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .data import build_dataset
from .judge import judge_checkpoint
from .runner import train_model


def _smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def write_demo_sources(directory: Path, *, sources: int = 10, seed: int = 17) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    step = 0.25
    count = int(72 / step)
    for source_index in range(sources):
        rng = np.random.default_rng(seed + source_index * 101)
        raw = _smooth(rng.normal(size=count), width=9)
        pulse_indexes = rng.choice(np.arange(12, count - 12), size=24, replace=False)
        for index in pulse_indexes:
            raw[index - 2:index + 3] += np.array([0.25, 0.9, 2.0, 0.9, 0.25])
        energy = (raw - raw.mean()) / max(raw.std(), 1e-6)
        gesture = energy + rng.normal(scale=0.08, size=count)
        body = 0.65 * energy + rng.normal(scale=0.14, size=count)
        speech = np.clip(0.72 + 0.18 * _smooth(rng.normal(size=count), 11), 0.0, 1.0)
        pitch = 125 + source_index * 4 + 10 * energy + rng.normal(scale=1.5, size=count)
        facing = np.clip(0.82 + 0.04 * _smooth(rng.normal(size=count), 17), 0.0, 1.0)
        rows = []
        for index in range(count):
            rows.append({
                "time": index * step,
                "audio_energy": float(max(energy[index] + 3.0, 0.0)),
                "speech_ratio": float(speech[index]),
                "pitch_hz": float(pitch[index]),
                "vocal_pulses": int(energy[index] > 1.0),
                "audio_energy_z": float(energy[index]),
                "body_motion": float(max(body[index] + 3.0, 0.0)),
                "body_motion_z": float(body[index]),
                "gesture_motion": float(max(gesture[index] + 3.0, 0.0)),
                "gesture_motion_z": float(gesture[index]),
                "camera_facing": float(facing[index]),
                "torso_lean": float(0.05 * np.sin(index / 31 + source_index)),
                "shoulder_tilt": float(2.0 * np.sin(index / 47 + source_index)),
                "hand_count": float(1 + (index // 29) % 2),
            })
        payload = {
            "engine": "AGAPE",
            "engine_version": "0.2.0",
            "training_demo": True,
            "job": f"synthetic-demo-{source_index}",
            "source": {"kind": "synthetic_demo", "source": f"demo-source-{source_index}"},
            "timeline_resolution_seconds": step,
            "fusion": {"timeline": rows},
        }
        source_directory = directory / f"source-{source_index:02d}"
        source_directory.mkdir()
        path = source_directory / "features.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    return paths


def run_demo(
    project_root: Path,
    *,
    epochs: int = 18,
    requested_device: str = "auto",
    seed: int = 17,
) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    demo_root = project_root / "training_data" / f"demo-{stamp}"
    suffix = 1
    while demo_root.exists():
        demo_root = project_root / "training_data" / f"demo-{stamp}-{suffix}"
        suffix += 1
    source_paths = write_demo_sources(demo_root / "sources", seed=seed)
    dataset_path = demo_root / "demo-synchrony.npz"
    dataset_manifest = build_dataset(source_paths, dataset_path, seed=seed)
    training = train_model(
        dataset_path,
        project_root / "training_runs",
        epochs=epochs,
        batch_size=64,
        hidden_size=32,
        blocks=2,
        patience=6,
        seed=seed,
        requested_device=requested_device,
    )
    judgment = judge_checkpoint(
        Path(training["best_checkpoint"]),
        dataset_path,
        requested_device=requested_device,
    )
    return {
        "kind": "agape_local_training_smoke_test",
        "dataset": dataset_manifest,
        "training": training,
        "judgment": judgment,
        "expected_verdict": "DO_NOT_PROMOTE because generated data only validates the training machinery",
    }
