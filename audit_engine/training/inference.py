from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import FEATURE_NAMES, intact_windows_for_clip, load_clip, sha256_file
from .model import load_checkpoint, select_device
from .runner import _json_safe, normalized, predict_probabilities


def score_features(
    checkpoint_path: Path,
    features_path: Path,
    *,
    output_path: Path | None = None,
    requested_device: str = "auto",
    allow_experimental: bool = False,
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    features_path = features_path.expanduser().resolve()
    judge_path = checkpoint_path.parent / "judge.json"
    verdict = None
    if judge_path.exists():
        judgment = json.loads(judge_path.read_text(encoding="utf-8"))
        judged_checkpoint = Path(str(judgment.get("checkpoint", ""))).expanduser().resolve()
        digest_matches = judgment.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        if judged_checkpoint == checkpoint_path and digest_matches:
            verdict = judgment.get("verdict")
    if verdict != "PROMOTE" and not allow_experimental:
        raise ValueError(
            "checkpoint has not received a PROMOTE verdict; run `agape judge` or pass --allow-experimental"
        )

    device = select_device(requested_device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("checkpoint feature contract does not match this AGAPE version")
    clip = load_clip(features_path)
    expected_step = float(checkpoint["timeline_step_seconds"])
    if abs(clip.step_seconds - expected_step) > max(0.02, expected_step * 0.1):
        raise ValueError("feature timeline resolution does not match the checkpoint")
    window_steps = int(checkpoint["window_steps"])
    windows, starts = intact_windows_for_clip(
        clip,
        window_steps=window_steps,
        stride_steps=max(1, window_steps // 4),
    )
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    probabilities = predict_probabilities(
        model,
        normalized(windows, mean, std),
        device=device,
        batch_size=128,
    )
    window_seconds = window_steps * clip.step_seconds
    result = {
        "kind": "agape_learned_synchrony_score",
        "checkpoint": str(checkpoint_path),
        "checkpoint_judgment": verdict or "UNJUDGED",
        "features": str(features_path),
        "target_boundary": checkpoint["target_boundary"],
        "summary": {
            "windows": len(probabilities),
            "mean_intact_probability": float(np.mean(probabilities)),
            "median_intact_probability": float(np.median(probabilities)),
            "minimum_intact_probability": float(np.min(probabilities)),
        },
        "windows": [
            {
                "start_seconds": float(start),
                "end_seconds": float(start + window_seconds),
                "intact_probability": float(probability),
            }
            for start, probability in zip(starts, probabilities, strict=False)
        ],
        "interpretation_boundary": "A low score means the learned model found patterns resembling its synthetic time shifts. It is not proof of poor delivery or a causal diagnosis.",
    }
    output_path = (output_path or features_path.with_name("learned-synchrony.json")).expanduser().resolve()
    output_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    result["output"] = str(output_path)
    return result
