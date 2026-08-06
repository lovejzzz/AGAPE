from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditConfig:
    runs_root: Path
    model_path: Path
    max_input_bytes: int = 750 * 1024 * 1024
    max_duration_seconds: int = 20 * 60
    proxy_width: int = 960
    proxy_fps: int = 8
    analysis_fps: float = 4.0
    audio_sample_rate: int = 16_000
    timeline_window_seconds: float = 0.25
    retain_media: bool = False
