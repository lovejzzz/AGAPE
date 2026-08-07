from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .config import AuditConfig
from .storage import JobPaths, copy_with_limit


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def yt_dlp_runtime_args() -> list[str]:
    if shutil.which("deno"):
        return []
    node = shutil.which("node")
    return ["--js-runtimes", f"node:{node}"] if node else []


def fetch_source(
    source: str,
    paths: JobPaths,
    config: AuditConfig,
    segment_seconds: int | None,
    segment_start_seconds: int = 0,
) -> dict:
    if not is_url(source):
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        copy_with_limit(source_path, paths.raw, config.max_input_bytes)
        return {"kind": "local", "source": str(source_path), "title": source_path.stem}

    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        candidate = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "yt-dlp"
        yt_dlp = str(candidate) if candidate.exists() else None
    if not yt_dlp:
        raise RuntimeError("yt-dlp is not installed in the project environment")

    output_template = str(paths.root / "download.%(ext)s")
    command = [
        yt_dlp,
        *yt_dlp_runtime_args(),
        "--no-playlist",
        "--max-filesize",
        str(config.max_input_bytes),
        "--format",
        "bv*[height<=720]+ba/b[height<=720]",
        "--merge-output-format",
        "mp4",
        "--output",
        output_template,
        "--print",
        "after_move:filepath",
    ]
    if segment_seconds:
        segment_end = segment_start_seconds + segment_seconds
        command.extend([
            "--download-sections", f"*{segment_start_seconds}-{segment_end}",
            "--force-keyframes-at-cuts",
        ])
    command.append(source)
    result = run(command)
    candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    downloaded = next((path for path in reversed(candidates) if path.exists()), None)
    if downloaded is None:
        downloaded = next(paths.root.glob("download.*"), None)
    if downloaded is None:
        raise RuntimeError("Downloader completed without producing a media file")
    if downloaded.stat().st_size > config.max_input_bytes:
        downloaded.unlink(missing_ok=True)
        raise ValueError("Downloaded media exceeded the configured storage limit")
    downloaded.replace(paths.raw)
    return {"kind": "url", "source": source, "title": urlparse(source).path.rsplit("/", 1)[-1] or "YouTube sample"}


def probe(path: Path) -> dict:
    result = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ])
    return json.loads(result.stdout)


def prepare_derivatives(paths: JobPaths, config: AuditConfig, segment_seconds: int | None) -> dict:
    metadata = probe(paths.raw)
    duration = float(metadata.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise ValueError("Could not determine video duration")
    effective_duration = min(duration, float(config.max_duration_seconds))
    if segment_seconds:
        effective_duration = min(effective_duration, float(segment_seconds))

    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(paths.raw),
        "-t", f"{effective_duration:.3f}",
        "-vf", f"scale='min({config.proxy_width},iw)':-2,fps={config.proxy_fps}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-movflags", "+faststart", str(paths.proxy),
    ])
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(paths.raw),
        "-t", f"{effective_duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(config.audio_sample_rate),
        "-c:a", "pcm_s16le", str(paths.audio),
    ])
    return {"probe": metadata, "source_duration_seconds": duration, "analyzed_duration_seconds": effective_duration}
