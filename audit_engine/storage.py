from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class JobPaths:
    root: Path
    raw: Path
    proxy: Path
    audio: Path
    features: Path
    chart: Path
    report_html: Path
    report_md: Path
    manifest: Path


def create_job(runs_root: Path, source_label: str) -> JobPaths:
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(source_label.encode("utf-8")).hexdigest()[:8]
    root = runs_root / f"{stamp}-{digest}"
    suffix = 1
    while root.exists():
        root = runs_root / f"{stamp}-{digest}-{suffix}"
        suffix += 1
    root.mkdir(mode=0o700)
    return JobPaths(
        root=root,
        raw=root / "source.mp4",
        proxy=root / "analysis-proxy.mp4",
        audio=root / "analysis-audio.wav",
        features=root / "features.json",
        chart=root / "timeline.png",
        report_html=root / "report.html",
        report_md=root / "summary.md",
        manifest=root / "manifest.json",
    )


def copy_with_limit(source: Path, destination: Path, max_bytes: int) -> None:
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Input is {size / 1024 / 1024:.1f} MB; limit is {max_bytes / 1024 / 1024:.0f} MB")
    copied = 0
    with source.open("rb") as src, destination.open("wb") as dst:
        while chunk := src.read(1024 * 1024):
            copied += len(chunk)
            if copied > max_bytes:
                destination.unlink(missing_ok=True)
                raise ValueError("Input exceeded storage limit while copying")
            dst.write(chunk)


def file_record(path: Path) -> dict:
    if not path.exists():
        return {"path": path.name, "present": False, "bytes": 0}
    return {"path": path.name, "present": True, "bytes": path.stat().st_size}


def write_manifest(paths: JobPaths, payload: dict) -> None:
    files = [
        paths.raw,
        paths.proxy,
        paths.audio,
        paths.features,
        paths.chart,
        paths.report_html,
        paths.report_md,
    ]
    payload = {**payload, "files": [file_record(path) for path in files]}
    paths.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cleanup_media(paths: JobPaths) -> list[str]:
    deleted: list[str] = []
    for path in (paths.raw, paths.proxy, paths.audio):
        if path.exists():
            path.unlink()
            deleted.append(path.name)
    return deleted


def remove_failed_job(paths: JobPaths) -> None:
    shutil.rmtree(paths.root, ignore_errors=True)
