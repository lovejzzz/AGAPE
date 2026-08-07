from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .analysis import analyze_audio, analyze_vision, fuse
from .comparison import compare_takes
from .comparison_report import write_comparison_reports
from .config import AuditConfig
from .lab import score_control, score_delay, summarize_calibration
from .lab_report import write_lab_reports
from .media import fetch_source, prepare_derivatives, probe, run
from .report import write_reports
from .storage import cleanup_media, create_job, remove_failed_job, write_manifest


def media_snapshot(paths) -> list[dict]:
    return [
        {"path": path.name, "bytes": path.stat().st_size if path.exists() else 0}
        for path in (paths.raw, paths.proxy, paths.audio)
    ]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_config(args: argparse.Namespace) -> AuditConfig:
    root = project_root()
    config = AuditConfig(
        runs_root=Path(args.runs_root).resolve() if args.runs_root else root / "runs",
        model_path=Path(args.model).resolve() if args.model else root / "models" / "holistic_landmarker.task",
        retain_media=args.retain_media,
    )
    if not config.model_path.exists():
        raise FileNotFoundError(f"MediaPipe model is missing: {config.model_path}")
    if args.segment_seconds and args.segment_seconds > config.max_duration_seconds:
        raise ValueError(f"segment-seconds cannot exceed {config.max_duration_seconds}")
    if args.segment_start_seconds < 0:
        raise ValueError("segment-start-seconds cannot be negative")
    return config


def analyze_source(
    source_label: str,
    requested_title: str | None,
    args: argparse.Namespace,
    config: AuditConfig,
) -> dict:
    paths = create_job(config.runs_root, source_label)
    started = datetime.now(UTC)
    try:
        source = fetch_source(
            source_label, paths, config, args.segment_seconds, args.segment_start_seconds
        )
        media = prepare_derivatives(paths, config, args.segment_seconds)
        duration = float(media["analyzed_duration_seconds"])
        audio = analyze_audio(paths.audio, config, duration)
        vision = analyze_vision(paths.proxy, config, duration)
        fusion = fuse(audio, vision, duration, coaching_context=args.context)
        title = requested_title or source.get("title") or "Communication audit"
        data = {
            "engine": "AGAPE",
            "engine_full_name": "Audiovisual Gesture And Prosody Engine",
            "engine_version": "0.2.0",
            "job": str(paths.root),
            "title": title,
            "created_at": datetime.now(UTC).isoformat(),
            "source": source,
            "media": media,
            "audio": audio,
            "vision": vision,
            "fusion": fusion,
            "transcript_used": False,
            "coaching_context": args.context,
            "timeline_resolution_seconds": config.timeline_window_seconds,
        }
        write_reports(data, paths)
        temporary_media = media_snapshot(paths)
        deleted = [] if config.retain_media else cleanup_media(paths)
        finished = datetime.now(UTC)
        processing_seconds = (finished - started).total_seconds()
        write_manifest(paths, {
            "engine": "AGAPE",
            "engine_version": "0.2.0",
            "status": "complete",
            "created_at": started.isoformat(),
            "completed_at": finished.isoformat(),
            "processing_seconds": processing_seconds,
            "retention": {
                "raw_media_retained": config.retain_media,
                "deleted_after_success": deleted,
                "policy": "derived reports retained; large media deleted by default",
                "temporary_media_before_cleanup": temporary_media,
                "temporary_media_bytes": sum(item["bytes"] for item in temporary_media),
            },
            "limits": {
                "max_input_bytes": config.max_input_bytes,
                "max_duration_seconds": config.max_duration_seconds,
            },
        })
        return {
            "data": data,
            "paths": paths,
            "processing_seconds": processing_seconds,
            "media_deleted": deleted,
        }
    except Exception:
        if args.keep_failed_job:
            write_manifest(paths, {
                "engine": "AGAPE",
                "engine_version": "0.2.0",
                "status": "failed",
                "created_at": started.isoformat(),
                "retention": {"policy": "failed job retained for debugging by explicit flag"},
            })
        else:
            remove_failed_job(paths)
        raise


def analyze_command(args: argparse.Namespace) -> int:
    config = make_config(args)
    result = analyze_source(args.source, args.title, args, config)
    paths = result["paths"]
    print(json.dumps({
        "engine": "AGAPE",
        "job": str(paths.root),
        "report": str(paths.report_html),
        "summary": str(paths.report_md),
        "processing_seconds": result["processing_seconds"],
        "media_deleted": result["media_deleted"],
    }, indent=2))
    return 0


def compare_command(args: argparse.Namespace) -> int:
    config = make_config(args)
    comparison_paths = None
    first = analyze_source(args.take_1, f"{args.title} — Take 1" if args.title else "Take 1", args, config)
    second = analyze_source(args.take_2, f"{args.title} — Take 2" if args.title else "Take 2", args, config)
    try:
        comparison_paths = create_job(config.runs_root, f"compare:{args.take_1}:{args.take_2}")
        title = args.title or "Take 1 versus Take 2"
        comparison = compare_takes(first["data"], second["data"], title)
        comparison.update({
            "job": str(comparison_paths.root),
            "created_at": datetime.now(UTC).isoformat(),
            "analysis_jobs": [str(first["paths"].root), str(second["paths"].root)],
        })
        write_comparison_reports(comparison, comparison_paths)
        write_manifest(comparison_paths, {
            "engine": "AGAPE",
            "engine_version": "0.2.0",
            "status": "complete",
            "kind": "paired_take_comparison",
            "created_at": comparison["created_at"],
            "analysis_jobs": comparison["analysis_jobs"],
            "retention": {
                "raw_media_retained": config.retain_media,
                "policy": "comparison stores derived features and reports only; source-job policy applies to media",
            },
        })
        print(json.dumps({
            "engine": "AGAPE",
            "comparison_job": str(comparison_paths.root),
            "report": str(comparison_paths.report_html),
            "summary": str(comparison_paths.report_md),
            "take_jobs": comparison["analysis_jobs"],
            "processing_seconds": first["processing_seconds"] + second["processing_seconds"],
            "media_deleted": first["media_deleted"] + second["media_deleted"],
        }, indent=2))
        return 0
    except Exception:
        if comparison_paths is not None and not args.keep_failed_job:
            remove_failed_job(comparison_paths)
        raise


def feedback_command(args: argparse.Namespace) -> int:
    job = Path(args.job).expanduser().resolve()
    features = job / "features.json"
    if not job.is_dir() or not features.is_file():
        raise ValueError("job must be a completed AGAPE report directory")
    feedback_path = job / "feedback.json"
    if feedback_path.exists():
        payload = json.loads(feedback_path.read_text(encoding="utf-8"))
    else:
        payload = {"engine": "AGAPE", "engine_version": "0.2.0", "ratings": []}
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        "finding": args.finding,
        "judgment": args.judgment,
        "notes": args.notes or "",
    }
    payload["ratings"].append(record)
    feedback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(feedback_path), "rating": record}, indent=2))
    return 0


def parse_delays(raw: str) -> list[float]:
    try:
        delays = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("delays must be comma-separated seconds, such as 0.25,0.50,0.75") from exc
    if not delays or len(delays) > 4 or any(item < 0.20 or item > 2.0 for item in delays):
        raise ValueError("provide one to four delays between 0.20 and 2.00 seconds")
    return delays


def lab_command(args: argparse.Namespace) -> int:
    config = make_config(args)
    delays = parse_delays(args.delays)
    lab_paths = create_job(config.runs_root, f"lab:{args.source}:{args.delays}")
    started = datetime.now(UTC)
    variants: list[Path] = []
    try:
        source = fetch_source(
            args.source, lab_paths, config, args.segment_seconds, args.segment_start_seconds
        )
        metadata = probe(lab_paths.raw)
        source_duration = float(metadata.get("format", {}).get("duration") or 0.0)
        duration = min(source_duration, float(args.segment_seconds or config.max_duration_seconds))
        if duration < 10.0:
            raise ValueError("AGAPE Lab requires at least 10 seconds of audiovisual material")

        base_title = args.title or source.get("title") or "AGAPE Lab sample"
        baseline = analyze_source(str(lab_paths.raw), f"{base_title} — baseline", args, config)
        repeat = analyze_source(str(lab_paths.raw), f"{base_title} — repeatability control", args, config)
        control = score_control(baseline["data"], repeat["data"])

        delay_tests = []
        variant_jobs = []
        for index, delay in enumerate(delays, start=1):
            variant = lab_paths.root / f"visual-delay-{index}.mp4"
            run([
                "ffmpeg", "-y", "-v", "error", "-i", str(lab_paths.raw),
                "-filter_complex", f"[0:v]tpad=start_mode=clone:start_duration={delay:.3f}[v]",
                "-map", "[v]", "-map", "0:a", "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", str(variant),
            ])
            variants.append(variant)
            result = analyze_source(
                str(variant), f"{base_title} — known visual delay {delay:.2f}s", args, config
            )
            variant_jobs.append(str(result["paths"].root))
            delay_tests.append(score_delay(baseline["data"], result["data"], delay))

        calibration = summarize_calibration(control, delay_tests)
        data = {
            "engine": "AGAPE",
            "engine_full_name": "Audiovisual Gesture And Prosody Engine",
            "engine_version": "0.2.0",
            "kind": "self_supervised_calibration",
            "title": f"{base_title} — AGAPE Lab",
            "created_at": datetime.now(UTC).isoformat(),
            "source": source,
            "transcript_used": False,
            "baseline_job": str(baseline["paths"].root),
            "repeatability_job": str(repeat["paths"].root),
            "variant_jobs": variant_jobs,
            "control": control,
            "delay_tests": delay_tests,
            "result": calibration,
        }
        write_lab_reports(data, lab_paths)
        temporary_files = [lab_paths.raw, *variants]
        temporary_records = [
            {"path": path.name, "bytes": path.stat().st_size if path.exists() else 0}
            for path in temporary_files
        ]
        deleted = []
        if not config.retain_media:
            for path in temporary_files:
                if path.exists():
                    path.unlink()
                    deleted.append(path.name)
        finished = datetime.now(UTC)
        write_manifest(lab_paths, {
            "engine": "AGAPE",
            "engine_version": "0.2.0",
            "status": "complete",
            "kind": "self_supervised_calibration",
            "created_at": started.isoformat(),
            "completed_at": finished.isoformat(),
            "processing_seconds": (finished - started).total_seconds(),
            "analysis_jobs": [
                str(baseline["paths"].root), str(repeat["paths"].root), *variant_jobs,
            ],
            "retention": {
                "raw_media_retained": config.retain_media,
                "deleted_after_success": deleted,
                "temporary_media_before_cleanup": temporary_records,
                "temporary_media_bytes": sum(item["bytes"] for item in temporary_records),
                "policy": "self-supervision variants and source media deleted by default",
            },
        })
        print(json.dumps({
            "engine": "AGAPE",
            "lab_job": str(lab_paths.root),
            "report": str(lab_paths.report_html),
            "summary": str(lab_paths.report_md),
            "result": calibration,
            "media_deleted": deleted,
        }, indent=2))
        return 0
    except Exception:
        if not args.keep_failed_job:
            remove_failed_job(lab_paths)
        raise


def add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title")
    parser.add_argument("--segment-seconds", type=int)
    parser.add_argument("--segment-start-seconds", type=int, default=0)
    parser.add_argument(
        "--context", choices=("camera", "stage"), default="camera",
        help="Use camera for direct-to-lens recordings or stage for audience presentations",
    )
    parser.add_argument("--runs-root")
    parser.add_argument("--model")
    parser.add_argument("--retain-media", action="store_true", help="Retain raw/proxy/audio only with consent")
    parser.add_argument("--keep-failed-job", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agape")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze one local video or openly usable video URL")
    analyze.add_argument("source")
    add_analysis_options(analyze)
    analyze.set_defaults(func=analyze_command)

    compare = subparsers.add_parser("compare", help="Analyze and compare two takes from the same speaker")
    compare.add_argument("take_1")
    compare.add_argument("take_2")
    add_analysis_options(compare)
    compare.set_defaults(func=compare_command)

    feedback = subparsers.add_parser("feedback", help="Record human calibration for a report finding")
    feedback.add_argument("job", help="Completed AGAPE report directory")
    feedback.add_argument("--finding", type=int, required=True, help="1-based finding number")
    feedback.add_argument("--judgment", choices=("helpful", "wrong", "uncertain"), required=True)
    feedback.add_argument("--notes")
    feedback.set_defaults(func=feedback_command)

    lab = subparsers.add_parser("lab", help="Run self-supervised audiovisual timing calibration")
    lab.add_argument("source")
    lab.add_argument("--delays", default="0.25,0.50,0.75")
    add_analysis_options(lab)
    lab.set_defaults(func=lab_command, segment_seconds=30)

    from .training.commands import (
        demo_command,
        judge_command,
        score_command,
        train_command,
        training_data_command,
        local_pipeline_command,
        youtube_discover_command,
        youtube_ingest_command,
        youtube_train_command,
    )

    training_data = subparsers.add_parser(
        "training-data",
        help="Build a leakage-safe local dataset from AGAPE features.json files",
    )
    training_data.add_argument("inputs", nargs="+", help="Feature files or directories containing them")
    training_data.add_argument("--output")
    training_data.add_argument("--window-seconds", type=float, default=8.0)
    training_data.add_argument("--stride-seconds", type=float, default=2.0)
    training_data.add_argument("--offsets", default="-2,-1,-0.5,0.5,1,2")
    training_data.add_argument(
        "--minimum-alignment-gain", type=float, default=0.10,
        help="Keep paired negatives only when shifting weakens correlation evidence by this amount",
    )
    training_data.add_argument(
        "--group-map",
        action="append",
        help="JSON object mapping feature-file paths to speaker/session group labels; repeat to merge maps",
    )
    training_data.add_argument(
        "--group-map-only",
        action="store_true",
        help="Include only feature files named by the supplied group maps",
    )
    training_data.add_argument(
        "--split-plan",
        help="JSON object assigning every speaker/session group label to train, val, or test",
    )
    training_data.add_argument("--seed", type=int, default=17)
    training_data.set_defaults(func=training_data_command)

    train = subparsers.add_parser(
        "train", help="Train the local temporal synchrony model without API calls"
    )
    train.add_argument("dataset")
    train.add_argument("--output-root")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--blocks", type=int, default=3)
    train.add_argument("--dropout", type=float, default=0.12)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--pairwise-weight", type=float, default=0.5)
    train.add_argument("--pairwise-margin", type=float, default=1.0)
    train.add_argument("--no-group-balancing", action="store_true")
    train.add_argument("--no-explicit-correlations", action="store_true")
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    train.add_argument("--resume-from", help="Warm-start training from a previous AGAPE checkpoint .pt file")
    train.set_defaults(func=train_command)

    judge = subparsers.add_parser(
        "judge", help="Evaluate a checkpoint on the untouched test split and apply promotion gates"
    )
    judge.add_argument("checkpoint")
    judge.add_argument("dataset")
    judge.add_argument("--output")
    judge.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    judge.add_argument("--require-pass", action="store_true")
    judge.set_defaults(func=judge_command)

    score = subparsers.add_parser(
        "learned-score", help="Score one completed AGAPE feature timeline with a judged checkpoint"
    )
    score.add_argument("checkpoint")
    score.add_argument("features")
    score.add_argument("--output")
    score.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    score.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Allow inference with an unjudged or rejected checkpoint for research only",
    )
    score.set_defaults(func=score_command)

    demo = subparsers.add_parser(
        "train-demo", help="Run a generated-data smoke test of the complete local training path"
    )
    demo.add_argument("--epochs", type=int, default=18)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    demo.set_defaults(func=demo_command)

    local = subparsers.add_parser(
        "train-local",
        help="Build data, train on local compute, and judge the checkpoint in one command",
    )
    local.add_argument("inputs", nargs="+", help="Feature files or directories containing them")
    local.add_argument("--dataset")
    local.add_argument("--output-root")
    local.add_argument("--group-map", action="append")
    local.add_argument("--group-map-only", action="store_true")
    local.add_argument("--split-plan")
    local.add_argument("--window-seconds", type=float, default=8.0)
    local.add_argument("--stride-seconds", type=float, default=2.0)
    local.add_argument("--offsets", default="-2,-1,-0.5,0.5,1,2")
    local.add_argument("--minimum-alignment-gain", type=float, default=0.10)
    local.add_argument("--epochs", type=int, default=60)
    local.add_argument("--batch-size", type=int, default=64)
    local.add_argument("--learning-rate", type=float, default=2e-3)
    local.add_argument("--hidden-size", type=int, default=48)
    local.add_argument("--blocks", type=int, default=3)
    local.add_argument("--patience", type=int, default=10)
    local.add_argument("--pairwise-weight", type=float, default=0.5)
    local.add_argument("--pairwise-margin", type=float, default=1.0)
    local.add_argument("--no-group-balancing", action="store_true")
    local.add_argument("--no-explicit-correlations", action="store_true")
    local.add_argument("--seed", type=int, default=17)
    local.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    local.add_argument("--resume-from", help="Warm-start training from a previous AGAPE checkpoint .pt file")
    local.add_argument("--require-pass", action="store_true")
    local.set_defaults(func=local_pipeline_command)

    youtube_discover = subparsers.add_parser(
        "youtube-discover",
        help="Find Creative Commons YouTube candidates using the API or no-key metadata search",
    )
    youtube_discover.add_argument("query")
    youtube_discover.add_argument("--output")
    youtube_discover.add_argument("--backend", choices=("auto", "api", "yt-dlp"), default="auto")
    youtube_discover.add_argument("--api-key-env", default="AGAPE_YOUTUBE_API_KEY")
    youtube_discover.add_argument("--max-results", type=int, default=25)
    youtube_discover.add_argument("--max-per-channel", type=int, default=3)
    youtube_discover.add_argument("--metadata-workers", type=int, default=4)
    youtube_discover.add_argument("--segment-start-seconds", type=int, default=15)
    youtube_discover.add_argument("--segment-seconds", type=int, default=90)
    youtube_discover.add_argument("--region-code")
    youtube_discover.add_argument("--relevance-language")
    youtube_discover.set_defaults(func=youtube_discover_command)

    def add_youtube_quality_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--minimum-visual-coverage", type=float, default=0.60)
        command.add_argument("--minimum-stage-face-coverage", type=float, default=0.25)
        command.add_argument("--minimum-speech-ratio", type=float, default=0.30)
        command.add_argument("--minimum-emphasis-events", type=int, default=3)
        command.add_argument("--fail-fast", action="store_true")

    youtube_ingest = subparsers.add_parser(
        "youtube-ingest",
        help="Analyze an attested YouTube manifest, delete media, and retain accepted features",
    )
    youtube_ingest.add_argument("manifest")
    youtube_ingest.add_argument("--output-root")
    add_youtube_quality_options(youtube_ingest)
    youtube_ingest.set_defaults(func=youtube_ingest_command)

    youtube_train = subparsers.add_parser(
        "youtube-train",
        help="Ingest an attested YouTube manifest, train locally, and judge the checkpoint",
    )
    youtube_train.add_argument("manifest")
    youtube_train.add_argument("--ingest-root")
    youtube_train.add_argument("--dataset")
    youtube_train.add_argument("--output-root")
    youtube_train.add_argument("--split-plan")
    youtube_train.add_argument("--window-seconds", type=float, default=8.0)
    youtube_train.add_argument("--stride-seconds", type=float, default=2.0)
    youtube_train.add_argument("--offsets", default="-2,-1,-0.5,0.5,1,2")
    youtube_train.add_argument("--minimum-alignment-gain", type=float, default=0.10)
    youtube_train.add_argument("--epochs", type=int, default=60)
    youtube_train.add_argument("--batch-size", type=int, default=64)
    youtube_train.add_argument("--learning-rate", type=float, default=2e-3)
    youtube_train.add_argument("--hidden-size", type=int, default=48)
    youtube_train.add_argument("--blocks", type=int, default=3)
    youtube_train.add_argument("--patience", type=int, default=10)
    youtube_train.add_argument("--pairwise-weight", type=float, default=0.5)
    youtube_train.add_argument("--pairwise-margin", type=float, default=1.0)
    youtube_train.add_argument("--no-group-balancing", action="store_true")
    youtube_train.add_argument("--no-explicit-correlations", action="store_true")
    youtube_train.add_argument("--seed", type=int, default=17)
    youtube_train.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    youtube_train.add_argument("--resume-from", help="Warm-start training from a previous AGAPE checkpoint .pt file")
    youtube_train.add_argument("--require-pass", action="store_true")
    add_youtube_quality_options(youtube_train)
    youtube_train.set_defaults(func=youtube_train_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"AGAPE failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
