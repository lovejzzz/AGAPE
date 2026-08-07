from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_group_map(path: str | list[str] | None) -> dict[str, str] | None:
    if not path:
        return None
    paths = [path] if isinstance(path, str) else path
    merged: dict[str, str] = {}
    for item in paths:
        payload = json.loads(Path(item).expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value.strip()
            for key, value in payload.items()
        ):
            raise ValueError("group map must be a JSON object from feature paths to non-empty group labels")
        conflicts = {
            key for key, value in payload.items() if key in merged and merged[key] != value
        }
        if conflicts:
            raise ValueError(f"group maps contain conflicting labels for: {sorted(conflicts)}")
        merged.update(payload)
    return merged


def load_split_plan(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value in {"train", "val", "test"}
        for key, value in payload.items()
    ):
        raise ValueError("split plan must be a JSON object from group labels to train, val, or test")
    return payload


def training_data_command(args) -> int:
    from .data import build_dataset, parse_offsets

    output = Path(args.output).expanduser().resolve() if args.output else project_root() / "training_data" / "synchrony-v1.npz"
    result = build_dataset(
        args.inputs,
        output,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        offsets_seconds=parse_offsets(args.offsets),
        seed=args.seed,
        group_map=load_group_map(args.group_map),
        group_map_only=args.group_map_only,
        split_plan=load_split_plan(args.split_plan),
        minimum_alignment_gain=args.minimum_alignment_gain,
    )
    print(json.dumps(result, indent=2))
    return 0


def train_command(args) -> int:
    from .runner import train_model

    result = train_model(
        Path(args.dataset),
        Path(args.output_root) if args.output_root else project_root() / "training_runs",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        blocks=args.blocks,
        dropout=args.dropout,
        patience=args.patience,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        group_balanced=not args.no_group_balancing,
        explicit_correlations=not args.no_explicit_correlations,
        seed=args.seed,
        requested_device=args.device,
        resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2))
    return 0


def judge_command(args) -> int:
    from .judge import judge_checkpoint

    report = judge_checkpoint(
        Path(args.checkpoint),
        Path(args.dataset),
        output_directory=Path(args.output) if args.output else None,
        requested_device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 2 if args.require_pass and report["verdict"] != "PROMOTE" else 0


def score_command(args) -> int:
    from .inference import score_features

    result = score_features(
        Path(args.checkpoint),
        Path(args.features),
        output_path=Path(args.output) if args.output else None,
        requested_device=args.device,
        allow_experimental=args.allow_experimental,
    )
    print(json.dumps(result, indent=2))
    return 0


def demo_command(args) -> int:
    from .demo import run_demo

    result = run_demo(
        project_root(),
        epochs=args.epochs,
        requested_device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


def local_pipeline_command(args) -> int:
    from .data import build_dataset, parse_offsets
    from .judge import judge_checkpoint
    from .runner import train_model

    root = project_root()
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else root / "training_data" / "synchrony-v1.npz"
    dataset = build_dataset(
        args.inputs,
        dataset_path,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        offsets_seconds=parse_offsets(args.offsets),
        seed=args.seed,
        group_map=load_group_map(args.group_map),
        group_map_only=args.group_map_only,
        split_plan=load_split_plan(args.split_plan),
        minimum_alignment_gain=args.minimum_alignment_gain,
    )
    training = train_model(
        dataset_path,
        Path(args.output_root) if args.output_root else root / "training_runs",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        blocks=args.blocks,
        patience=args.patience,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        group_balanced=not args.no_group_balancing,
        explicit_correlations=not args.no_explicit_correlations,
        seed=args.seed,
        requested_device=args.device,
        resume_from=args.resume_from,
    )
    judgment = judge_checkpoint(
        Path(training["best_checkpoint"]),
        dataset_path,
        requested_device=args.device,
    )
    print(json.dumps({"dataset": dataset, "training": training, "judgment": judgment}, indent=2))
    return 2 if args.require_pass and judgment["verdict"] != "PROMOTE" else 0


def youtube_discover_command(args) -> int:
    from .youtube import discover_youtube_candidates

    output = Path(args.output).expanduser().resolve() if args.output else project_root() / "training_data" / "youtube-manifest.json"
    result = discover_youtube_candidates(
        args.query,
        output,
        backend=args.backend,
        api_key_env=args.api_key_env,
        max_results=args.max_results,
        max_per_channel=args.max_per_channel,
        segment_start_seconds=args.segment_start_seconds,
        segment_seconds=args.segment_seconds,
        region_code=args.region_code,
        relevance_language=args.relevance_language,
        metadata_workers=args.metadata_workers,
    )
    print(json.dumps(result, indent=2))
    return 0


def youtube_ingest_command(args) -> int:
    from ..config import AuditConfig
    from .youtube import ingest_youtube_manifest

    root = project_root()
    config = AuditConfig(
        runs_root=root / "runs",
        model_path=root / "models" / "holistic_landmarker.task",
    )
    if not config.model_path.exists():
        raise FileNotFoundError(f"MediaPipe model is missing: {config.model_path}")
    result = ingest_youtube_manifest(
        Path(args.manifest),
        config,
        Path(args.output_root) if args.output_root else root / "runs" / "youtube-training",
        minimum_visual_coverage=args.minimum_visual_coverage,
        minimum_stage_face_coverage=args.minimum_stage_face_coverage,
        minimum_speech_ratio=args.minimum_speech_ratio,
        minimum_emphasis_events=args.minimum_emphasis_events,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(result, indent=2))
    return 0


def youtube_train_command(args) -> int:
    from ..config import AuditConfig
    from .data import build_dataset, parse_offsets
    from .judge import judge_checkpoint
    from .runner import train_model
    from .youtube import ingest_youtube_manifest

    root = project_root()
    config = AuditConfig(
        runs_root=root / "runs",
        model_path=root / "models" / "holistic_landmarker.task",
    )
    if not config.model_path.exists():
        raise FileNotFoundError(f"MediaPipe model is missing: {config.model_path}")
    ingestion = ingest_youtube_manifest(
        Path(args.manifest),
        config,
        Path(args.ingest_root) if args.ingest_root else root / "runs" / "youtube-training",
        minimum_visual_coverage=args.minimum_visual_coverage,
        minimum_stage_face_coverage=args.minimum_stage_face_coverage,
        minimum_speech_ratio=args.minimum_speech_ratio,
        minimum_emphasis_events=args.minimum_emphasis_events,
        fail_fast=args.fail_fast,
    )
    if len(ingestion["accepted_features"]) < 3:
        raise ValueError(
            f"only {len(ingestion['accepted_features'])} YouTube segments passed; at least three independent sources are required"
        )
    batch_directory = Path(ingestion["batch_directory"])
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else batch_directory / "synchrony.npz"
    dataset = build_dataset(
        ingestion["accepted_features"],
        dataset_path,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        offsets_seconds=parse_offsets(args.offsets),
        seed=args.seed,
        group_map=ingestion["group_map"],
        split_plan=load_split_plan(args.split_plan),
        minimum_alignment_gain=args.minimum_alignment_gain,
    )
    training = train_model(
        dataset_path,
        Path(args.output_root) if args.output_root else root / "training_runs" / "youtube",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        blocks=args.blocks,
        patience=args.patience,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        group_balanced=not args.no_group_balancing,
        explicit_correlations=not args.no_explicit_correlations,
        seed=args.seed,
        requested_device=args.device,
        resume_from=args.resume_from,
    )
    judgment = judge_checkpoint(
        Path(training["best_checkpoint"]),
        dataset_path,
        requested_device=args.device,
    )
    result = {
        "ingestion": ingestion,
        "dataset": dataset,
        "training": training,
        "judgment": judgment,
    }
    print(json.dumps(result, indent=2))
    return 2 if args.require_pass and judgment["verdict"] != "PROMOTE" else 0
