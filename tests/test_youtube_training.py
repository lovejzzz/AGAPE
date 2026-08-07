import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from audit_engine.media import yt_dlp_runtime_args
from audit_engine.training.youtube import (
    discover_creative_commons_with_ytdlp,
    extract_video_id,
    ingest_youtube_manifest,
    load_youtube_manifest,
    quality_assessment,
    resolve_youtube_api_key,
    verify_youtube_metadata,
)


VIDEO_ID = "dQw4w9WgXcQ"


def manifest_payload(confirmed: bool = True) -> dict:
    return {
        "kind": "agape_youtube_training_manifest",
        "schema_version": 1,
        "reuse_attestation": {"confirmed": confirmed},
        "items": [{
            "selected": True,
            "url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
            "video_id": VIDEO_ID,
            "group": "speaker-session-01",
            "title": "Example",
            "channel": "Example channel",
            "channel_id": "channel-01",
            "segment_start_seconds": 10,
            "segment_seconds": 60,
            "context": "camera",
            "rights_basis": "creative_commons",
            "detected_license": "creativeCommon",
            "permission_reference": "",
        }],
    }


def test_extract_video_id_supports_normal_short_and_shorts_urls():
    assert extract_video_id(f"https://www.youtube.com/watch?v={VIDEO_ID}") == VIDEO_ID
    assert extract_video_id(f"https://youtu.be/{VIDEO_ID}") == VIDEO_ID
    assert extract_video_id(f"https://youtube.com/shorts/{VIDEO_ID}") == VIDEO_ID


def test_node_runtime_is_enabled_when_deno_is_unavailable(monkeypatch):
    import audit_engine.media as media

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/local/bin/node" if name == "node" else None)
    assert yt_dlp_runtime_args() == ["--js-runtimes", "node:/usr/local/bin/node"]


def test_manifest_requires_explicit_reuse_attestation(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload(confirmed=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="reuse_attestation"):
        load_youtube_manifest(path)
    payload, items = load_youtube_manifest(path, require_attestation=False)
    assert payload["schema_version"] == 1
    assert items[0].group == "speaker-session-01"


def test_current_metadata_must_confirm_creative_commons(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload()), encoding="utf-8")
    _, items = load_youtube_manifest(path)
    metadata = {
        "id": VIDEO_ID,
        "duration": 120,
        "license": "Standard YouTube License",
        "age_limit": 0,
        "live_status": "not_live",
    }
    with pytest.raises(ValueError, match="Creative Commons"):
        verify_youtube_metadata(items[0], metadata)
    metadata["license"] = None
    with pytest.raises(ValueError, match="Creative Commons"):
        verify_youtube_metadata(items[0], metadata)


def test_quality_assessment_rejects_low_pose_coverage():
    features = {
        "vision": {"summary": {"face_coverage": 0.9, "pose_coverage": 0.2}},
        "audio": {"summary": {"speech_ratio": 0.8}},
        "fusion": {"summary": {"duration_seconds": 60, "vocal_emphasis_events": 8}},
    }
    result = quality_assessment(features)
    assert not result["passed"]
    assert not result["checks"]["pose_coverage"]


def test_stage_quality_allows_partial_face_when_pose_is_strong():
    features = {
        "vision": {"summary": {"face_coverage": 0.35, "pose_coverage": 0.9}},
        "audio": {"summary": {"speech_ratio": 0.8}},
        "fusion": {"summary": {"duration_seconds": 60, "vocal_emphasis_events": 8}},
    }
    result = quality_assessment(features, coaching_context="stage")
    assert result["passed"]
    assert result["thresholds"]["minimum_face_coverage"] == 0.25


def test_ingestion_keeps_only_accepted_features_and_writes_attribution(tmp_path, monkeypatch):
    from audit_engine.config import AuditConfig
    import audit_engine.cli
    import audit_engine.training.youtube as youtube

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload()), encoding="utf-8")
    monkeypatch.setattr(youtube, "probe_youtube", lambda _url: {
        "id": VIDEO_ID,
        "title": "Example",
        "channel": "Example channel",
        "channel_id": "channel-01",
        "duration": 120,
        "license": "Creative Commons Attribution license",
        "age_limit": 0,
        "live_status": "not_live",
        "webpage_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
    })

    def fake_analyze(_source, _title, _args, config):
        job = config.runs_root / "job-01"
        job.mkdir(parents=True)
        features_path = job / "features.json"
        features_path.write_text(json.dumps({
            "vision": {"summary": {"face_coverage": 0.9, "pose_coverage": 0.85}},
            "audio": {"summary": {"speech_ratio": 0.8}},
            "fusion": {"summary": {"duration_seconds": 60, "vocal_emphasis_events": 8}},
        }), encoding="utf-8")
        return {
            "paths": SimpleNamespace(root=job, features=features_path),
            "media_deleted": ["source.mp4", "analysis-proxy.mp4", "analysis-audio.wav"],
        }

    monkeypatch.setattr(audit_engine.cli, "analyze_source", fake_analyze)
    model_path = tmp_path / "model.task"
    model_path.touch()
    result = ingest_youtube_manifest(
        manifest,
        AuditConfig(runs_root=tmp_path / "runs", model_path=model_path),
        tmp_path / "youtube-batches",
    )
    assert result["summary"]["accepted"] == 1
    assert len(result["accepted_features"]) == 1
    assert result["outcomes"] == [{
        "video_id": VIDEO_ID,
        "title": "Example",
        "status": "accepted",
        "error": None,
        "failed_quality_checks": [],
    }]
    assert "Example channel" in (tmp_path / "youtube-batches" / Path(result["batch_directory"]).name / "ATTRIBUTION.md").read_text()


def test_no_key_discovery_filters_using_current_metadata(tmp_path, monkeypatch):
    import audit_engine.training.youtube as youtube

    monkeypatch.setattr(youtube, "find_yt_dlp", lambda: "yt-dlp")
    monkeypatch.setattr(youtube, "run", lambda _command: SimpleNamespace(stdout=json.dumps({
        "entries": [{"id": VIDEO_ID, "title": "Example", "channel": "Example channel"}],
    })))
    monkeypatch.setattr(youtube, "probe_youtube", lambda _url: {
        "id": VIDEO_ID,
        "title": "Example",
        "channel": "Example channel",
        "channel_id": "channel-01",
        "duration": 120,
        "license": "Creative Commons Attribution license (reuse allowed)",
        "age_limit": 0,
        "live_status": "not_live",
    })
    output = tmp_path / "discovered.json"
    result = discover_creative_commons_with_ytdlp("speaking", output, max_results=1)
    assert result["items"][0]["video_id"] == VIDEO_ID
    assert not result["reuse_attestation"]["confirmed"]
    assert output.exists()


def test_environment_api_key_takes_priority(monkeypatch):
    monkeypatch.setenv("TEST_YOUTUBE_KEY", "environment-secret")
    assert resolve_youtube_api_key("TEST_YOUTUBE_KEY") == (
        "environment-secret",
        "environment",
    )
