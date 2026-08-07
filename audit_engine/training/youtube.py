from __future__ import annotations

import html
import getpass
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from ..config import AuditConfig
from ..media import run, yt_dlp_runtime_args
from .data import sha256_file


YOUTUBE_MANIFEST_SCHEMA_VERSION = 1
RIGHTS_BASES = {"creative_commons", "owned", "permission"}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


@dataclass(frozen=True)
class YouTubeItem:
    url: str
    video_id: str
    group: str
    title: str
    channel: str
    channel_id: str
    segment_start_seconds: int
    segment_seconds: int
    context: str
    rights_basis: str
    detected_license: str
    permission_reference: str
    selected: bool = True
    made_for_kids: bool = False

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.segment_start_seconds}:{self.segment_seconds}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_youtube_api_key(api_key_env: str) -> tuple[str | None, str | None]:
    api_key = os.environ.get(api_key_env)
    if api_key:
        return api_key, "environment"
    if sys.platform == "darwin" and shutil.which("security"):
        try:
            result = run([
                "security",
                "find-generic-password",
                "-a",
                getpass.getuser(),
                "-s",
                api_key_env,
                "-w",
            ])
            api_key = result.stdout.strip()
            if api_key:
                return api_key, "macOS Keychain"
        except Exception:
            pass
    return None, None


def extract_video_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if host not in YOUTUBE_HOSTS:
        raise ValueError(f"not a supported YouTube URL: {url}")
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path == "/watch":
        candidate = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        candidate = parsed.path.strip("/").split("/", 1)[1].split("/", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        raise ValueError(f"could not identify one YouTube video from URL: {url}")
    return candidate


def parse_iso8601_duration(value: str) -> int:
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?",
        value,
    )
    if not match:
        return 0
    parts = {key: int(item or 0) for key, item in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def youtube_api_get(endpoint: str, parameters: dict[str, str | int], api_key: str) -> dict:
    query = urllib.parse.urlencode({**parameters, "key": api_key})
    request = urllib.request.Request(
        f"https://www.googleapis.com/youtube/v3/{endpoint}?{query}",
        headers={"Accept": "application/json", "User-Agent": "AGAPE-offline-training/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_creative_commons(
    query: str,
    output_path: Path,
    *,
    api_key_env: str = "AGAPE_YOUTUBE_API_KEY",
    max_results: int = 25,
    max_per_channel: int = 3,
    segment_start_seconds: int = 15,
    segment_seconds: int = 90,
    region_code: str | None = None,
    relevance_language: str | None = None,
) -> dict:
    api_key, api_key_source = resolve_youtube_api_key(api_key_env)
    if not api_key:
        raise ValueError(
            f"set {api_key_env} or store it in macOS Keychain; the key is never written to project files"
        )
    if not query.strip() or not 1 <= max_results <= 200:
        raise ValueError("query is required and max-results must be between 1 and 200")
    if max_per_channel < 1 or segment_start_seconds < 0 or not 30 <= segment_seconds <= 180:
        raise ValueError("max-per-channel must be positive; start must be nonnegative; segments must be 30–180 seconds")

    search_items: list[dict] = []
    page_token: str | None = None
    while len(search_items) < max_results * 3:
        parameters: dict[str, str | int] = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoLicense": "creativeCommon",
            "videoEmbeddable": "true",
            "safeSearch": "strict",
            "maxResults": min(50, max_results * 3 - len(search_items)),
        }
        if page_token:
            parameters["pageToken"] = page_token
        if region_code:
            parameters["regionCode"] = region_code
        if relevance_language:
            parameters["relevanceLanguage"] = relevance_language
        page = youtube_api_get("search", parameters, api_key)
        search_items.extend(page.get("items", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            break

    ids = [str(item.get("id", {}).get("videoId", "")) for item in search_items]
    ids = [item for item in ids if re.fullmatch(r"[A-Za-z0-9_-]{11}", item)]
    details: dict[str, dict] = {}
    for start in range(0, len(ids), 50):
        page = youtube_api_get(
            "videos",
            {
                "part": "snippet,contentDetails,status",
                "id": ",".join(ids[start:start + 50]),
                "maxResults": 50,
            },
            api_key,
        )
        details.update({str(item["id"]): item for item in page.get("items", [])})

    channel_counts: dict[str, int] = {}
    items: list[dict] = []
    skipped = {"not_creative_commons": 0, "age_restricted": 0, "made_for_kids": 0, "too_short": 0, "channel_cap": 0}
    for video_id in ids:
        detail = details.get(video_id)
        if not detail:
            continue
        status = detail.get("status", {})
        content = detail.get("contentDetails", {})
        snippet = detail.get("snippet", {})
        if status.get("license") != "creativeCommon":
            skipped["not_creative_commons"] += 1
            continue
        if content.get("contentRating", {}).get("ytRating") == "ytAgeRestricted":
            skipped["age_restricted"] += 1
            continue
        if bool(status.get("madeForKids")):
            skipped["made_for_kids"] += 1
            continue
        duration = parse_iso8601_duration(str(content.get("duration", "")))
        usable_seconds = min(segment_seconds, duration - segment_start_seconds)
        if usable_seconds < 30:
            skipped["too_short"] += 1
            continue
        channel_id = str(snippet.get("channelId") or f"unknown-{video_id}")
        if channel_counts.get(channel_id, 0) >= max_per_channel:
            skipped["channel_cap"] += 1
            continue
        channel_counts[channel_id] = channel_counts.get(channel_id, 0) + 1
        items.append({
            "selected": True,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "group": f"channel:{channel_id}",
            "title": html.unescape(str(snippet.get("title") or video_id)),
            "channel": html.unescape(str(snippet.get("channelTitle") or "Unknown channel")),
            "channel_id": channel_id,
            "segment_start_seconds": segment_start_seconds,
            "segment_seconds": int(usable_seconds),
            "context": "stage",
            "rights_basis": "creative_commons",
            "detected_license": "creativeCommon",
            "permission_reference": "",
            "made_for_kids": False,
            "published_at": snippet.get("publishedAt"),
            "duration_seconds": duration,
        })
        if len(items) >= max_results:
            break

    manifest = {
        "kind": "agape_youtube_training_manifest",
        "schema_version": YOUTUBE_MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now(),
        "discovery": {
            "query": query,
            "license_filter": "creativeCommon",
            "safe_search": "strict",
            "region_code": region_code,
            "relevance_language": relevance_language,
            "credential_source": api_key_source,
            "skipped": skipped,
            "note": "YouTube API search and metadata consume ordinary API quota, not AI tokens.",
        },
        "reuse_attestation": {
            "confirmed": False,
            "confirmed_by": "",
            "confirmed_at": "",
            "notes": "Review the candidates, speaker/session groups, CC BY attribution duties, and intended research use before setting confirmed to true.",
        },
        "items": items,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["output"] = str(output_path)
    return manifest


def discover_creative_commons_with_ytdlp(
    query: str,
    output_path: Path,
    *,
    max_results: int = 25,
    max_per_channel: int = 3,
    segment_start_seconds: int = 15,
    segment_seconds: int = 90,
    metadata_workers: int = 4,
) -> dict:
    if not query.strip() or not 1 <= max_results <= 100:
        raise ValueError("query is required and yt-dlp max-results must be between 1 and 100")
    if max_per_channel < 1 or segment_start_seconds < 0 or not 30 <= segment_seconds <= 180:
        raise ValueError("max-per-channel must be positive; start must be nonnegative; segments must be 30–180 seconds")
    if not 1 <= metadata_workers <= 8:
        raise ValueError("metadata-workers must be between 1 and 8")
    candidate_limit = min(200, max(max_results * 5, 25))
    search = run([
        find_yt_dlp(),
        *yt_dlp_runtime_args(),
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-end",
        str(candidate_limit),
        "--no-warnings",
        f"ytsearch{candidate_limit}:{query}",
    ])
    search_payload = json.loads(search.stdout)
    entries = search_payload.get("entries", [])
    channel_counts: dict[str, int] = {}
    items: list[dict] = []
    skipped = {
        "metadata_failed": 0,
        "not_creative_commons": 0,
        "age_restricted": 0,
        "live": 0,
        "too_short": 0,
        "channel_cap": 0,
    }
    def inspect(entry: dict) -> tuple[dict, dict | None]:
        video_id = str(entry.get("id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return entry, None
        try:
            return entry, probe_youtube(f"https://www.youtube.com/watch?v={video_id}")
        except Exception:
            return entry, None

    with ThreadPoolExecutor(max_workers=metadata_workers) as executor:
        inspected = list(executor.map(inspect, entries))
    for entry, metadata in inspected:
        video_id = str(entry.get("id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            continue
        url = f"https://www.youtube.com/watch?v={video_id}"
        if metadata is None:
            skipped["metadata_failed"] += 1
            continue
        license_value = str(metadata.get("license") or "").lower().replace(" ", "")
        if "creativecommons" not in license_value and "creativecommon" not in license_value:
            skipped["not_creative_commons"] += 1
            continue
        if int(metadata.get("age_limit") or 0) > 0:
            skipped["age_restricted"] += 1
            continue
        if metadata.get("is_live") or metadata.get("live_status") == "is_live":
            skipped["live"] += 1
            continue
        duration = int(float(metadata.get("duration") or 0))
        usable_seconds = min(segment_seconds, duration - segment_start_seconds)
        if usable_seconds < 30:
            skipped["too_short"] += 1
            continue
        channel_id = str(metadata.get("channel_id") or metadata.get("uploader_id") or f"unknown-{video_id}")
        if channel_counts.get(channel_id, 0) >= max_per_channel:
            skipped["channel_cap"] += 1
            continue
        channel_counts[channel_id] = channel_counts.get(channel_id, 0) + 1
        items.append({
            "selected": True,
            "url": url,
            "video_id": video_id,
            "group": f"channel:{channel_id}",
            "title": str(metadata.get("title") or entry.get("title") or video_id),
            "channel": str(metadata.get("channel") or metadata.get("uploader") or entry.get("channel") or "Unknown channel"),
            "channel_id": channel_id,
            "segment_start_seconds": segment_start_seconds,
            "segment_seconds": int(usable_seconds),
            "context": "stage",
            "rights_basis": "creative_commons",
            "detected_license": str(metadata.get("license")),
            "permission_reference": "",
            "made_for_kids": False,
            "duration_seconds": duration,
        })
        if len(items) >= max_results:
            break

    manifest = {
        "kind": "agape_youtube_training_manifest",
        "schema_version": YOUTUBE_MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now(),
        "discovery": {
            "query": query,
            "backend": "yt-dlp metadata search",
            "metadata_workers": metadata_workers,
            "license_filter": "post-search current metadata verification",
            "safe_search": "not exposed by this backend",
            "skipped": skipped,
            "note": "No API key and no video download were used for discovery. Review is required because this backend cannot verify made-for-kids status.",
        },
        "reuse_attestation": {
            "confirmed": False,
            "confirmed_by": "",
            "confirmed_at": "",
            "notes": "Review candidates, adult-speaker suitability, groups, CC BY attribution duties, and intended research use before confirming.",
        },
        "items": items,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["output"] = str(output_path)
    return manifest


def discover_youtube_candidates(
    query: str,
    output_path: Path,
    *,
    backend: str = "auto",
    api_key_env: str = "AGAPE_YOUTUBE_API_KEY",
    max_results: int = 25,
    max_per_channel: int = 3,
    segment_start_seconds: int = 15,
    segment_seconds: int = 90,
    region_code: str | None = None,
    relevance_language: str | None = None,
    metadata_workers: int = 4,
) -> dict:
    if backend not in {"auto", "api", "yt-dlp"}:
        raise ValueError("backend must be auto, api, or yt-dlp")
    api_key, _ = resolve_youtube_api_key(api_key_env)
    use_api = backend == "api" or (backend == "auto" and bool(api_key))
    if use_api:
        return discover_creative_commons(
            query,
            output_path,
            api_key_env=api_key_env,
            max_results=max_results,
            max_per_channel=max_per_channel,
            segment_start_seconds=segment_start_seconds,
            segment_seconds=segment_seconds,
            region_code=region_code,
            relevance_language=relevance_language,
        )
    return discover_creative_commons_with_ytdlp(
        query,
        output_path,
        max_results=max_results,
        max_per_channel=max_per_channel,
        segment_start_seconds=segment_start_seconds,
        segment_seconds=segment_seconds,
        metadata_workers=metadata_workers,
    )


def _item_from_payload(payload: dict, index: int) -> YouTubeItem:
    if not isinstance(payload, dict):
        raise ValueError(f"manifest item {index} must be an object")
    selected = bool(payload.get("selected", True))
    url = str(payload.get("url", ""))
    video_id = extract_video_id(url)
    declared_id = str(payload.get("video_id") or video_id)
    if declared_id != video_id:
        raise ValueError(f"manifest item {index} video_id does not match its URL")
    group = str(payload.get("group", "")).strip()
    if selected and not group:
        raise ValueError(f"manifest item {index} needs a speaker/session group")
    rights_basis = str(payload.get("rights_basis", "")).strip()
    if selected and rights_basis not in RIGHTS_BASES:
        raise ValueError(f"manifest item {index} rights_basis must be one of {sorted(RIGHTS_BASES)}")
    permission_reference = str(payload.get("permission_reference", "")).strip()
    if selected and rights_basis in {"owned", "permission"} and not permission_reference:
        raise ValueError(f"manifest item {index} needs a permission_reference")
    start = int(payload.get("segment_start_seconds", 0))
    seconds = int(payload.get("segment_seconds", 90))
    if selected and (start < 0 or not 30 <= seconds <= 180):
        raise ValueError(f"manifest item {index} must use a nonnegative start and a 30–180 second segment")
    context = str(payload.get("context", "stage"))
    if context not in {"camera", "stage"}:
        raise ValueError(f"manifest item {index} context must be camera or stage")
    return YouTubeItem(
        url=url,
        video_id=video_id,
        group=group,
        title=str(payload.get("title") or video_id),
        channel=str(payload.get("channel") or "Unknown channel"),
        channel_id=str(payload.get("channel_id") or ""),
        segment_start_seconds=start,
        segment_seconds=seconds,
        context=context,
        rights_basis=rights_basis,
        detected_license=str(payload.get("detected_license") or ""),
        permission_reference=permission_reference,
        selected=selected,
        made_for_kids=bool(payload.get("made_for_kids", False)),
    )


def load_youtube_manifest(path: Path, *, require_attestation: bool = True) -> tuple[dict, list[YouTubeItem]]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "agape_youtube_training_manifest":
        raise ValueError("not an AGAPE YouTube training manifest")
    if int(payload.get("schema_version", -1)) != YOUTUBE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported YouTube training manifest schema")
    attestation = payload.get("reuse_attestation", {})
    if require_attestation and not bool(attestation.get("confirmed")):
        raise ValueError(
            "reuse_attestation.confirmed is false; review rights, privacy, attribution, and grouping before ingestion"
        )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("manifest needs at least one item")
    items = [_item_from_payload(item, index) for index, item in enumerate(raw_items, start=1)]
    selected = [item for item in items if item.selected]
    if not selected:
        raise ValueError("manifest has no selected items")
    keys = [item.key for item in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate video segments")
    if any(item.made_for_kids for item in selected):
        raise ValueError("child-directed videos are excluded from this training pipeline")
    return payload, selected


def find_yt_dlp() -> str:
    command = shutil.which("yt-dlp")
    if command:
        return command
    candidate = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "yt-dlp"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("yt-dlp is not installed; run scripts/bootstrap-training.sh")


def probe_youtube(url: str) -> dict:
    result = run([
        find_yt_dlp(),
        *yt_dlp_runtime_args(),
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        url,
    ])
    return json.loads(result.stdout)


def reduced_metadata(metadata: dict) -> dict:
    return {
        "id": metadata.get("id"),
        "title": metadata.get("title"),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "channel_id": metadata.get("channel_id") or metadata.get("uploader_id"),
        "duration": metadata.get("duration"),
        "license": metadata.get("license"),
        "availability": metadata.get("availability"),
        "live_status": metadata.get("live_status"),
        "age_limit": metadata.get("age_limit"),
        "webpage_url": metadata.get("webpage_url"),
    }


def verify_youtube_metadata(item: YouTubeItem, metadata: dict) -> None:
    if str(metadata.get("id")) != item.video_id:
        raise ValueError("yt-dlp metadata ID does not match the manifest")
    if metadata.get("is_live") or metadata.get("live_status") == "is_live":
        raise ValueError("live streams are not accepted; use a completed recording")
    if int(metadata.get("age_limit") or 0) > 0:
        raise ValueError("age-restricted videos are excluded")
    duration = float(metadata.get("duration") or 0)
    if duration and duration - item.segment_start_seconds < 30:
        raise ValueError("the selected start leaves less than 30 seconds of usable video")
    if item.rights_basis == "creative_commons":
        license_value = str(metadata.get("license") or "").lower().replace(" ", "")
        if "creativecommons" not in license_value and "creativecommon" not in license_value:
            raise ValueError("current metadata does not verify a Creative Commons license")


def quality_assessment(
    features: dict,
    *,
    minimum_visual_coverage: float = 0.60,
    minimum_stage_face_coverage: float = 0.25,
    minimum_speech_ratio: float = 0.30,
    minimum_emphasis_events: int = 3,
    coaching_context: str | None = None,
) -> dict:
    vision = features.get("vision", {}).get("summary", {})
    audio = features.get("audio", {}).get("summary", {})
    fusion = features.get("fusion", {}).get("summary", {})
    face = float(vision.get("face_coverage") or 0.0)
    pose = float(vision.get("pose_coverage") or 0.0)
    speech = float(audio.get("speech_ratio") or 0.0)
    emphasis = int(fusion.get("vocal_emphasis_events") or 0)
    duration = float(fusion.get("duration_seconds") or 0.0)
    context = coaching_context or str(features.get("coaching_context") or "camera")
    minimum_face_coverage = (
        minimum_stage_face_coverage if context == "stage" else minimum_visual_coverage
    )
    checks = {
        "duration": duration >= 30.0,
        "face_coverage": face >= minimum_face_coverage,
        "pose_coverage": pose >= minimum_visual_coverage,
        "speech_ratio": speech >= minimum_speech_ratio,
        "emphasis_events": emphasis >= minimum_emphasis_events,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "duration_seconds": duration,
            "face_coverage": face,
            "pose_coverage": pose,
            "speech_ratio": speech,
            "vocal_emphasis_events": emphasis,
        },
        "thresholds": {
            "minimum_duration_seconds": 30.0,
            "coaching_context": context,
            "minimum_face_coverage": minimum_face_coverage,
            "minimum_pose_coverage": minimum_visual_coverage,
            "minimum_speech_ratio": minimum_speech_ratio,
            "minimum_emphasis_events": minimum_emphasis_events,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _attribution_markdown(records: list[dict]) -> str:
    lines = ["# YouTube training-source attribution", ""]
    for record in records:
        if record.get("status") != "accepted":
            continue
        item = record["item"]
        metadata = record["metadata"]
        lines.extend([
            f"## {metadata.get('title') or item['title']}",
            "",
            f"- Creator/channel: {metadata.get('channel') or item['channel']}",
            f"- Source: {item['url']}",
            f"- License/basis: {metadata.get('license') or item['rights_basis']}",
            f"- Segment: {item['segment_start_seconds']}s–{item['segment_start_seconds'] + item['segment_seconds']}s",
            "",
        ])
    return "\n".join(lines)


def ingest_youtube_manifest(
    manifest_path: Path,
    config: AuditConfig,
    output_root: Path,
    *,
    minimum_visual_coverage: float = 0.60,
    minimum_stage_face_coverage: float = 0.25,
    minimum_speech_ratio: float = 0.30,
    minimum_emphasis_events: int = 3,
    fail_fast: bool = False,
) -> dict:
    if not 0.0 <= minimum_visual_coverage <= 1.0:
        raise ValueError("minimum visual coverage must be between 0 and 1")
    if not 0.0 <= minimum_stage_face_coverage <= 1.0:
        raise ValueError("minimum stage face coverage must be between 0 and 1")
    if not 0.0 <= minimum_speech_ratio <= 1.0:
        raise ValueError("minimum speech ratio must be between 0 and 1")
    if minimum_emphasis_events < 0:
        raise ValueError("minimum emphasis events cannot be negative")
    manifest_path = manifest_path.expanduser().resolve()
    manifest, items = load_youtube_manifest(manifest_path)
    digest = sha256_file(manifest_path)
    batch_directory = output_root.expanduser().resolve() / f"{manifest_path.stem}-{digest[:8]}"
    batch_directory.mkdir(parents=True, exist_ok=True)
    analysis_root = batch_directory / "analysis"
    analysis_root.mkdir(exist_ok=True)
    snapshot_path = batch_directory / "source-manifest.json"
    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state_path = batch_directory / "batch.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "kind": "agape_youtube_ingestion_batch",
            "schema_version": 1,
            "created_at": utc_now(),
            "manifest": str(manifest_path),
            "manifest_sha256": digest,
            "records": {},
        }

    from ..cli import analyze_source

    for item in items:
        existing = state["records"].get(item.key)
        if existing and existing.get("status") in {"accepted", "rejected_quality"}:
            feature_path = Path(existing.get("features", ""))
            if feature_path.is_file():
                existing_features = json.loads(feature_path.read_text(encoding="utf-8"))
                assessment = quality_assessment(
                    existing_features,
                    minimum_visual_coverage=minimum_visual_coverage,
                    minimum_stage_face_coverage=minimum_stage_face_coverage,
                    minimum_speech_ratio=minimum_speech_ratio,
                    minimum_emphasis_events=minimum_emphasis_events,
                    coaching_context=item.context,
                )
                existing["quality"] = assessment
                existing["status"] = "accepted" if assessment["passed"] else "rejected_quality"
                existing["reassessed_at"] = utc_now()
                _write_json(state_path, state)
                continue
        record = {
            "item": asdict(item),
            "started_at": utc_now(),
            "status": "processing",
        }
        state["records"][item.key] = record
        _write_json(state_path, state)
        try:
            metadata = probe_youtube(item.url)
            verify_youtube_metadata(item, metadata)
            record["metadata"] = reduced_metadata(metadata)
            analysis_args = SimpleNamespace(
                segment_seconds=item.segment_seconds,
                segment_start_seconds=item.segment_start_seconds,
                context=item.context,
                retain_media=False,
                keep_failed_job=False,
            )
            batch_config = AuditConfig(
                runs_root=analysis_root,
                model_path=config.model_path,
                max_input_bytes=config.max_input_bytes,
                max_duration_seconds=config.max_duration_seconds,
                proxy_width=config.proxy_width,
                proxy_fps=config.proxy_fps,
                analysis_fps=config.analysis_fps,
                audio_sample_rate=config.audio_sample_rate,
                timeline_window_seconds=config.timeline_window_seconds,
                retain_media=False,
            )
            result = analyze_source(item.url, item.title, analysis_args, batch_config)
            features_path = result["paths"].features
            features = json.loads(features_path.read_text(encoding="utf-8"))
            assessment = quality_assessment(
                features,
                minimum_visual_coverage=minimum_visual_coverage,
                minimum_stage_face_coverage=minimum_stage_face_coverage,
                minimum_speech_ratio=minimum_speech_ratio,
                minimum_emphasis_events=minimum_emphasis_events,
                coaching_context=item.context,
            )
            record.update({
                "status": "accepted" if assessment["passed"] else "rejected_quality",
                "completed_at": utc_now(),
                "job": str(result["paths"].root),
                "features": str(features_path),
                "quality": assessment,
                "media_deleted": result["media_deleted"],
            })
        except Exception as exc:
            record.update({
                "status": "failed",
                "completed_at": utc_now(),
                "error": str(exc),
            })
            _write_json(state_path, state)
            if fail_fast:
                raise
        _write_json(state_path, state)

    records = list(state["records"].values())
    accepted = [record for record in records if record.get("status") == "accepted"]
    group_map = {
        record["features"]: record["item"]["group"] for record in accepted
    }
    group_map_path = batch_directory / "group-map.json"
    group_map_path.write_text(json.dumps(group_map, indent=2), encoding="utf-8")
    attribution_path = batch_directory / "ATTRIBUTION.md"
    attribution_path.write_text(_attribution_markdown(records), encoding="utf-8")
    state["completed_at"] = utc_now()
    state["summary"] = {
        "selected": len(items),
        "accepted": len(accepted),
        "rejected_quality": sum(record.get("status") == "rejected_quality" for record in records),
        "failed": sum(record.get("status") == "failed" for record in records),
        "independent_groups": len(set(group_map.values())),
    }
    state["group_map"] = str(group_map_path)
    state["attribution"] = str(attribution_path)
    _write_json(state_path, state)
    return {
        "batch_directory": str(batch_directory),
        "batch_state": str(state_path),
        "accepted_features": sorted(group_map),
        "group_map": group_map,
        "group_map_path": str(group_map_path),
        "attribution": str(attribution_path),
        "summary": state["summary"],
        "outcomes": [
            {
                "video_id": record["item"]["video_id"],
                "title": record["item"]["title"],
                "status": record.get("status"),
                "error": record.get("error"),
                "failed_quality_checks": sorted(
                    name for name, passed in record.get("quality", {}).get("checks", {}).items()
                    if not passed
                ),
            }
            for record in records
        ],
    }
