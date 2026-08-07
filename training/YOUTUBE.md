# YouTube-to-AGAPE local training pipeline

This pipeline can discover Creative Commons candidates, download short approved segments, extract transcript-free AGAPE features, delete all downloaded media, filter unusable samples, create source-isolated splits, train on local MPS/CPU, and judge the checkpoint.

YouTube discovery uses ordinary YouTube Data API quota. Feature extraction and neural training use no AI API and consume no AI tokens.

## Safety and reuse boundary

A public URL is not sufficient authorization. Before ingestion, the manifest requires a top-level `reuse_attestation.confirmed: true`. Review:

- the current license or documented permission;
- whether the intended research/training use is appropriate;
- speaker privacy and consent expectations;
- required CC BY attribution;
- whether the speaker/session grouping prevents data leakage.

The automatic discovery route filters for YouTube's `creativeCommon` license, safe search, embeddable videos, and excludes child-directed, age-restricted, live, and too-short candidates. Ingestion checks current `yt-dlp` metadata again. The pipeline also supports `owned` and `permission` rights bases, but those require a non-empty `permission_reference` in each item.

YouTube explains that a Creative Commons Attribution license permits reuse subject to CC BY terms and attribution. It does not make unrelated privacy or research-use decisions for you. This pipeline records provenance and enforces review; it does not provide legal advice.

Official references: [YouTube license types](https://support.google.com/youtube/answer/2797468), [YouTube Data API search filters](https://developers.google.com/youtube/v3/docs/search/list), and [video license metadata](https://developers.google.com/youtube/v3/docs/videos).

## Route A: discover Creative Commons candidates

The default `auto` backend uses the YouTube Data API when `AGAPE_YOUTUBE_API_KEY` is available in the environment or, on macOS, in a Keychain item with that service name. The key is never written to project files. Without a key, discovery falls back to metadata-only `yt-dlp` search and verifies each result's current Creative Commons license without downloading video. The no-key backend cannot verify made-for-kids status, so its generated manifest calls that limitation out for manual review.

```bash
.venv/bin/agape youtube-discover "public speaking keynote presentation" \
  --max-results 40 \
  --max-per-channel 2 \
  --segment-seconds 90 \
  --relevance-language en \
  --output training_data/youtube-manifest.json
```

Use `--backend api` to require the official API or `--backend yt-dlp` to force no-key discovery.

Discovery groups videos conservatively by channel. Review every item. Change `group` when you know which clips share a speaker or recording session; clips that could leak the same person/session across evaluation splits must share one group. Set unwanted items to `"selected": false`.

Finally, fill in the attestation and set `reuse_attestation.confirmed` to `true`.

## Route B: provide known URLs

Copy [`youtube-manifest.example.json`](youtube-manifest.example.json), replace the placeholder items, and use one of these rights bases:

- `creative_commons`: current metadata must still report Creative Commons;
- `owned`: you control the recording; include a `permission_reference`;
- `permission`: you have documented authorization; include a `permission_reference`.

Each selected segment must be 30–180 seconds. Prefer stable shots where one adult speaker's face and upper body remain visible.

## Inspect ingestion before training

```bash
.venv/bin/agape youtube-ingest training_data/youtube-manifest.json
```

The resumable batch is written beneath `runs/youtube-training/`. Each item receives one of three outcomes:

- `accepted`: meets rights/metadata and audiovisual quality gates;
- `rejected_quality`: media was processed and deleted, but face, pose, speech, duration, or emphasis coverage was insufficient;
- `failed`: metadata verification, download, or analysis failed.

The batch contains `batch.json`, a manifest snapshot, `group-map.json`, `ATTRIBUTION.md`, derived analysis jobs, and no downloaded source/proxy/audio after successful analysis.

Default acceptance gates are 60% pose coverage, 30% detected speech ratio, at least three emphasis events, and at least 30 analyzed seconds. Camera recordings require 60% face coverage. Stage recordings use a 25% face floor because pose/gesture timing remains usable and the learned feature contract carries explicit missing-value masks. Thresholds can be raised on the command line.

Re-running the same manifest resumes accepted entries rather than downloading them again.

## Ingest, train, and judge locally

```bash
.venv/bin/agape youtube-train training_data/youtube-manifest.json \
  --epochs 60 \
  --device auto
```

This performs the full path:

1. Verify metadata and ingest selected segments.
2. Discard low-quality segments and delete media.
3. Build balanced same-window intact/shifted pairs, retaining a pair only when the shift measurably weakens cross-modal evidence.
4. Split by the reviewed speaker/session groups before extracting windows.
5. Train the temporal synchrony model on local MPS/CPU.
6. Create an exclusive holdout-consumption record, judge the untouched test groups once, and write `judge.json` and `judge.md`.

Use `--require-pass` when this runs in a larger local script and a rejected checkpoint should return a nonzero status.

## What remains local

- Downloaded YouTube media exists only temporarily during feature extraction.
- Retained training examples are numeric AGAPE timelines without transcripts.
- Dataset files, checkpoints, batch state, source URLs, attribution, and reports remain local and are ignored by Git.
- The YouTube API key is never persisted.
- The neural optimizer and deterministic judge make no network calls.
