# AGAPE Audit Engine

AGAPE—the **Audiovisual Gesture And Prosody Engine**—is an offline-first engine developed by Second Take Lab. It analyzes vocal delivery and visible movement on one shared media clock. It deliberately does **not** transcribe speech and does not infer personality, emotion, honesty, diagnosis, or protected traits.

> Research status: AGAPE v0.2 is a synchronized feature, comparison, and self-supervised calibration engine. The repository includes a local temporal synchrony model and a promotion-gated training loop. One local checkpoint has cleared the current one-time held-out synchrony gates, but checkpoints and training data are not distributed with the repository, and the default coaching engine remains the interpretable ruleset.

## Set up another computer

Requirements: Python 3.11–3.13, `ffmpeg`, `ffprobe`, and `curl`.

```bash
git clone https://github.com/lovejzzz/AGAPE.git
cd AGAPE
./scripts/bootstrap.sh
source .venv/bin/activate
pytest -q
agape --help
```

The bootstrap script creates an isolated Python environment, installs AGAPE and its test dependencies, and downloads the official MediaPipe Holistic Landmarker bundle into `models/`. The model binary is deliberately not committed to Git.

## What it measures

- audio energy, pitch variation, voicing, vocal pulses, and pauses;
- approximate camera-facing head direction, posture, hand visibility, and movement;
- cross-modal timing: vocal emphasis with gesture, pauses with stillness, and movement that competes with delivery;
- time-coded coaching evidence, confidence, and tracker coverage.

AGAPE v0.2 resolves the shared timeline at 250 ms, detects vocal-emphasis events and gesture preparation/stroke/recovery phases, and measures the nearest voice–gesture lag against the speaker's own within-take baseline.

## Paired-take coaching

The primary workflow compares the same speaker delivering substantially the same material twice. AGAPE reports observable changes and labels apparent improvements as **candidate gains** until a human confirms them.

```bash
.venv/bin/python -m audit_engine compare /path/take-1.mp4 /path/take-2.mp4 \
  --title "Pitch practice" --context camera
```

After reviewing a finding, record calibration locally:

```bash
.venv/bin/python -m audit_engine feedback /path/to/runs/<comparison-job> \
  --finding 1 --judgment helpful --notes "Matched what I perceived"
```

## Self-supervised AGAPE Lab

AGAPE Lab creates known visual delays, checks repeatability and directional sensitivity, and deletes every temporary variant after the calibration report is saved.

```bash
.venv/bin/agape lab /path/to/video.mp4 --segment-seconds 30 \
  --delays 0.25,0.50,0.75 --context camera
```

## Local neural training (no AI API or token usage)

The learned model trains entirely on this computer with PyTorch. It uses only AGAPE's numeric, transcript-free timelines. Known time shifts provide self-supervised labels; no LLM labels, API keys, cloud training, or raw media are required during model fitting.

```bash
./scripts/bootstrap-training.sh
.venv/bin/agape train-demo
```

The demo proves the MPS/CPU training machinery but is deliberately rejected for promotion because its features are generated. For a real run, first analyze at least eight independent recordings, then point the one-command pipeline at their completed run directories:

```bash
.venv/bin/agape train-local /absolute/path/to/curated-runs --epochs 60
```

Use `--group-map /path/groups.json` when multiple recordings contain the same speaker or session. For a preregistered final evaluation, also use `--split-plan /path/splits.json` to assign every group label explicitly to `train`, `val`, or `test`. See [`training/README.md`](training/README.md) for the dataset contract, supervision boundary, commands, artifacts, and promotion gates.

### YouTube training sources

AGAPE can build the same local dataset from reviewed YouTube segments. It supports Creative Commons discovery through the YouTube Data API or a no-key metadata-only fallback, resumable ingestion, current-license verification, quality filtering, attribution records, automatic media deletion, local MPS training, and held-out judgment.

```bash
.venv/bin/agape youtube-discover "public speaking keynote" \
  --output training_data/youtube-manifest.json

# Review the manifest and confirm its reuse attestation first.
.venv/bin/agape youtube-train training_data/youtube-manifest.json --epochs 60
```

See [`training/YOUTUBE.md`](training/YOUTUBE.md) for the manifest and reuse boundary.

## Storage model

Each run gets an isolated job directory. The engine makes a 960-pixel, 8-fps proxy plus mono 16 kHz audio, analyzes those derivatives, and writes JSON, a chart, Markdown, HTML, and a manifest. On success, raw video, proxy video, and extracted audio are deleted by default. Only derived timeline features and reports remain. Use `--retain-media` only for debugging with consent.

Default limits are 20 minutes and 750 MB. YouTube tests are capped at 720p and may be clipped with `--segment-seconds`; use `--segment-start-seconds` to skip an intro. The manifest records both temporary peak media bytes and the post-cleanup files.

## Run

```bash
.venv/bin/python -m audit_engine analyze /absolute/path/video.mp4 --title "Practice pitch"
.venv/bin/python -m audit_engine analyze "https://www.youtube.com/watch?v=..." --segment-seconds 180
.venv/bin/python -m audit_engine analyze "https://www.youtube.com/watch?v=..." --segment-start-seconds 600 --segment-seconds 90 --context camera
```

Outputs are written under `runs/<job-id>/`.

## Interpretation boundary

Camera-facing is a head-direction proxy, not verified eye contact. Vocal-pulse rate is not words per minute. Findings are observable hypotheses tied to timestamps. They are prompts for reflection and a second take, not authoritative judgments.
