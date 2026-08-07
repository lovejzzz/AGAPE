# AGAPE local training space

This workspace trains a compact temporal convolutional network to distinguish an intact AGAPE audiovisual feature timeline from a copy whose visual features were shifted in time. Every expensive loop is local. It does not call an AI service, consume model tokens, transcribe speech, upload recordings, or need an API key.

## What is—and is not—being learned

The target is narrow: **intact shared timeline (1) versus the same source and time window with its visual stream shifted (0)**. A pair is admitted only when the shift weakens masked cross-modal correlation evidence by at least `0.10` by default. This prevents arbitrary shifts that preserve or improve alignment from becoming mislabeled negatives. The checkpoint can learn synchrony cues between energy, pitch, vocal pulses, body motion, gesture motion, camera-facing proxy, posture, and hand visibility.

It is not trained to infer communication quality, personality, emotion, honesty, diagnosis, intent, identity, or protected traits. A synchrony probability must not be presented as any of those things.

## 1. Install the local training environment

```bash
./scripts/bootstrap-training.sh
```

The bootstrap chooses Python 3.11–3.13, installs the `train` dependency group, and reports whether Apple MPS is available. On this machine, `--device auto` should select the M2 Max GPU. CPU remains a supported fallback.

## 2. Prove the machinery

```bash
.venv/bin/agape train-demo --epochs 18
```

This generates disposable numeric timelines, prepares the split, trains, and runs the judge. The expected verdict is `DO_NOT_PROMOTE`: generated data validates plumbing and compute only. Other metric gates should still be useful for catching implementation failures.

## 3. Curate real local timelines

Run normal AGAPE analysis on consented recordings. Raw video, proxy video, and audio are deleted after successful analysis by default; `features.json` is sufficient for training.

```bash
.venv/bin/agape analyze /absolute/path/recording-01.mp4 --title "Training sample 01"
```

Minimum promotion set:

- at least eight independent source groups;
- at least four genuinely new groups reserved for the untouched test split;
- at least 80 held-out windows, with 30 per class;
- the same timeline resolution for every input (normally 250 ms);
- consent and a documented right to use every recording.

More diversity is better. Vary speakers, recording conditions, delivery styles, framing, microphones, and naturally occurring synchrony. Do not populate the set only with one person or one session.

If multiple feature files are from the same speaker or recording session, put them in the same group so they cannot cross splits. Create a JSON map whose keys are feature-file paths and whose values are private group labels:

```json
{
  "/absolute/runs/take-a/features.json": "speaker-01-session-01",
  "/absolute/runs/take-b/features.json": "speaker-01-session-01",
  "/absolute/runs/take-c/features.json": "speaker-02-session-01"
}
```

Labels are hashed in the dataset artifact. To freeze assignments rather than use the seeded group shuffle, provide `--split-plan splits.json`, where every group label maps to `train`, `val`, or `test`. The plan must cover every group exactly once and include all three splits; a `"*"` entry may assign a recorded default split to every group not named explicitly. Repeat `--group-map` to merge reviewed batches, and use `--group-map-only` to exclude feature files that were not accepted into those maps.

## 4. Train and judge in one command

```bash
.venv/bin/agape train-local /absolute/path/to/curated-runs \
  --group-map /absolute/path/groups.json \
  --epochs 60 \
  --device auto
```

This runs three deterministic stages:

1. `training-data` assigns source groups to train/validation/test before windowing, then creates balanced, same-window intact/shifted pairs whose evidence loss clears `--minimum-alignment-gain`.
2. `train` normalizes from the training split only, samples matched pairs with inverse source-group frequency by default, and combines binary classification loss with a pairwise ranking loss. It calibrates the threshold for macro balanced accuracy across validation groups, records per-group and worst-group metrics, prefers gate-ready epochs by group-macro performance, applies early stopping, and never reads test labels. Use `--no-group-balancing` or `--no-explicit-correlations` only for recorded development ablations.
3. `judge` creates an exclusive dataset-adjacent holdout-consumption record before opening the test split, then applies explicit promotion gates. Any later attempt to judge the same dataset path fails, even with another checkpoint or after an interrupted evaluation.

The stages can also be run separately:

```bash
.venv/bin/agape training-data /absolute/path/to/curated-runs \
  --group-map /absolute/path/groups.json \
  --output training_data/synchrony-v2.npz

.venv/bin/agape train training_data/synchrony-v2.npz --epochs 60

.venv/bin/agape judge \
  training_runs/<run>/best.pt \
  training_data/synchrony-v2.npz \
  --require-pass
```

## Artifacts

Generated data and checkpoints are ignored by Git:

- `training_data/*.npz`: numeric windows, labels, group hashes, split, synthetic offset, and timestamps;
- `training_data/*.manifest.json`: input hashes, label contract, split strategy and counts, and dataset hash;
- `training_data/*.holdout-consumption.json`: exclusive record of the only final-holdout judging attempt for that dataset path;
- `training_runs/<run>/config.json`: exact hyperparameters and device;
- `training_runs/<run>/history.json`: per-epoch training loss and validation metrics;
- `training_runs/<run>/best.pt` and `last.pt`: local checkpoints;
- `training_runs/<run>/training-summary.json`: selected epoch and validation result;
- `training_runs/<run>/judge.json` and `judge.md`: held-out metrics, every gate, and the verdict.

The dataset manifest reports per-split delay counts and warns before training when the future test split cannot meet the judge's five-example-per-delay requirement. The judge verifies dataset provenance, split isolation, real-data status, at least four held-out source groups, test sample count, balanced accuracy, AUROC, intact/shifted margin, validation-to-test gap, held-out coverage for every configured shift, and response across those time shifts.

## Scoring after promotion

Only a checkpoint whose neighboring `judge.json` says `PROMOTE` is accepted by default:

```bash
.venv/bin/agape learned-score \
  training_runs/<run>/best.pt \
  runs/<analysis-job>/features.json
```

`--allow-experimental` exists for research inspection of rejected checkpoints. Experimental scores must not be wired into coaching reports.

## Reproducibility and supervision

- Dataset splitting, negative sampling, weight initialization, and batch order use the recorded seed.
- Input and dataset SHA-256 hashes make accidental dataset drift visible.
- Training uses no test split for optimization or checkpoint selection.
- The local judge is deterministic and costs no AI tokens.
- Human review still owns the final scientific decision. A promotion verdict only clears the current synchrony gates; it does not establish general validity or permission for high-stakes use.
