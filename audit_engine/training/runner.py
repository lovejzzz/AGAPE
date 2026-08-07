from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import FEATURE_NAMES, load_dataset, sha256_file, validate_split_integrity
from .metrics import (
    best_group_macro_threshold,
    binary_metrics,
    group_macro_metrics,
    sigmoid,
)
from .model import (
    ModelConfig,
    TemporalSynchronyNet,
    checkpoint_payload,
    load_checkpoint,
    select_device,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def group_balanced_pair_weights(group_ids: np.ndarray) -> np.ndarray:
    pair_groups = np.asarray(group_ids)[::2]
    groups, counts = np.unique(pair_groups, return_counts=True)
    inverse = {group: 1.0 / count for group, count in zip(groups, counts, strict=True)}
    return np.asarray([inverse[group] for group in pair_groups], dtype=np.float64)


def _paired_loader(
    x: np.ndarray,
    y: np.ndarray,
    clip_ids: np.ndarray,
    group_ids: np.ndarray,
    starts: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    group_balanced: bool,
) -> DataLoader:
    if len(x) % 2 or not np.all(y[::2] == 1) or not np.all(y[1::2] == 0):
        raise ValueError("dataset does not contain ordered positive/negative pairs")
    if not np.array_equal(clip_ids[::2], clip_ids[1::2]) or not np.array_equal(
        group_ids[::2], group_ids[1::2]
    ) or not np.allclose(starts[::2], starts[1::2]):
        raise ValueError("paired examples must share a group, clip, and start time")
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x[::2]), torch.from_numpy(x[1::2]))
    sampler = None
    if group_balanced:
        sampler = WeightedRandomSampler(
            torch.from_numpy(group_balanced_pair_weights(group_ids)),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=not group_balanced,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = _loader(x, np.zeros(len(x)), batch_size=batch_size, shuffle=False, seed=0)
    logits: list[np.ndarray] = []
    model.eval()
    for values, _ in loader:
        result = model(values.to(device)).detach().cpu().numpy()
        logits.append(result)
    return sigmoid(np.concatenate(logits))


def normalize_from_train(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    train = dataset["x"][dataset["split"] == "train"]
    flattened = train.reshape(-1, train.shape[-1]).astype(np.float64)
    mean = flattened.mean(axis=0).astype(np.float32)
    std = flattened.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    mask_offset = len(FEATURE_NAMES) // 2
    mean[mask_offset:] = 0.0
    std[mask_offset:] = 1.0
    return mean, std


def normalized(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def validation_selection_score(metrics: dict) -> tuple[float, bool, float]:
    separation = float(metrics["mean_positive_score"] - metrics["mean_negative_score"])
    group_macro = metrics.get("group_macro", {})
    group_macro_balanced = float(
        group_macro.get("balanced_accuracy", metrics["balanced_accuracy"])
    )
    group_macro_auroc = float(group_macro.get("auroc", metrics["auroc"]))
    worst_group_balanced = float(
        group_macro.get("worst_group_balanced_accuracy", metrics["balanced_accuracy"])
    )
    ready = bool(
        metrics["balanced_accuracy"] >= 0.70
        and group_macro_balanced >= 0.70
        and metrics["auroc"] >= 0.75
        and separation >= 0.12
    )
    # A gate-ready epoch always outranks an unready one; AUROC breaks ties.
    score = (
        3.0 * int(ready)
        + group_macro_balanced
        + 0.1 * group_macro_auroc
        + 0.01 * worst_group_balanced
    )
    return (score, ready, separation)


def train_model(
    dataset_path: Path,
    output_root: Path,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 48,
    blocks: int = 3,
    dropout: float = 0.12,
    patience: int = 10,
    pairwise_weight: float = 0.5,
    pairwise_margin: float = 1.0,
    group_balanced: bool = True,
    explicit_correlations: bool = True,
    seed: int = 17,
    requested_device: str = "auto",
    resume_from: str | Path | None = None,
) -> dict:
    if epochs < 1 or batch_size < 2 or patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    if pairwise_weight < 0 or pairwise_margin <= 0:
        raise ValueError("pairwise weight must be non-negative and margin must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    dataset_path = dataset_path.expanduser().resolve()
    dataset = load_dataset(dataset_path)
    integrity = validate_split_integrity(dataset)
    if not integrity["passed"]:
        raise ValueError(f"dataset split integrity failed: {integrity}")
    device = select_device(requested_device)
    mean, std = normalize_from_train(dataset)
    x = normalized(dataset["x"], mean, std)
    y = dataset["y"]
    split = dataset["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    train_loader = _paired_loader(
        x[train_mask],
        y[train_mask],
        dataset["clip_id"][train_mask],
        dataset["group_id"][train_mask],
        dataset["start_seconds"][train_mask],
        batch_size=batch_size,
        seed=seed,
        group_balanced=group_balanced,
    )

    model_config = ModelConfig(
        input_features=x.shape[-1],
        hidden_size=hidden_size,
        blocks=blocks,
        dropout=dropout,
        explicit_correlations=explicit_correlations,
    )
    model = TemporalSynchronyNet(model_config).to(device)
    if resume_from is not None:
        previous_model, previous = load_checkpoint(Path(resume_from), device)
        previous_config = previous.get("model_config", {})
        if previous_config != asdict(model_config):
            raise ValueError(
                f"resume checkpoint model config {previous_config} does not match "
                f"new dataset config {asdict(model_config)}"
            )
        model.load_state_dict(previous_model.state_dict())
        model.train()
    positives = int(np.sum(y[train_mask] == 1))
    negatives = int(np.sum(y[train_mask] == 0))
    pos_weight = torch.tensor([negatives / max(positives, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = output_root.expanduser().resolve() / f"{stamp}-{sha256_file(dataset_path)[:8]}"
    suffix = 1
    while run_directory.exists():
        run_directory = output_root.expanduser().resolve() / f"{stamp}-{sha256_file(dataset_path)[:8]}-{suffix}"
        suffix += 1
    run_directory.mkdir(parents=True)
    best_path = run_directory / "best.pt"
    last_path = run_directory / "last.pt"
    history_path = run_directory / "history.json"
    summary_path = run_directory / "training-summary.json"
    config = {
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_size": hidden_size,
        "blocks": blocks,
        "dropout": dropout,
        "patience": patience,
        "pairwise_weight": pairwise_weight,
        "pairwise_margin": pairwise_margin,
        "group_balanced": group_balanced,
        "explicit_correlations": explicit_correlations,
        "threshold_calibration": "maximize macro balanced accuracy across validation groups",
        "seed": seed,
        "device": str(device),
        "resume_from": str(resume_from) if resume_from else None,
        "checkpoint_selection": (
            "Prefer epochs that clear validation balanced-accuracy, AUROC, and score-separation gates; "
            "then maximize validation AUROC."
        ),
    }
    (run_directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_score = -math.inf
    stale_epochs = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        pairwise_losses: list[float] = []
        for positive_values, negative_values in train_loader:
            positive_values = positive_values.to(device)
            negative_values = negative_values.to(device)
            values = torch.cat([positive_values, negative_values], dim=0)
            labels = torch.cat([
                torch.ones(len(positive_values), device=device),
                torch.zeros(len(negative_values), device=device),
            ])
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            pair_count = len(positive_values)
            pairwise_loss = F.softplus(
                pairwise_margin - (logits[:pair_count] - logits[pair_count:])
            ).mean()
            loss = criterion(logits, labels) + pairwise_weight * pairwise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pairwise_losses.append(float(pairwise_loss.detach().cpu()))

        val_probabilities = predict_probabilities(
            model, x[val_mask], device=device, batch_size=batch_size
        )
        threshold = best_group_macro_threshold(
            y[val_mask], val_probabilities, dataset["group_id"][val_mask]
        )
        val_metrics = binary_metrics(y[val_mask], val_probabilities, threshold)
        val_metrics["group_macro"] = group_macro_metrics(
            y[val_mask], val_probabilities, dataset["group_id"][val_mask], threshold
        )
        selection_score, selection_ready, validation_margin = validation_selection_score(val_metrics)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_pairwise_loss": float(np.mean(pairwise_losses)),
            "validation_selection_ready": selection_ready,
            "validation_score_margin": validation_margin,
            "validation": val_metrics,
        }
        history.append(record)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(record["train_loss"], 5),
            "train_pairwise_loss": round(record["train_pairwise_loss"], 5),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 4),
            "val_auroc": round(val_metrics["auroc"], 4),
            "val_score_margin": round(validation_margin, 4),
            "val_selection_ready": selection_ready,
        }))

        payload = checkpoint_payload(
            model,
            feature_names=list(FEATURE_NAMES),
            normalization_mean=mean.tolist(),
            normalization_std=std.tolist(),
            window_steps=int(x.shape[1]),
            timeline_step_seconds=float(np.asarray(dataset["timeline_step_seconds"]).item()),
            threshold=threshold,
            epoch=epoch,
            dataset_sha256=config["dataset_sha256"],
            validation_metrics=val_metrics,
            training_config=config,
        )
        torch.save(payload, last_path)
        if selection_score > best_score + 1e-5:
            best_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(payload, best_path)
        else:
            stale_epochs += 1
        history_path.write_text(json.dumps(_json_safe(history), indent=2), encoding="utf-8")
        if stale_epochs >= patience:
            break

    _, best = load_checkpoint(best_path, device)
    summary = {
        "kind": "agape_training_summary",
        "status": "complete",
        "run_directory": str(run_directory),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "device": str(device),
        "dataset_sha256": config["dataset_sha256"],
        "split_integrity": integrity,
        "validation_metrics": best["validation_metrics"],
        "note": "The held-out test split was not used during training or checkpoint selection.",
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    split_name: str,
    *,
    requested_device: str = "auto",
    batch_size: int = 128,
) -> tuple[dict, np.ndarray, np.ndarray, dict, dict[str, np.ndarray]]:
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    device = select_device(requested_device)
    dataset = load_dataset(dataset_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if list(dataset["feature_names"]) != checkpoint["feature_names"]:
        raise ValueError("checkpoint and dataset feature contracts differ")
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    mask = dataset["split"] == split_name
    values = normalized(dataset["x"][mask], mean, std)
    probabilities = predict_probabilities(
        model, values, device=device, batch_size=batch_size
    )
    labels = dataset["y"][mask]
    metrics = binary_metrics(labels, probabilities, float(checkpoint["threshold"]))
    selected = {key: value[mask] for key, value in dataset.items() if getattr(value, "ndim", 0) > 0 and len(value) == len(mask)}
    return metrics, probabilities, labels, checkpoint, selected
