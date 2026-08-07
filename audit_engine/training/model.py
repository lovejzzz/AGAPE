from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from . import CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True)
class ModelConfig:
    input_features: int
    hidden_size: int = 48
    blocks: int = 3
    dropout: float = 0.12
    explicit_correlations: bool = True


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class TemporalSynchronyNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(*[
            ResidualTemporalBlock(config.hidden_size, 2 ** index, config.dropout)
            for index in range(config.blocks)
        ])
        self.crossmodal_pairs = ((0, 4), (0, 5), (1, 4), (1, 5))
        correlation_features = len(self.crossmodal_pairs) + 1 if config.explicit_correlations else 0
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size * 2 + correlation_features, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def _crossmodal_correlations(self, values: torch.Tensor) -> torch.Tensor:
        mask_offset = self.config.input_features // 2
        correlations = []
        for first, second in self.crossmodal_pairs:
            valid = (
                (values[:, :, mask_offset + first] > 0.5)
                & (values[:, :, mask_offset + second] > 0.5)
            ).to(values.dtype)
            count = valid.sum(dim=1).clamp_min(1.0)
            left = values[:, :, first]
            right = values[:, :, second]
            left_centered = (left - (left * valid).sum(dim=1, keepdim=True) / count[:, None]) * valid
            right_centered = (right - (right * valid).sum(dim=1, keepdim=True) / count[:, None]) * valid
            numerator = (left_centered * right_centered).sum(dim=1)
            denominator = torch.sqrt(
                (left_centered.square().sum(dim=1) * right_centered.square().sum(dim=1)).clamp_min(1e-8)
            )
            correlation = numerator / denominator
            correlation = torch.where(count >= 8, correlation, torch.zeros_like(correlation))
            correlations.append(correlation)
        return torch.stack(correlations, dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(values).transpose(1, 2)
        encoded = self.temporal(projected)
        pooled_features = [encoded.mean(dim=-1), encoded.amax(dim=-1)]
        if self.config.explicit_correlations:
            correlations = self._crossmodal_correlations(values)
            pooled_features.extend([correlations, correlations.amax(dim=1, keepdim=True)])
        return self.head(torch.cat(pooled_features, dim=1)).squeeze(1)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def checkpoint_payload(
    model: TemporalSynchronyNet,
    *,
    feature_names: list[str],
    normalization_mean: list[float],
    normalization_std: list[float],
    window_steps: int,
    timeline_step_seconds: float,
    threshold: float,
    epoch: int,
    dataset_sha256: str,
    validation_metrics: dict,
    training_config: dict,
) -> dict:
    return {
        "kind": "agape_temporal_synchrony_checkpoint",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "feature_names": feature_names,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "window_steps": window_steps,
        "timeline_step_seconds": timeline_step_seconds,
        "threshold": threshold,
        "epoch": epoch,
        "dataset_sha256": dataset_sha256,
        "validation_metrics": validation_metrics,
        "training_config": training_config,
        "torch_version": str(torch.__version__),
        "target_boundary": "Paired intact-versus-shifted windows with measurable synchrony loss; not a speaking-quality score.",
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[TemporalSynchronyNet, dict]:
    payload = torch.load(path.expanduser().resolve(), map_location=device, weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported AGAPE checkpoint schema")
    model = TemporalSynchronyNet(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
