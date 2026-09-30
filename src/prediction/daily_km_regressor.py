"""Daily_Km_Assigned regressor.

Two architectures from the notebook are kept, selectable by name:

``simple``         cell 9  - MLP 64-32-16 with ReLU + Dropout(0.2). Best recorded result.
``ultra_complex``  cell 12 - three parallel paths (deep / wide / interaction) merged by a
                   combiner, BatchNorm + GELU + Dropout, ~1.05 M parameters.

Dropped on purpose: the notebook's ``ComplexRegressionModel`` was defined but never used,
and its skip connections add a 256-d tensor to a 128-d one, so it could not run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config as cfg
from src.data.preprocessing import REGRESSOR_TASK, build_task_frame
from src.prediction.common import get_device, load_bundle, predict_array, save_bundle


def _block(in_dim: int, out_dim: int, dropout: float | None) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    return layers


class SimpleRegressionMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class UltraComplexRegressionMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.input_bn = nn.BatchNorm1d(input_size)
        self.main_path = nn.Sequential(
            *_block(input_size, 1024, 0.4), *_block(1024, 512, 0.4), *_block(512, 256, 0.3),
            *_block(256, 128, 0.3), *_block(128, 64, 0.2),
        )
        self.wide_path = nn.Sequential(*_block(input_size, 512, 0.3), *_block(512, 256, 0.2))
        self.interaction_path = nn.Sequential(*_block(input_size, 256, 0.2), *_block(256, 128, 0.2))
        self.combiner = nn.Sequential(
            *_block(64 + 256 + 128, 256, 0.3), *_block(256, 128, 0.2), *_block(128, 64, 0.1),
            *_block(64, 32, None),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        combined = torch.cat([self.main_path(x), self.wide_path(x), self.interaction_path(x)], dim=1)
        return self.combiner(combined).squeeze(1)


REGRESSOR_ARCHITECTURES = {"simple": SimpleRegressionMLP, "ultra_complex": UltraComplexRegressionMLP}


def build_regressor(architecture: str, kwargs: dict) -> nn.Module:
    if architecture not in REGRESSOR_ARCHITECTURES:
        raise ValueError(f"Unknown regressor architecture {architecture!r}; choose from {sorted(REGRESSOR_ARCHITECTURES)}")
    return REGRESSOR_ARCHITECTURES[architecture](**kwargs)


class DailyKmRegressor:
    """A fitted regressor: network + fitted preprocessor. ``predict`` takes a RAW dataframe."""

    def __init__(self, model: nn.Module, preprocessor, architecture: str,
                 metadata: dict | None = None, device: str = "auto"):
        self.model = model
        self.preprocessor = preprocessor
        self.architecture = architecture
        self.metadata = metadata or {}
        self.device = get_device(device)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Raw predicted daily km (unclipped). Clipping/gating happen in the inference pipeline."""
        X, _ = build_task_frame(df, REGRESSOR_TASK, require_target=False)
        Z = self.preprocessor.transform(X)
        return predict_array(self.model, Z, self.device)

    def save(self, directory: str | Path = cfg.REGRESSOR_DIR) -> Path:
        return save_bundle(directory, model=self.model, preprocessor=self.preprocessor,
                           architecture=self.architecture,
                           architecture_kwargs={"input_size": int(self.model.input_size)},
                           metadata=dict(self.metadata, architecture=self.architecture))

    @classmethod
    def load(cls, directory: str | Path = cfg.REGRESSOR_DIR, device: str = "auto") -> "DailyKmRegressor":
        model, preprocessor, meta = load_bundle(directory, build_regressor)
        return cls(model, preprocessor, architecture=meta.get("architecture", "simple"), metadata=meta, device=device)
