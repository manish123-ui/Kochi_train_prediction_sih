"""FinalDecision classifier: Service (1) vs Standby (0).

Architecture is the notebook's (cell 8) MLP: 10-100-100-100 with LeakyReLU(0.2)/Tanh.
CHANGED: the network now outputs a *logit* and is trained with BCEWithLogitsLoss
instead of ending in ``Sigmoid`` + ``BCELoss``. It is the same function, but
numerically stable. ``predict_proba`` applies the sigmoid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch import nn

from src import config as cfg
from src.data.preprocessing import CLASSIFIER_TASK, build_task_frame
from src.prediction.common import get_device, load_bundle, predict_array, save_bundle


class InductionMLP(nn.Module):
    """Binary classifier head returning one logit per row."""

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # squeeze(1), not squeeze(): keeps shape (B,) when B == 1


CLASSIFIER_ARCHITECTURES = {"InductionMLP": InductionMLP}


def build_classifier(architecture: str, kwargs: dict) -> nn.Module:
    if architecture not in CLASSIFIER_ARCHITECTURES:
        raise ValueError(f"Unknown classifier architecture {architecture!r}")
    return CLASSIFIER_ARCHITECTURES[architecture](**kwargs)


class InductionClassifier:
    """A fitted classifier: network + fitted preprocessor + decision threshold.

    ``predict_*`` take a RAW dataframe (the CSV columns) and reproduce the training-time
    encoding and scaling, so training and inference cannot drift apart.
    """

    def __init__(self, model: nn.Module, preprocessor, threshold: float = 0.5,
                 metadata: dict | None = None, device: str = "auto"):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.metadata = metadata or {}
        self.device = get_device(device)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(Service) for each row of ``df``."""
        X, _ = build_task_frame(df, CLASSIFIER_TASK, require_target=False)
        Z = self.preprocessor.transform(X)
        return expit(predict_array(self.model, Z, self.device))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """1 = Service, 0 = Standby."""
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def predict_labels(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(self.predict(df) == 1, cfg.SERVICE_LABEL, cfg.STANDBY_LABEL)

    def save(self, directory: str | Path = cfg.CLASSIFIER_DIR) -> Path:
        input_size = int(self.model.input_size)
        meta = dict(self.metadata, decision_threshold=self.threshold)
        return save_bundle(directory, model=self.model, preprocessor=self.preprocessor,
                           architecture="InductionMLP", architecture_kwargs={"input_size": input_size},
                           metadata=meta)

    @classmethod
    def load(cls, directory: str | Path = cfg.CLASSIFIER_DIR, device: str = "auto") -> "InductionClassifier":
        model, preprocessor, meta = load_bundle(directory, build_classifier)
        return cls(model, preprocessor, threshold=float(meta.get("decision_threshold", 0.5)),
                   metadata=meta, device=device)
