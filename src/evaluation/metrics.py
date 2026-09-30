"""Evaluation metrics for the classifier, the regressor and (experiment) score models."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Binary metrics for P(Service). ``majority_baseline_accuracy`` gives context for accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_service": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_service": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_service": float(f1_score(y_true, y_pred, zero_division=0)),
        "majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),  # rows=true, cols=pred (Standby, Service)
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    return out


def regression_metrics(y_true, y_pred) -> dict:
    """MSE / RMSE / MAE / R2 (the four numbers the notebook reported)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def format_metrics(metrics: dict, prefix: str = "") -> str:
    parts = [f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)]
    return f"{prefix}{' | '.join(parts)}"


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
