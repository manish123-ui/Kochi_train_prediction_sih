"""Plot helpers. Every function returns the figure and optionally saves it."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)  # no-op inside notebooks that already picked a backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(history: dict, title: str = "Training history", loss_label: str = "Loss", save_path=None):
    """``history`` needs ``train_loss`` and ``val_loss`` lists."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train", linewidth=2)
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path)


def plot_pred_vs_actual(y_true, y_pred, title: str = "Predicted vs actual", xlabel: str = "Actual",
                        ylabel: str = "Predicted", save_path=None):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=12)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path)


def plot_confusion_matrix(cm, labels=("Standby", "Service"), title: str = "Confusion matrix", save_path=None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return _finish(fig, save_path)


def plot_feature_importance(names, importances, top_n: int = 20, title: str = "Feature importance", save_path=None):
    order = np.argsort(np.asarray(importances))[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(order))))
    ax.barh([names[i] for i in order], [float(importances[i]) for i in order])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _finish(fig, save_path)


def plot_km_by_decision(df, decision_col: str = "FinalDecision", km_col: str = "Daily_Km_Assigned",
                        title: str = "Daily km assigned by decision", save_path=None):
    """EDA helper: checks how Daily_Km_Assigned relates to Service/Standby."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, grp in df.groupby(decision_col):
        ax.hist(grp[km_col].dropna(), bins=30, alpha=0.6, label=str(label))
    ax.set_xlabel(km_col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, save_path)
