"""Shared PyTorch utilities: seeding, loaders, one training loop, artifact bundles.

The notebook copy-pasted a training loop per model. All models now share
:func:`fit_model`, which also fixes two bugs found in the notebook:

* best weights were "saved" with ``state_dict().copy()`` (a shallow copy that keeps
  changing as training continues); we ``deepcopy`` instead;
* early stopping in the simple regressor was commented out, so the final-epoch
  weights were used; here it is a config option that restores the best-validation weights.
"""
from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(X, y, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    """DataLoader over float32 arrays. Drops a trailing batch of 1 (BatchNorm cannot train on it)."""
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    ds = TensorDataset(X_t, y_t)
    drop_last = shuffle and len(ds) > batch_size and len(ds) % batch_size == 1
    generator = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, generator=generator)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "epochs_run": len(self.train_loss),
        }


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    """Sample-weighted mean loss over ``loader`` (model put in eval mode)."""
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    scheduler=None,
    scheduler_mode: str = "none",  # "none" | "plateau" (step on val loss per epoch) | "batch" (step per batch)
    grad_clip: float | None = None,
    patience: int | None = None,
    min_delta: float = 1e-6,
    log_every: int = 10,
    label: str = "model",
) -> TrainingHistory:
    """Train ``model`` in place; restore the best-validation weights at the end.

    Without a validation loader, all epochs run and the final weights are kept.
    """
    if scheduler_mode != "none" and scheduler is None:
        raise ValueError("scheduler_mode is set but no scheduler was given")
    model.to(device)
    history = TrainingHistory()
    best_val = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler_mode == "batch":
                scheduler.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running / max(seen, 1)
        history.train_loss.append(train_loss)
        history.lr.append(float(optimizer.param_groups[0]["lr"]))

        if val_loader is None:
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                logger.info("[%s] epoch %d/%d train_loss=%.5f", label, epoch, epochs, train_loss)
            continue

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history.val_loss.append(val_loss)
        if scheduler_mode == "plateau":
            scheduler.step(val_loss)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            logger.info("[%s] epoch %d/%d train_loss=%.5f val_loss=%.5f lr=%.2e",
                        label, epoch, epochs, train_loss, val_loss, history.lr[-1])
        if patience is not None and stale >= patience:
            history.stopped_early = True
            logger.info("[%s] early stopping at epoch %d (best epoch %s)", label, epoch, history.best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_array(model: nn.Module, X, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """Raw model outputs for a 2-D array (logits for the classifier, km for the regressor)."""
    model.to(device)
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    outputs = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size]).to(device)
        outputs.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Artifact bundles: model.pt (weights only) + preprocessor.joblib + metadata.json
# --------------------------------------------------------------------------- #
def save_bundle(directory: str | Path, *, model: nn.Module, preprocessor, architecture: str,
                architecture_kwargs: dict, metadata: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "architecture": architecture,
            "architecture_kwargs": architecture_kwargs,
        },
        directory / "model.pt",
    )
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved bundle to %s", directory)
    return directory


def load_bundle(directory: str | Path, build_model: Callable[[str, dict], nn.Module]):
    """Return ``(model, preprocessor, metadata)``.

    ``preprocessor.joblib`` is a pickle: only load bundles you created yourself.
    """
    directory = Path(directory)
    if not (directory / "model.pt").exists():
        raise FileNotFoundError(f"No trained bundle in {directory}. Train first (see README).")
    ckpt = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model = build_model(ckpt["architecture"], ckpt["architecture_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    meta_path = directory / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, preprocessor, metadata


def run_environment() -> dict:
    """Library versions + UTC timestamp, stored in every bundle's metadata for reproducibility."""
    import datetime
    import platform

    import pandas as pd
    import sklearn

    return {
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
