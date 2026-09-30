"""Dataset loading, schema validation and the synthetic sample generator.

The real Kochi Metro induction dataset is private and is not part of this
repository. To make the code runnable anyway, :func:`generate_synthetic_dataset`
builds a table with the same columns as the notebook's dataset.

WARNING: the synthetic data is for smoke-testing only. Its labels follow an
arbitrary made-up rule so that models have something to fit. They are NOT Kochi
Metro operational rules, and metrics computed on it mean nothing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, SAMPLE_DATA_PATH, SEED

logger = logging.getLogger(__name__)

# The 19 columns the original notebook printed for kochi_induction_dataset.csv.
RAW_COLUMNS = [
    "Date", "Trainset", "Depot", "FinalDecision", "DecisionReason",
    "Rolling_Valid", "Signalling_Valid", "Telecom_Valid", "Any_Fitness_Invalid",
    "Open_JobCards", "Critical_JobCard", "Branding_Hours_Left", "Cum_Mileage",
    "Mileage_Label", "Days_Since_Last_Deep_Clean", "Cleaning_Slot_Scheduled",
    "Bay_Position", "Stabling_Difficulty", "Daily_Km_Assigned",
]


def validate_columns(df: pd.DataFrame, required: Iterable[str], context: str = "") -> None:
    """Raise a readable error if ``df`` lacks any of the ``required`` columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        where = f" for {context}" if context else ""
        raise ValueError(f"Missing required column(s){where}: {missing}. Found: {list(df.columns)}")


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Read the induction CSV with pandas defaults (as the notebook did)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Place your private CSV at {RAW_DATA_PATH} or pass --data <path>, "
            f"or use --use-sample for the synthetic smoke-test data."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s with shape %s", path, df.shape)
    return df


def resolve_data_path(path: str | Path | None = None, use_sample: bool = False) -> tuple[Path, str]:
    """Return ``(csv_path, source)`` where source is ``"real"`` or ``"synthetic"``."""
    if path is not None:
        return Path(path), "real"
    if use_sample:
        if not SAMPLE_DATA_PATH.exists():
            write_synthetic_sample(SAMPLE_DATA_PATH)
        logger.warning("Using SYNTHETIC sample data - results are meaningless for real operations.")
        return SAMPLE_DATA_PATH, "synthetic"
    return RAW_DATA_PATH, "real"


def generate_synthetic_dataset(n_trainsets: int = 25, n_days: int = 60, seed: int = SEED) -> pd.DataFrame:
    """Create a SYNTHETIC table with the notebook dataset's columns (see module warning)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    base_mileage = rng.integers(15_000, 60_000, size=n_trainsets)
    rows = []
    for day_idx, date in enumerate(dates):
        for t in range(n_trainsets):
            rolling = rng.random() > 0.05
            signalling = rng.random() > 0.05
            telecom = rng.random() > 0.05
            any_invalid = not (rolling and signalling and telecom)
            open_cards = int(rng.poisson(0.6))
            critical = bool(rng.random() < 0.05)
            branding = int(rng.integers(1, 300)) if rng.random() < 0.5 else 0
            cum_mileage = int(base_mileage[t] + day_idx * 550 + rng.integers(-300, 300))
            deep_clean = int(rng.integers(0, 31))
            scheduled = rng.random() < 0.4
            # Arbitrary synthetic rule (NOT an operational rule):
            logit = 2.0 - 4.0 * any_invalid - 4.0 * critical - 0.5 * open_cards + rng.normal(0, 0.7)
            service = rng.random() < 1.0 / (1.0 + np.exp(-logit))
            km = int(np.clip(rng.normal(550, 80), 300, 800)) if service else 0
            rows.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Trainset": f"SYN-{t + 1:02d}",
                "Depot": str(rng.choice(["DepotA", "DepotB"])),
                "FinalDecision": "Service" if service else "Standby",
                "DecisionReason": "synthetic-service" if service else "synthetic-standby",
                "Rolling_Valid": bool(rolling),
                "Signalling_Valid": bool(signalling),
                "Telecom_Valid": bool(telecom),
                "Any_Fitness_Invalid": bool(any_invalid),
                "Open_JobCards": open_cards,
                "Critical_JobCard": critical,
                "Branding_Hours_Left": branding,
                "Cum_Mileage": cum_mileage,
                "Mileage_Label": None,  # filled below relative to the fleet mean
                "Days_Since_Last_Deep_Clean": deep_clean,
                "Cleaning_Slot_Scheduled": "Scheduled" if scheduled else None,
                "Bay_Position": int(rng.integers(1, 31)),
                "Stabling_Difficulty": str(rng.choice(["Low", "Medium", "High"])),
                "Daily_Km_Assigned": km,
            })
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    mean_mileage = df["Cum_Mileage"].mean()
    df["Mileage_Label"] = np.where(
        df["Cum_Mileage"] < 0.8 * mean_mileage, "Underused",
        np.where(df["Cum_Mileage"] > 1.2 * mean_mileage, "Overused", "Balanced"),
    )
    return df


def write_synthetic_sample(path: str | Path = SAMPLE_DATA_PATH, **kwargs) -> Path:
    """Generate the synthetic dataset and write it to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_dataset(**kwargs)
    df.to_csv(path, index=False)
    logger.info("Wrote SYNTHETIC sample (%d rows) to %s", len(df), path)
    return path
