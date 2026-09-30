"""Central configuration for the Kochi Metro train-induction project.

Everything the original Colab notebook hard-coded (paths, column lists,
category mappings, hyper-parameters) lives here, so the rest of the code base
contains no magic values.

Values marked "ORIGINAL" are taken from the notebook. Values marked "CHANGED"
deviate from the notebook on purpose; the reasoning is in docs/methodology.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
# The real dataset is private and is never committed. Put it here (gitignored).
RAW_DATA_PATH = DATA_DIR / "raw" / "kochi_induction_dataset.csv"
# Synthetic smoke-test data produced by `python -m src.main generate-sample`.
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "synthetic_kochi_induction_sample.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_DIR = MODELS_DIR / "induction_classifier"
REGRESSOR_DIR = MODELS_DIR / "daily_km_regressor"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_RUN_DIR = OUTPUTS_DIR / "latest"

SEED = 42  # ORIGINAL (notebook used 42 everywhere)

# --------------------------------------------------------------------------- #
# Schema (column names as printed by the notebook: 19 columns, 4500 rows)
# --------------------------------------------------------------------------- #
DATE_COLUMN = "Date"
TRAINSET_COLUMN = "Trainset"
DEPOT_COLUMN = "Depot"
DECISION_REASON_COLUMN = "DecisionReason"  # free text; dropped (label leakage risk)
ID_COLUMNS = [DATE_COLUMN, TRAINSET_COLUMN]

CLASSIFIER_TARGET = "FinalDecision"
REGRESSOR_TARGET = "Daily_Km_Assigned"

SERVICE_LABEL = "Service"
STANDBY_LABEL = "Standby"
DECISION_MAP = {SERVICE_LABEL: 1, STANDBY_LABEL: 0}  # ORIGINAL
DECISION_LABELS = {1: SERVICE_LABEL, 0: STANDBY_LABEL}

BOOLEAN_COLUMNS = [  # ORIGINAL
    "Rolling_Valid",
    "Signalling_Valid",
    "Telecom_Valid",
    "Any_Fitness_Invalid",
    "Critical_JobCard",
]

MILEAGE_COLUMN = "Cum_Mileage"
# CHANGED: one 3-level map for both tasks. The notebook's classifier cell used a
# 2-level map that turns "Balanced" into NaN; its regressor cell used this map.
MILEAGE_LABEL_MAP = {"Underused": 0, "Balanced": 1, "Overused": 2}
STABLING_DIFFICULTY_MAP = {"Low": 0, "Medium": 1, "High": 2}  # ORIGINAL
CLEANING_SCHEDULED_VALUE = "Scheduled"  # anything else (incl. missing) -> 0. ORIGINAL
# ORIGINAL thresholds for the derived Mileage_Category feature (Low/Medium/High).
# NOTE: the only rows the notebook printed have Cum_Mileage of ~20k-53k, i.e. all
# "Low". Check the feature is not constant on your data (a warning is logged).
MILEAGE_CATEGORY_THRESHOLDS = (120_000, 200_000)

# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
# Classifier: ORIGINAL feature list minus `Daily_Km_Assigned` (CHANGED: target
# leakage, see docs/methodology.md) and with `Depot` one-hot encoded (CHANGED: the
# notebook left it as a string column, which would crash StandardScaler).
CLASSIFIER_NUMERIC_FEATURES = [
    "Rolling_Valid",
    "Signalling_Valid",
    "Telecom_Valid",
    "Any_Fitness_Invalid",
    "Open_JobCards",
    "Critical_JobCard",
    "Branding_Hours_Left",
    "Cum_Mileage",
    "Mileage_Label",
    "Days_Since_Last_Deep_Clean",
    "Cleaning_Slot_Scheduled",
    "Bay_Position",
    "Stabling_Difficulty",
    "Mileage_Category",
]
CLASSIFIER_CATEGORICAL_FEATURES = [DEPOT_COLUMN]

# Regressor: ORIGINAL 11 features (notebook output: "Feature set shape: (4500, 11)").
# The notebook dropped Depot, Bay_Position, Stabling_Difficulty and FinalDecision
# for this task without stating why; that choice is preserved.
REGRESSOR_NUMERIC_FEATURES = [
    "Rolling_Valid",
    "Signalling_Valid",
    "Telecom_Valid",
    "Any_Fitness_Invalid",
    "Open_JobCards",
    "Critical_JobCard",
    "Branding_Hours_Left",
    "Cum_Mileage",
    "Mileage_Label",
    "Days_Since_Last_Deep_Clean",
    "Cleaning_Slot_Scheduled",
]
REGRESSOR_CATEGORICAL_FEATURES: list[str] = []


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassifierTrainConfig:
    """FinalDecision classifier (notebook cell 8)."""

    test_size: float = 0.2  # ORIGINAL
    val_size: float = 0.1  # CHANGED: the notebook had no validation split
    stratify: bool = True  # CHANGED: the notebook did not stratify
    split_strategy: str = "random"  # "random" (ORIGINAL) or "chronological"
    scaler: str = "standard"  # ORIGINAL
    batch_size: int = 16  # ORIGINAL
    epochs: int = 20  # ORIGINAL
    learning_rate: float = 1e-3  # ORIGINAL
    weight_decay: float = 0.0  # ORIGINAL (plain Adam)
    patience: int | None = None  # None: run all epochs, then restore best-val weights
    decision_threshold: float = 0.5  # ORIGINAL
    seed: int = SEED


@dataclass(frozen=True)
class RegressorTrainConfig:
    """Daily_Km_Assigned regressor (notebook cells 9 and 12, merged)."""

    model_type: str = "simple"  # "simple" (cell 9) or "ultra_complex" (cell 12)
    test_size: float = 0.10  # ORIGINAL
    val_size: float = 0.10  # ORIGINAL (fraction of the non-test part)
    split_strategy: str = "random"  # ORIGINAL
    scalers: tuple[str, ...] = ("standard", "minmax", "robust")  # ORIGINAL
    batch_size: int = 64  # ORIGINAL
    epochs: int = 500
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam" | "adamw"
    loss: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    scheduler: str = "plateau"  # "plateau" | "one_cycle" | "none"
    grad_clip: float | None = 1.0
    patience: int | None = 20  # CHANGED for "simple": early stopping was commented out
    seed: int = SEED


# Presets reproduce the two notebook variants.
REGRESSOR_PRESETS: dict[str, RegressorTrainConfig] = {
    "simple": RegressorTrainConfig(),
    "ultra_complex": RegressorTrainConfig(
        model_type="ultra_complex",
        epochs=1000,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        loss="huber",
        huber_delta=1.0,
        scheduler="one_cycle",
        grad_clip=1.0,
        patience=30,
    ),
}

# --------------------------------------------------------------------------- #
# Inference / scheduling behaviour
# --------------------------------------------------------------------------- #
# Whether to force Predicted_Daily_Km to 0 for trains predicted "Standby".
# ASSUMPTION, NOT VERIFIED: the notebook only shows one printed row where
# Standby had 0 km. Off by default; validate on the real data first
# (notebooks/01_kochi_metro_eda.ipynb has the check).
GATE_KM_BY_DECISION = False
# Physical sanity clip: a regression net can output negative km.
CLIP_MIN_KM = 0.0

# Operational constraints are NOT defined in the original notebook, so none are
# enabled by default. See src/optimization/constraints.py for the spec format.
OPERATIONAL_CONSTRAINT_SPECS: list[dict] = []
