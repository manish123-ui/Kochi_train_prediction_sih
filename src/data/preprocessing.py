"""Leak-free preprocessing shared by the classifier, regressor and experiments.

Design (see docs/methodology.md for what changed relative to the notebook):

1. Stateless, deterministic encodings (:func:`encode_features`) - safe to apply
   before splitting because they learn nothing from the data.
2. Split the *raw* frame (:func:`split_frame`).
3. Learned steps (median imputation, scaling, one-hot) live in a scikit-learn
   ``ColumnTransformer`` (:func:`build_preprocessor`) that is fitted on the
   training split only and then applied to validation/test/new data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src import config as cfg
from src.data.loader import validate_columns

logger = logging.getLogger(__name__)

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
MISSING_CATEGORY = "__missing__"


# --------------------------------------------------------------------------- #
# Task specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskSpec:
    """Which columns a task uses and what its target is."""

    name: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target: str

    @property
    def feature_columns(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)


CLASSIFIER_TASK = TaskSpec(
    "induction_classifier",
    tuple(cfg.CLASSIFIER_NUMERIC_FEATURES),
    tuple(cfg.CLASSIFIER_CATEGORICAL_FEATURES),
    cfg.CLASSIFIER_TARGET,
)
REGRESSOR_TASK = TaskSpec(
    "daily_km_regressor",
    tuple(cfg.REGRESSOR_NUMERIC_FEATURES),
    tuple(cfg.REGRESSOR_CATEGORICAL_FEATURES),
    cfg.REGRESSOR_TARGET,
)


# --------------------------------------------------------------------------- #
# Stateless encodings
# --------------------------------------------------------------------------- #
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _is_missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


def _to_binary(series: pd.Series) -> pd.Series:
    """bool / 0-1 / 'True'-'False' -> float 1.0/0.0 (unknown or missing -> NaN)."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    out = []
    for v in series.astype(object):
        if _is_missing(v):
            out.append(np.nan)
        elif isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
            out.append(float(v))
        else:
            s = str(v).strip().lower()
            out.append(1.0 if s in _TRUE else 0.0 if s in _FALSE else np.nan)
    return pd.Series(out, index=series.index, dtype=float)


def _map_ordinal(series: pd.Series, mapping: dict[str, int], column: str) -> pd.Series:
    """Map string categories to ints; unknown non-null values become NaN with a warning."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    out, unknown = [], set()
    for v in series.astype(object):
        if _is_missing(v):
            out.append(np.nan)
            continue
        key = str(v).strip()
        if key in mapping:
            out.append(float(mapping[key]))
        else:
            unknown.add(key)
            out.append(np.nan)
    if unknown:
        logger.warning("Column %s: unmapped value(s) %s -> NaN (imputed later)", column, sorted(unknown))
    return pd.Series(out, index=series.index, dtype=float)


def _cleaning_scheduled(series: pd.Series) -> pd.Series:
    """'Scheduled' -> 1.0; anything else, including missing -> 0.0 (as the notebook did)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).fillna(0.0)
    vals = [1.0 if isinstance(v, str) and v.strip() == cfg.CLEANING_SCHEDULED_VALUE else 0.0
            for v in series.astype(object)]
    return pd.Series(vals, index=series.index, dtype=float)


def mileage_category(cum_mileage: pd.Series) -> pd.Series:
    """0=Low (<120k), 1=Medium (<200k), 2=High, from ``Cum_Mileage`` (NaN stays NaN)."""
    values = pd.to_numeric(cum_mileage, errors="coerce")
    cats = np.digitize(values.to_numpy(dtype=float), bins=list(cfg.MILEAGE_CATEGORY_THRESHOLDS)).astype(float)
    cats[values.isna().to_numpy()] = np.nan
    return pd.Series(cats, index=cum_mileage.index, dtype=float)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's deterministic encodings to whichever columns are present."""
    out = df.copy()
    for col in cfg.BOOLEAN_COLUMNS:
        if col in out.columns:
            out[col] = _to_binary(out[col])
    if "Mileage_Label" in out.columns:
        out["Mileage_Label"] = _map_ordinal(out["Mileage_Label"], cfg.MILEAGE_LABEL_MAP, "Mileage_Label")
    if "Stabling_Difficulty" in out.columns:
        out["Stabling_Difficulty"] = _map_ordinal(
            out["Stabling_Difficulty"], cfg.STABLING_DIFFICULTY_MAP, "Stabling_Difficulty")
    if "Cleaning_Slot_Scheduled" in out.columns:
        out["Cleaning_Slot_Scheduled"] = _cleaning_scheduled(out["Cleaning_Slot_Scheduled"])
    if cfg.MILEAGE_COLUMN in out.columns:
        out["Mileage_Category"] = mileage_category(out[cfg.MILEAGE_COLUMN])
    return out


def encode_decision(series: pd.Series) -> pd.Series:
    """'Service'->1, 'Standby'->0. Unknown or missing labels become NaN (dropped by callers)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return _map_ordinal(series, cfg.DECISION_MAP, cfg.CLASSIFIER_TARGET)


# --------------------------------------------------------------------------- #
# Task frames
# --------------------------------------------------------------------------- #
def drop_unlabeled_rows(df: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    """Remove rows whose target is missing or not a valid label (logged, never silent)."""
    validate_columns(df, [task.target], context=f"{task.name} target")
    if task is CLASSIFIER_TASK or task.target == cfg.CLASSIFIER_TARGET:
        valid = encode_decision(df[task.target]).notna()
    else:
        valid = pd.to_numeric(df[task.target], errors="coerce").notna()
    if not valid.all():
        logger.warning("%s: dropping %d row(s) with missing/invalid target %s",
                       task.name, int((~valid).sum()), task.target)
    return df.loc[valid]


def build_task_frame(df: pd.DataFrame, task: TaskSpec, require_target: bool = True):
    """Return ``(X, y)`` for a task; ``y`` is None when ``require_target`` is False.

    ``X`` keeps ``df``'s index so predictions can be aligned with the input rows.
    """
    enc = encode_features(df)
    validate_columns(enc, task.feature_columns, context=f"{task.name} features")
    X = enc[task.feature_columns].copy()
    for col in task.categorical_features:
        X[col] = X[col].astype(object).where(X[col].notna(), MISSING_CATEGORY).astype(str)
    y = None
    if require_target:
        validate_columns(df, [task.target], context=f"{task.name} target")
        if task.target == cfg.CLASSIFIER_TARGET:
            y = encode_decision(df[task.target])
        else:
            y = pd.to_numeric(df[task.target], errors="coerce").astype(float)
        if y.isna().any():
            raise ValueError(f"{task.name}: target contains missing values; call drop_unlabeled_rows first.")
    if require_target and "Mileage_Category" in X.columns and X["Mileage_Category"].nunique(dropna=True) <= 1:
        logger.warning("Mileage_Category is constant in this data (thresholds %s vs Cum_Mileage range "
                       "%s-%s); it carries no signal.", cfg.MILEAGE_CATEGORY_THRESHOLDS,
                       enc[cfg.MILEAGE_COLUMN].min(), enc[cfg.MILEAGE_COLUMN].max())
    return X, y


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def split_frame(
    df: pd.DataFrame,
    *,
    test_size: float,
    val_size: float,
    seed: int = cfg.SEED,
    strategy: str = "random",
    stratify: pd.Series | None = None,
) -> DataSplit:
    """Split a RAW frame into train/val/test. ``val_size`` is a fraction of the non-test part.

    strategy="random"        - shuffled split (what the notebook did).
    strategy="chronological" - whole dates: earliest -> train, then val, latest -> test.
                               Use it to check that same-train-on-adjacent-days rows are not
                               inflating scores in a random split.
    """
    if strategy == "random":
        train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
        strat_tv = stratify.loc[train_val.index] if stratify is not None else None
        train, val = train_test_split(train_val, test_size=val_size, random_state=seed, stratify=strat_tv)
        return DataSplit(train, val, test)
    if strategy == "chronological":
        validate_columns(df, [cfg.DATE_COLUMN], context="chronological split")
        dates = pd.to_datetime(df[cfg.DATE_COLUMN], errors="coerce")
        if dates.isna().any():
            raise ValueError("Chronological split needs parseable values in the Date column.")
        unique = np.sort(dates.unique())
        n_test = max(1, int(round(len(unique) * test_size)))
        n_val = max(1, int(round((len(unique) - n_test) * val_size)))
        test_dates = set(unique[-n_test:])
        val_dates = set(unique[-(n_test + n_val):-n_test])
        is_test = dates.isin(test_dates)
        is_val = dates.isin(val_dates)
        return DataSplit(df.loc[~is_test & ~is_val], df.loc[is_val], df.loc[is_test])
    raise ValueError(f"Unknown split strategy: {strategy!r}")


# --------------------------------------------------------------------------- #
# Learned preprocessing (fit on train only)
# --------------------------------------------------------------------------- #
def build_preprocessor(
    numeric_features: list[str] | tuple[str, ...],
    categorical_features: list[str] | tuple[str, ...] = (),
    scaler: str = "standard",
) -> ColumnTransformer:
    """Median-impute + scale numeric columns; one-hot categorical columns.

    Nothing is learned until ``fit`` is called on the TRAINING split.
    """
    if scaler not in SCALERS:
        raise ValueError(f"Unknown scaler {scaler!r}; choose from {sorted(SCALERS)}")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", SCALERS[scaler]()),
    ])
    transformers = [("num", numeric, list(numeric_features))]
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical_features))
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
