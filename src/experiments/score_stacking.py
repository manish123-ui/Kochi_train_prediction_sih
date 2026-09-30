"""Score-prediction experiment: tree model (LightGBM or RandomForest) -> stacked MLP -> ensemble.

Refactor of notebook cell 15. That cell NEVER RAN (it has no outputs), it works on a DIFFERENT
file (``Copy of output_mileage_mtd_scored(1).csv``, target ``score``) whose columns are not
visible in the notebook, and it is unrelated to the FinalDecision / Daily_Km models. It is kept
here as an experiment, not as part of the prediction pipeline.

    python -m src.experiments.score_stacking --data path/to/scored.csv --target score
    python -m src.experiments.score_stacking --data path/to/scored.csv --no-mlp   # trees only

Fixes relative to the notebook (details in docs/methodology.md):
  * imputation medians, winsorisation bounds and category codes were computed on the FULL data
    before splitting; they are now fitted on the training split only (``ScorePreprocessor``);
  * LightGBM 4 removed ``early_stopping_rounds``/``verbose_eval``; callbacks are used instead;
  * the feature-importance table used a name list one longer than the importance vector, and the
    error was swallowed by ``try/except``; importance is now averaged over folds, base features only;
  * "best weights" were kept with a non-copying ``.cpu()``; ``deepcopy`` is used (via fit_model);
  * only the last fold's booster was saved; every fold's booster is saved now;
  * small-cardinality NUMERIC columns were turned into lexicographic string category codes
    ("10" < "2"); they now stay numeric. String columns are still category-coded.

Caution: if ``score`` is a deterministic function of the other columns (its file name suggests it
was computed), a high R^2 would only mean the model re-learned that formula.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import warnings
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.evaluation.metrics import regression_metrics, save_json

logger = logging.getLogger(__name__)

ID_LIKE_NAMES = {"id", "index", "serial"}


@dataclasses.dataclass(frozen=True)
class ScoreExperimentConfig:
    target: str = "score"  # ORIGINAL
    test_size: float = 0.15  # ORIGINAL
    val_size: float = 0.15  # ORIGINAL (fraction of the non-test part)
    n_folds: int = 5  # ORIGINAL
    seed: int = cfg.SEED
    clip_quantiles: tuple[float, float] = (0.01, 0.99)  # ORIGINAL winsorisation
    max_int_categories: int = 30  # kept for STRING columns only; see module docstring
    lgb_rounds: int = 5000  # ORIGINAL
    lgb_early_stopping: int = 200  # ORIGINAL
    rf_trees: int = 300  # ORIGINAL fallback
    use_mlp: bool = True
    mlp_hidden: tuple[int, ...] = (256, 128)  # ORIGINAL
    mlp_dropout: float = 0.25  # ORIGINAL
    mlp_epochs: int = 400  # ORIGINAL
    mlp_lr: float = 1e-3  # ORIGINAL
    mlp_batch_size: int = 64  # ORIGINAL
    mlp_patience: int = 40  # ORIGINAL
    tree_weight: float = 0.5  # ORIGINAL: ensemble = 0.5*tree + 0.5*mlp


# --------------------------------------------------------------------------- #
# Preprocessing fitted on train only
# --------------------------------------------------------------------------- #
def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _parse_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", format="mixed")


class ScorePreprocessor:
    """Automatic preprocessing for an unknown tabular file; learns everything from ``fit`` data.

    Column roles are decided on the training data: text that is mostly numeric -> numeric; text
    that is mostly dates -> year/month/day/weekday; other text -> category codes (unseen -> -1);
    numeric -> median-imputed and winsorised (if >5 distinct values); constant columns dropped.
    """

    def __init__(self, target: str, clip_quantiles=(0.01, 0.99)):
        self.target = target
        self.clip_quantiles = clip_quantiles

    def fit(self, df: pd.DataFrame) -> "ScorePreprocessor":
        X = df.drop(columns=[self.target] + [c for c in df.columns if c.lower() in ID_LIKE_NAMES], errors="ignore")
        self.plan_: dict[str, dict] = {}
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_bool_dtype(s):
                s = s.astype(int)
            if _is_text(s):
                numeric = pd.to_numeric(s, errors="coerce")
                if numeric.notna().mean() >= 0.5:
                    self.plan_[c] = {"kind": "numeric", "from_text": True}
                    s = numeric
                else:
                    dates = _parse_datetime(s)
                    if dates.notna().mean() >= 0.5:
                        self.plan_[c] = {"kind": "datetime"}
                        continue
                    cats = sorted(s.astype(object).where(s.notna(), "__missing__").astype(str).unique())
                    self.plan_[c] = {"kind": "category", "codes": {v: i for i, v in enumerate(cats)}}
                    continue
            else:
                self.plan_[c] = {"kind": "numeric", "from_text": False}
            med = float(pd.to_numeric(s, errors="coerce").median())
            med = 0.0 if np.isnan(med) else med
            filled = pd.to_numeric(s, errors="coerce").fillna(med)
            plan = self.plan_[c]
            plan["median"] = med
            if filled.nunique() > 5:
                plan["bounds"] = (float(filled.quantile(self.clip_quantiles[0])), float(filled.quantile(self.clip_quantiles[1])))
        full = self._transform_all(df)
        self.feature_names_ = [c for c in full.columns if full[c].nunique(dropna=True) > 1]
        dropped = sorted(set(full.columns) - set(self.feature_names_))
        if dropped:
            logger.info("Dropping %d constant column(s): %s", len(dropped), dropped)
        return self

    def _transform_all(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for c, plan in self.plan_.items():
            if c not in df.columns:
                raise ValueError(f"Column {c!r} missing at transform time")
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                s = s.astype(int)
            if plan["kind"] == "numeric":
                v = pd.to_numeric(s, errors="coerce").fillna(plan["median"])
                if "bounds" in plan:
                    v = v.clip(*plan["bounds"])
                cols[c] = v.astype(float)
            elif plan["kind"] == "datetime":
                d = _parse_datetime(s)
                for part in ("year", "month", "day", "weekday"):
                    cols[f"{c}_{part}"] = getattr(d.dt, part).fillna(0).astype(float)
            else:
                text = s.astype(object).where(s.notna(), "__missing__").astype(str)
                cols[f"{c}_code"] = text.map(plan["codes"]).fillna(-1).astype(float)
        return pd.DataFrame(cols, index=df.index)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._transform_all(df)[self.feature_names_].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Tree stage: out-of-fold predictions
# --------------------------------------------------------------------------- #
def _lightgbm_module():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def tree_oof_predictions(X_train, y_train, X_val, X_test, feature_names, config: ScoreExperimentConfig):
    """K-fold OOF predictions on train + fold-averaged predictions for val/test.

    Returns ``(oof, val_pred, test_pred, importance, backend, models)``.
    """
    lgb = _lightgbm_module()
    backend = "lightgbm" if lgb is not None else "random_forest"
    if lgb is None:
        logger.warning("lightgbm not installed -> falling back to RandomForest (as the notebook did)")
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    oof = np.zeros(len(y_train))
    val_pred = np.zeros(len(X_val))
    test_pred = np.zeros(len(X_test))
    importance = np.zeros(len(feature_names))
    models = []
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05, "num_leaves": 31,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbosity": -1,
              "seed": config.seed, "lambda_l1": 0.1, "lambda_l2": 0.1}
    for fold, (tr, va) in enumerate(kf.split(X_train, y_train), start=1):
        logger.info("%s fold %d/%d", backend, fold, config.n_folds)
        if lgb is not None:
            booster = lgb.train(
                params, lgb.Dataset(X_train[tr], label=y_train[tr]), num_boost_round=config.lgb_rounds,
                valid_sets=[lgb.Dataset(X_train[va], label=y_train[va])],
                callbacks=[lgb.early_stopping(config.lgb_early_stopping, verbose=False), lgb.log_evaluation(200)],
            )
            predict = lambda X, b=booster: b.predict(X, num_iteration=b.best_iteration)  # noqa: E731
            importance += booster.feature_importance(importance_type="gain") / config.n_folds
            models.append(booster)
        else:
            forest = RandomForestRegressor(n_estimators=config.rf_trees, random_state=config.seed, n_jobs=-1)
            forest.fit(X_train[tr], y_train[tr])
            predict = forest.predict
            importance += forest.feature_importances_ / config.n_folds
            models.append(forest)
        oof[va] = predict(X_train[va])
        val_pred += predict(X_val) / config.n_folds
        test_pred += predict(X_test) / config.n_folds
    return oof, val_pred, test_pred, importance, backend, models


# --------------------------------------------------------------------------- #
# MLP stage (needs PyTorch; imported lazily so the tree stage works without it)
# --------------------------------------------------------------------------- #
def _train_stacked_mlp(Z_train, y_train, Z_val, y_val, config: ScoreExperimentConfig):
    import torch
    from torch import nn

    from src.prediction.common import fit_model, get_device, make_loader, predict_array, set_seed

    set_seed(config.seed)
    device = get_device("auto")
    layers, in_dim = [], Z_train.shape[1]
    for h in config.mlp_hidden:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(config.mlp_dropout)]
        in_dim = h
    model = nn.Sequential(*layers, nn.Linear(in_dim, 1), nn.Flatten(start_dim=0))  # -> shape (B,)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.mlp_lr, weight_decay=1e-4)
    history = fit_model(
        model, make_loader(Z_train, y_train, config.mlp_batch_size, True, config.seed),
        make_loader(Z_val, y_val, config.mlp_batch_size, False), loss_fn=nn.MSELoss(), optimizer=optimizer,
        epochs=config.mlp_epochs, device=device, patience=config.mlp_patience, log_every=25, label="score-mlp")
    return model, history, (lambda Z: predict_array(model, Z, device))


def _versions() -> dict:
    out = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "lightgbm", "torch"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def run_score_experiment(df: pd.DataFrame, config: ScoreExperimentConfig = ScoreExperimentConfig(),
                         output_dir: str | Path | None = None) -> dict:
    """Run the experiment and return a report dict (also written to ``output_dir`` if given)."""
    if config.target not in df.columns:
        raise ValueError(f"Target column {config.target!r} not found. Columns: {list(df.columns)}")
    df = df[pd.to_numeric(df[config.target], errors="coerce").notna()]
    train_val, test = train_test_split(df, test_size=config.test_size, random_state=config.seed)
    train, val = train_test_split(train_val, test_size=config.val_size, random_state=config.seed)

    pre = ScorePreprocessor(config.target, config.clip_quantiles).fit(train)  # TRAIN ONLY
    X_tr, X_va, X_te = (pre.transform(d) for d in (train, val, test))
    y_tr, y_va, y_te = (pd.to_numeric(d[config.target]).to_numpy(float) for d in (train, val, test))
    logger.info("Features: %d | train %d, val %d, test %d", len(pre.feature_names_), len(train), len(val), len(test))

    oof, tree_val, tree_test, imp, backend, models = tree_oof_predictions(
        X_tr, y_tr, X_va, X_te, pre.feature_names_, config)
    report: dict = {
        "note": "Experiment on a different file than the induction models. Test metrics are the honest ones; "
                "MLP/ensemble validation metrics are optimistic (val drives MLP early stopping).",
        "config": dataclasses.asdict(config), "tree_backend": backend, "n_features": len(pre.feature_names_),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "metrics": {"tree_oof_train": regression_metrics(y_tr, oof),
                    "tree_val": regression_metrics(y_va, tree_val), "tree_test": regression_metrics(y_te, tree_test)},
        "top_feature_importance": dict(sorted(zip(pre.feature_names_, map(float, imp)), key=lambda kv: -kv[1])[:30]),
        "versions": _versions(),
    }

    mlp_state = None
    if config.use_mlp:
        scaler = StandardScaler().fit(np.hstack([X_tr, oof[:, None]]))  # OOF as an extra feature (stacking)
        Z_tr, Z_va, Z_te = (scaler.transform(np.hstack([X, p[:, None]]))
                            for X, p in ((X_tr, oof), (X_va, tree_val), (X_te, tree_test)))
        model, history, predict = _train_stacked_mlp(Z_tr, y_tr, Z_va, y_va, config)
        mlp_val, mlp_test = predict(Z_va), predict(Z_te)
        w = config.tree_weight
        report["metrics"].update({
            "mlp_val": regression_metrics(y_va, mlp_val), "mlp_test": regression_metrics(y_te, mlp_test),
            "ensemble_val": regression_metrics(y_va, w * tree_val + (1 - w) * mlp_val),
            "ensemble_test": regression_metrics(y_te, w * tree_test + (1 - w) * mlp_test),
        })
        report["mlp_history"] = history.as_dict()
        mlp_state = (model, scaler)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(report, out / "score_experiment_report.json")
        joblib.dump(pre, out / "score_preprocessor.joblib")
        if backend == "lightgbm":
            for i, booster in enumerate(models, start=1):
                booster.save_model(str(out / f"lgb_fold{i}.txt"))
        if mlp_state is not None:
            import torch
            torch.save(mlp_state[0].state_dict(), out / "score_mlp_stack.pt")
            joblib.dump(mlp_state[1], out / "score_stack_scaler.joblib")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="CSV with the score column")
    parser.add_argument("--target", default="score")
    parser.add_argument("--no-mlp", action="store_true", help="tree stage only (no PyTorch needed)")
    parser.add_argument("--output-dir", type=Path, default=cfg.DEFAULT_RUN_DIR / "score_experiment")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = ScoreExperimentConfig(target=args.target, use_mlp=not args.no_mlp)
    report = run_score_experiment(pd.read_csv(args.data), config, args.output_dir)
    for name, m in report["metrics"].items():
        logger.info("%-16s RMSE=%.4f MAE=%.4f R2=%.4f", name, m["rmse"], m["mae"], m["r2"])


if __name__ == "__main__":
    main()
