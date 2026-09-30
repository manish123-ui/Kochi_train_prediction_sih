"""Train the Daily_Km_Assigned regressor.

    python -m src.prediction.train_regressor --use-sample --model simple
    python -m src.prediction.train_regressor --data path/to.csv --model ultra_complex

Replaces the notebook's two near-identical pipelines (cells 9 and 12). Same split for every
scaler variant; the best variant is chosen on VALIDATION RMSE. The notebook picked the "best"
model by test R^2, which leaks the test set into model selection.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src import config as cfg
from src.data.loader import load_raw_dataset, resolve_data_path
from src.data.preprocessing import (
    REGRESSOR_TASK, build_preprocessor, build_task_frame, drop_unlabeled_rows, split_frame,
)
from src.evaluation.metrics import regression_metrics, save_json
from src.evaluation.visualization import plot_pred_vs_actual, plot_training_history
from src.prediction.common import fit_model, get_device, make_loader, run_environment, set_seed
from src.prediction.daily_km_regressor import DailyKmRegressor, build_regressor

logger = logging.getLogger(__name__)


def _training_components(model: nn.Module, config: cfg.RegressorTrainConfig, steps_per_epoch: int):
    loss_fn = nn.MSELoss() if config.loss == "mse" else nn.HuberLoss(delta=config.huber_delta)
    opt_cls = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.scheduler == "plateau":
        return loss_fn, optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10), "plateau"
    if config.scheduler == "one_cycle":
        return loss_fn, optimizer, torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate, epochs=config.epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.1), "batch"
    return loss_fn, optimizer, None, "none"


def train_regressor(
    df: pd.DataFrame,
    config: cfg.RegressorTrainConfig | None = None,
    *,
    output_dir: str | Path = cfg.REGRESSOR_DIR,
    report_dir: str | Path | None = None,
    device: str = "auto",
    data_source: str = "unknown",
    save: bool = True,
) -> tuple[DailyKmRegressor, pd.DataFrame, dict]:
    """Train one model per scaler; return ``(best_regressor, comparison_table, report)``."""
    config = config or cfg.REGRESSOR_PRESETS["simple"]
    dev = get_device(device)

    df = drop_unlabeled_rows(df, REGRESSOR_TASK)
    split = split_frame(df, test_size=config.test_size, val_size=config.val_size, seed=config.seed,
                        strategy=config.split_strategy)
    logger.info("Split sizes: %s (strategy=%s)", split.sizes(), config.split_strategy)
    X_tr, y_tr = build_task_frame(split.train, REGRESSOR_TASK)
    X_va, y_va = build_task_frame(split.val, REGRESSOR_TASK)
    _, y_te = build_task_frame(split.test, REGRESSOR_TASK)

    rows, candidates = [], {}
    for scaler in config.scalers:
        logger.info("=== %s model, %s scaler ===", config.model_type, scaler)
        set_seed(config.seed)  # identical initialisation for every scaler variant
        preprocessor = build_preprocessor(REGRESSOR_TASK.numeric_features, REGRESSOR_TASK.categorical_features, scaler)
        Z_tr = preprocessor.fit_transform(X_tr)  # fitted on TRAIN only
        Z_va = preprocessor.transform(X_va)
        model = build_regressor(config.model_type, {"input_size": Z_tr.shape[1]})
        train_loader = make_loader(Z_tr, y_tr, config.batch_size, shuffle=True, seed=config.seed)
        val_loader = make_loader(Z_va, y_va, config.batch_size, shuffle=False)
        loss_fn, optimizer, scheduler, mode = _training_components(model, config, len(train_loader))
        history = fit_model(model, train_loader, val_loader, loss_fn=loss_fn, optimizer=optimizer,
                            epochs=config.epochs, device=dev, scheduler=scheduler, scheduler_mode=mode,
                            grad_clip=config.grad_clip, patience=config.patience,
                            label=f"{config.model_type}/{scaler}")
        regressor = DailyKmRegressor(model, preprocessor, config.model_type, device=device)
        val_m = regression_metrics(y_va, regressor.predict(split.val))
        test_m = regression_metrics(y_te, regressor.predict(split.test))
        logger.info("%s/%s VAL rmse=%.2f r2=%.4f | TEST rmse=%.2f mae=%.2f r2=%.4f", config.model_type, scaler,
                    val_m["rmse"], val_m["r2"], test_m["rmse"], test_m["mae"], test_m["r2"])
        rows.append({"model": config.model_type, "scaler": scaler,
                     **{f"val_{k}": v for k, v in val_m.items() if k != "n"},
                     **{f"test_{k}": v for k, v in test_m.items() if k != "n"},
                     "epochs_run": len(history.train_loss), "best_epoch": history.best_epoch})
        candidates[scaler] = (regressor, history)

    table = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    best_scaler = str(table.loc[0, "scaler"])
    best, best_history = candidates[best_scaler]
    logger.info("Selected scaler=%s by validation RMSE (test metrics were NOT used for selection).", best_scaler)

    report = {
        "task": "Daily_Km_Assigned regression",
        "data_source": data_source,
        "rows": {"total_labeled": int(len(df)), **split.sizes()},
        "selected": {"model": config.model_type, "scaler": best_scaler, "selection_metric": "validation RMSE"},
        "config": dataclasses.asdict(config),
        "comparison": json.loads(table.to_json(orient="records")),
        "history": best_history.as_dict(),
        "environment": run_environment(),
    }
    best.metadata = report
    if save:
        best.save(output_dir)
    if report_dir is not None:
        report_dir = Path(report_dir)
        save_json(report, report_dir / "regressor_report.json")
        table.to_csv(report_dir / "regressor_comparison.csv", index=False)
        plot_training_history(best_history.as_dict(), f"Regressor training ({config.model_type}/{best_scaler})", "Loss",
                              report_dir / "regressor_history.png")
        plot_pred_vs_actual(y_te, best.predict(split.test), f"Test: predicted vs actual daily km ({best_scaler})",
                            save_path=report_dir / "regressor_pred_vs_actual_test.png")
    return best, table, report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--use-sample", action="store_true", help="use the SYNTHETIC sample data")
    parser.add_argument("--model", choices=sorted(cfg.REGRESSOR_PRESETS), default="simple")
    parser.add_argument("--scalers", nargs="+", choices=["standard", "minmax", "robust"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--split-strategy", choices=["random", "chronological"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=cfg.REGRESSOR_DIR)
    parser.add_argument("--report-dir", type=Path, default=cfg.DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    overrides = {}
    if args.scalers:
        overrides["scalers"] = tuple(args.scalers)
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.split_strategy:
        overrides["split_strategy"] = args.split_strategy
    config = dataclasses.replace(cfg.REGRESSOR_PRESETS[args.model], **overrides)
    path, source = resolve_data_path(args.data, args.use_sample)
    train_regressor(load_raw_dataset(path), config, output_dir=args.output_dir, report_dir=args.report_dir,
                    device=args.device, data_source=source)


if __name__ == "__main__":
    main()
