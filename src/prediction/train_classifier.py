"""Train the FinalDecision (Service/Standby) classifier.

    python -m src.prediction.train_classifier --use-sample          # synthetic smoke test
    python -m src.prediction.train_classifier --data path/to.csv    # real (private) data

Pipeline: drop unlabeled rows -> split RAW rows -> deterministic encodings -> fit
scaler/one-hot on TRAIN only -> train MLP -> evaluate on val, then once on test.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch
from torch import nn

from src import config as cfg
from src.data.loader import load_raw_dataset, resolve_data_path
from src.data.preprocessing import (
    CLASSIFIER_TASK, build_preprocessor, build_task_frame, drop_unlabeled_rows, encode_decision, split_frame,
)
from src.evaluation.metrics import classification_metrics, format_metrics, save_json
from src.evaluation.visualization import plot_confusion_matrix, plot_training_history
from src.prediction.common import fit_model, get_device, make_loader, run_environment, set_seed
from src.prediction.induction_classifier import InductionClassifier, InductionMLP

logger = logging.getLogger(__name__)


def train_classifier(
    df,
    config: cfg.ClassifierTrainConfig = cfg.ClassifierTrainConfig(),
    *,
    output_dir: str | Path = cfg.CLASSIFIER_DIR,
    report_dir: str | Path | None = None,
    device: str = "auto",
    data_source: str = "unknown",
    save: bool = True,
) -> tuple[InductionClassifier, dict]:
    """Train and evaluate; returns ``(classifier, report)``."""
    set_seed(config.seed)
    dev = get_device(device)

    df = drop_unlabeled_rows(df, CLASSIFIER_TASK)
    stratify = encode_decision(df[cfg.CLASSIFIER_TARGET]) if (config.stratify and config.split_strategy == "random") else None
    split = split_frame(df, test_size=config.test_size, val_size=config.val_size, seed=config.seed,
                        strategy=config.split_strategy, stratify=stratify)
    logger.info("Split sizes: %s (strategy=%s)", split.sizes(), config.split_strategy)

    X_tr, y_tr = build_task_frame(split.train, CLASSIFIER_TASK)
    X_va, y_va = build_task_frame(split.val, CLASSIFIER_TASK)
    _, y_te = build_task_frame(split.test, CLASSIFIER_TASK)

    preprocessor = build_preprocessor(CLASSIFIER_TASK.numeric_features, CLASSIFIER_TASK.categorical_features, config.scaler)
    Z_tr = preprocessor.fit_transform(X_tr)  # fitted on TRAIN only
    Z_va = preprocessor.transform(X_va)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]

    model = InductionMLP(input_size=Z_tr.shape[1])
    train_loader = make_loader(Z_tr, y_tr, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = make_loader(Z_va, y_va, 256, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = fit_model(model, train_loader, val_loader, loss_fn=nn.BCEWithLogitsLoss(), optimizer=optimizer,
                        epochs=config.epochs, device=dev, patience=config.patience, log_every=5, label="classifier")

    classifier = InductionClassifier(model, preprocessor, threshold=config.decision_threshold, device=device)
    # Evaluate through the public inference path (raw frame in) so train/serve parity is exercised.
    val_metrics = classification_metrics(y_va, classifier.predict_proba(split.val), config.decision_threshold)
    test_metrics = classification_metrics(y_te, classifier.predict_proba(split.test), config.decision_threshold)
    logger.info(format_metrics({k: v for k, v in val_metrics.items() if isinstance(v, float)}, "VAL  "))
    logger.info(format_metrics({k: v for k, v in test_metrics.items() if isinstance(v, float)}, "TEST "))

    report = {
        "task": "FinalDecision classification (Service=1, Standby=0)",
        "data_source": data_source,
        "rows": {"total_labeled": int(len(df)), **split.sizes()},
        "features": feature_names,
        "config": dataclasses.asdict(config),
        "history": history.as_dict(),
        "metrics": {"validation": val_metrics, "test": test_metrics},
        "note": "Test metrics are computed once after training; no selection was done on them.",
        "environment": run_environment(),
    }
    classifier.metadata = report
    if save:
        classifier.save(output_dir)
    if report_dir is not None:
        report_dir = Path(report_dir)
        save_json(report, report_dir / "classifier_report.json")
        plot_training_history(history.as_dict(), "Classifier training", "BCE loss", report_dir / "classifier_history.png")
        plot_confusion_matrix(test_metrics["confusion_matrix"], title="Classifier - test", save_path=report_dir / "classifier_confusion_test.png")
    return classifier, report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="CSV path (default: data/raw/kochi_induction_dataset.csv)")
    parser.add_argument("--use-sample", action="store_true", help="use the SYNTHETIC sample data")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scaler", choices=["standard", "minmax", "robust"])
    parser.add_argument("--split-strategy", choices=["random", "chronological"])
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=cfg.CLASSIFIER_DIR)
    parser.add_argument("--report-dir", type=Path, default=cfg.DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    overrides = {k: v for k, v in {"epochs": args.epochs, "scaler": args.scaler,
                                   "split_strategy": args.split_strategy}.items() if v is not None}
    if args.no_stratify:
        overrides["stratify"] = False
    config = dataclasses.replace(cfg.ClassifierTrainConfig(), **overrides)
    path, source = resolve_data_path(args.data, args.use_sample)
    train_classifier(load_raw_dataset(path), config, output_dir=args.output_dir, report_dir=args.report_dir,
                     device=args.device, data_source=source)


if __name__ == "__main__":
    main()
