"""Command-line entry point for the Kochi Metro train-induction project.

    python -m src.main generate-sample                 # write SYNTHETIC smoke-test data
    python -m src.main train-classifier --use-sample   # FinalDecision model
    python -m src.main train-regressor  --use-sample   # Daily_Km model
    python -m src.main predict          --use-sample   # both models -> predictions.csv
    python -m src.main schedule --input predictions.csv [--constraints c.json]
    python -m src.main run-all --use-sample            # whole pipeline, end to end

Pipeline:  data -> preprocessing -> classifier (Service/Standby) + regressor (daily km)
           -> constraints -> scheduling scaffold -> plan.
Options of ``train-classifier`` / ``train-regressor`` / ``predict`` are documented by
``python -m src.main <command> --help``.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from src import config as cfg

logger = logging.getLogger("kochi")

TORCH_HINT = "PyTorch is required for this command. Install dependencies with: pip install -r requirements.txt"


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_constraints(path: Path | None):
    from src.optimization.constraints import constraints_from_specs, load_constraint_specs

    specs = load_constraint_specs(path) if path else cfg.OPERATIONAL_CONSTRAINT_SPECS
    return constraints_from_specs(specs)


def cmd_generate_sample(args) -> None:
    from src.data.loader import write_synthetic_sample

    path = write_synthetic_sample(args.output, n_trainsets=args.n_trainsets, n_days=args.n_days, seed=args.seed)
    logger.warning("SYNTHETIC data written to %s. Labels follow an arbitrary rule; not real operations.", path)


def cmd_schedule(args) -> None:
    from src.data.loader import load_raw_dataset, resolve_data_path
    from src.optimization.scheduler import TrainPlanScheduler

    path, _ = resolve_data_path(args.input, args.use_sample)
    df = load_raw_dataset(path)
    if args.from_labels:
        # DEMO ONLY: treat the file's own labels as if they were model predictions.
        logger.warning("--from-labels: using ground-truth FinalDecision/Daily_Km_Assigned as 'predictions' (demo).")
        df = df.assign(Predicted_Decision=df[cfg.CLASSIFIER_TARGET],
                       Predicted_Daily_Km=df[cfg.REGRESSOR_TARGET].astype(float))
    plan = TrainPlanScheduler(_build_constraints(args.constraints)).build_plan(df)
    plan.save(args.output_dir)
    logger.info("Plan KPIs: %s", plan.kpis)
    logger.info("Constraint violations: %d | optimized=%s (optimisation is future work) | saved to %s",
                len(plan.violations), plan.optimized, args.output_dir)


def cmd_run_all(args) -> None:
    from src.data.loader import load_raw_dataset, resolve_data_path
    from src.optimization.scheduler import TrainPlanScheduler
    from src.prediction.inference import InductionPipeline
    from src.prediction.train_classifier import train_classifier
    from src.prediction.train_regressor import train_regressor

    path, source = resolve_data_path(args.data, args.use_sample)
    df = load_raw_dataset(path)
    out = args.output_dir
    clf_cfg, reg_cfg = cfg.ClassifierTrainConfig(), cfg.REGRESSOR_PRESETS[args.model]
    if args.epochs is not None:
        clf_cfg = dataclasses.replace(clf_cfg, epochs=args.epochs)
        reg_cfg = dataclasses.replace(reg_cfg, epochs=args.epochs)

    logger.info("[1/4] Training FinalDecision classifier")
    classifier, _ = train_classifier(df, clf_cfg, report_dir=out, device=args.device, data_source=source)
    logger.info("[2/4] Training Daily_Km regressor (%s)", args.model)
    regressor, _, _ = train_regressor(df, reg_cfg, report_dir=out, device=args.device, data_source=source)
    logger.info("[3/4] Predicting")
    logger.warning("Predictions below are on the TRAINING file (in-sample): a pipeline demo, not an evaluation. "
                   "Use `predict --input <new.csv>` for new data.")
    preds = InductionPipeline(classifier, regressor).predict(df)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False)
    logger.info("[4/4] Building plan (ML pass-through baseline; no optimiser)")
    plan = TrainPlanScheduler(_build_constraints(args.constraints)).build_plan(preds)
    plan.save(out)
    logger.info("Done. Outputs in %s | KPIs: %s", out, plan.kpis)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-sample", help="write SYNTHETIC smoke-test data")
    p.add_argument("--output", type=Path, default=cfg.SAMPLE_DATA_PATH)
    p.add_argument("--n-trainsets", type=int, default=25)
    p.add_argument("--n-days", type=int, default=60)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.set_defaults(func=cmd_generate_sample)

    for name, helptext in (("train-classifier", "train the FinalDecision classifier"),
                           ("train-regressor", "train the Daily_Km regressor"),
                           ("predict", "run both models on a CSV")):
        sub.add_parser(name, help=helptext, add_help=False)  # options forwarded to the module's own parser

    p = sub.add_parser("schedule", help="build a plan from a predictions CSV")
    p.add_argument("--input", type=Path, help="predictions CSV (needs Predicted_Decision, Predicted_Daily_Km)")
    p.add_argument("--use-sample", action="store_true")
    p.add_argument("--from-labels", action="store_true", help="DEMO: use the file's true labels as predictions")
    p.add_argument("--constraints", type=Path, help="JSON list of constraint specs")
    p.add_argument("--output-dir", type=Path, default=cfg.DEFAULT_RUN_DIR)
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("run-all", help="train both models, predict, build a plan")
    p.add_argument("--data", type=Path)
    p.add_argument("--use-sample", action="store_true")
    p.add_argument("--model", choices=sorted(cfg.REGRESSOR_PRESETS), default="simple")
    p.add_argument("--epochs", type=int, help="override epochs for both models (quick runs)")
    p.add_argument("--constraints", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=cfg.DEFAULT_RUN_DIR)
    p.set_defaults(func=cmd_run_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    _setup_logging()
    forwarded = {"train-classifier": "src.prediction.train_classifier",
                 "train-regressor": "src.prediction.train_regressor",
                 "predict": "src.prediction.inference"}
    try:
        if argv and argv[0] in forwarded:
            import importlib

            importlib.import_module(forwarded[argv[0]]).main(argv[1:])
            return
        args = build_parser().parse_args(argv)
        args.func(args)
    except ImportError as exc:
        if "torch" in str(exc):
            raise SystemExit(TORCH_HINT) from exc
        raise
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
