"""Inference: classifier (Service/Standby) + regressor (daily km) -> one prediction table.

    python -m src.prediction.inference --input new_day.csv --output predictions.csv
    python -m src.prediction.inference --use-sample --output predictions.csv

The output keeps every input column and appends:
    P_Service               classifier probability of Service
    Predicted_Decision      "Service" / "Standby" (threshold from training)
    Predicted_Daily_Km_Raw  regressor output as-is
    Predicted_Daily_Km      Raw, clipped at CLIP_MIN_KM; optionally 0 for Standby (see config)
The input needs the feature columns only. Target columns are not required.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.data.loader import load_raw_dataset, resolve_data_path
from src.prediction.daily_km_regressor import DailyKmRegressor
from src.prediction.induction_classifier import InductionClassifier

logger = logging.getLogger(__name__)


class InductionPipeline:
    """Runs both trained models on a raw dataframe."""

    def __init__(self, classifier: InductionClassifier, regressor: DailyKmRegressor, *,
                 gate_km_by_decision: bool = cfg.GATE_KM_BY_DECISION, clip_min_km: float = cfg.CLIP_MIN_KM):
        self.classifier = classifier
        self.regressor = regressor
        self.gate_km_by_decision = gate_km_by_decision
        self.clip_min_km = clip_min_km

    @classmethod
    def load(cls, classifier_dir: str | Path = cfg.CLASSIFIER_DIR, regressor_dir: str | Path = cfg.REGRESSOR_DIR,
             device: str = "auto", **kwargs) -> "InductionPipeline":
        return cls(InductionClassifier.load(classifier_dir, device), DailyKmRegressor.load(regressor_dir, device), **kwargs)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        proba = self.classifier.predict_proba(df)
        decision = np.where(proba >= self.classifier.threshold, cfg.SERVICE_LABEL, cfg.STANDBY_LABEL)
        km_raw = self.regressor.predict(df)
        km = np.clip(km_raw, self.clip_min_km, None)
        if self.gate_km_by_decision:
            km = np.where(decision == cfg.STANDBY_LABEL, 0.0, km)
        out["P_Service"] = proba
        out["Predicted_Decision"] = decision
        out["Predicted_Daily_Km_Raw"] = km_raw
        out["Predicted_Daily_Km"] = km
        return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="CSV with the feature columns")
    parser.add_argument("--use-sample", action="store_true", help="predict on the SYNTHETIC sample data")
    parser.add_argument("--output", type=Path, default=cfg.DEFAULT_RUN_DIR / "predictions.csv")
    parser.add_argument("--classifier-dir", type=Path, default=cfg.CLASSIFIER_DIR)
    parser.add_argument("--regressor-dir", type=Path, default=cfg.REGRESSOR_DIR)
    parser.add_argument("--gate-km", action="store_true", help="force Predicted_Daily_Km=0 for predicted Standby (UNVERIFIED assumption)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    path, _ = resolve_data_path(args.input, args.use_sample)
    pipeline = InductionPipeline.load(args.classifier_dir, args.regressor_dir, device=args.device,
                                      gate_km_by_decision=args.gate_km or cfg.GATE_KM_BY_DECISION)
    preds = pipeline.predict(load_raw_dataset(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(preds), args.output)


if __name__ == "__main__":
    main()
