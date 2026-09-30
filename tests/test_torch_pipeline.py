"""End-to-end PyTorch tests (skipped automatically when PyTorch is not installed).

They train tiny models for a few epochs on SYNTHETIC data, so they check plumbing (shapes, save/load
parity, column contracts), not model quality.
"""
import dataclasses
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

import numpy as np

from src import config as cfg
from src.data.loader import generate_synthetic_dataset


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestArchitectures(unittest.TestCase):
    def test_ultra_complex_matches_notebook_parameter_count(self):
        from src.prediction.daily_km_regressor import UltraComplexRegressionMLP

        n = sum(p.numel() for p in UltraComplexRegressionMLP(11).parameters())
        self.assertEqual(n, 1_048_471)  # printed by the original notebook (cell 12)

    def test_forward_shapes_including_batch_of_one(self):
        from src.prediction.daily_km_regressor import SimpleRegressionMLP, UltraComplexRegressionMLP
        from src.prediction.induction_classifier import InductionMLP

        for model in (InductionMLP(16), SimpleRegressionMLP(11), UltraComplexRegressionMLP(11)):
            model.eval()
            width = model.input_size
            self.assertEqual(tuple(model(torch.zeros(1, width)).shape), (1,))
            self.assertEqual(tuple(model(torch.zeros(5, width)).shape), (5,))


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic_dataset(n_trainsets=12, n_days=30, seed=3)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_classifier_train_save_load_parity(self):
        from src.prediction.induction_classifier import InductionClassifier
        from src.prediction.train_classifier import train_classifier

        conf = dataclasses.replace(cfg.ClassifierTrainConfig(), epochs=3)
        clf, report = train_classifier(self.df, conf, output_dir=self.root / "clf", report_dir=self.root / "rep",
                                       data_source="synthetic")
        self.assertIn("accuracy", report["metrics"]["test"])
        self.assertNotIn("Daily_Km_Assigned", report["features"])
        loaded = InductionClassifier.load(self.root / "clf")
        np.testing.assert_allclose(clf.predict_proba(self.df), loaded.predict_proba(self.df), rtol=1e-5, atol=1e-6)
        feats_only = self.df.drop(columns=["FinalDecision", "Daily_Km_Assigned", "DecisionReason"])
        self.assertEqual(len(loaded.predict_labels(feats_only)), len(self.df))
        self.assertTrue((self.root / "rep" / "classifier_report.json").exists())

    def test_regressor_both_architectures_train_and_reload(self):
        from src.prediction.daily_km_regressor import DailyKmRegressor
        from src.prediction.train_regressor import train_regressor

        for name in ("simple", "ultra_complex"):
            conf = dataclasses.replace(cfg.REGRESSOR_PRESETS[name], epochs=2, scalers=("standard", "robust"))
            reg, table, _ = train_regressor(self.df, conf, output_dir=self.root / name, data_source="synthetic")
            self.assertEqual(len(table), 2)
            self.assertTrue(table["val_rmse"].is_monotonic_increasing)  # sorted by VALIDATION rmse
            loaded = DailyKmRegressor.load(self.root / name)
            self.assertEqual(loaded.architecture, name)
            np.testing.assert_allclose(reg.predict(self.df), loaded.predict(self.df), rtol=1e-4, atol=1e-3)

    def test_full_pipeline_to_plan(self):
        from src.optimization.scheduler import TrainPlanScheduler
        from src.prediction.inference import InductionPipeline
        from src.prediction.train_classifier import train_classifier
        from src.prediction.train_regressor import train_regressor

        clf, _ = train_classifier(self.df, dataclasses.replace(cfg.ClassifierTrainConfig(), epochs=2), save=False)
        reg, _, _ = train_regressor(self.df, dataclasses.replace(cfg.REGRESSOR_PRESETS["simple"], epochs=2,
                                                                 scalers=("standard",)), save=False)
        preds = InductionPipeline(clf, reg).predict(self.df)
        for col in ("P_Service", "Predicted_Decision", "Predicted_Daily_Km_Raw", "Predicted_Daily_Km"):
            self.assertIn(col, preds.columns)
        self.assertGreaterEqual(preds["Predicted_Daily_Km"].min(), cfg.CLIP_MIN_KM)
        self.assertTrue(preds["P_Service"].between(0, 1).all())
        gated = InductionPipeline(clf, reg, gate_km_by_decision=True).predict(self.df)
        self.assertTrue((gated.loc[gated["Predicted_Decision"] == "Standby", "Predicted_Daily_Km"] == 0).all())
        plan = TrainPlanScheduler().build_plan(preds)
        self.assertEqual(plan.kpis["n_rows"], len(self.df))


if __name__ == "__main__":
    unittest.main()
