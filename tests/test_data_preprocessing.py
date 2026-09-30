"""Data layer: encodings, leakage guards, splits, train-only fitting."""
import unittest

import numpy as np
import pandas as pd

from src import config as cfg
from src.data.loader import RAW_COLUMNS, generate_synthetic_dataset, validate_columns
from src.data.preprocessing import (
    CLASSIFIER_TASK, REGRESSOR_TASK, build_preprocessor, build_task_frame, drop_unlabeled_rows,
    encode_decision, encode_features, mileage_category, split_frame,
)


class TestLeakageGuards(unittest.TestCase):
    """The notebook's classifier fed Daily_Km_Assigned (0 for Standby) to the model."""

    def test_classifier_features_exclude_targets_and_text(self):
        cols = set(CLASSIFIER_TASK.feature_columns)
        for banned in ("FinalDecision", "Daily_Km_Assigned", "DecisionReason", "Date", "Trainset"):
            self.assertNotIn(banned, cols)

    def test_regressor_features_exclude_targets_and_text(self):
        cols = set(REGRESSOR_TASK.feature_columns)
        for banned in ("FinalDecision", "Daily_Km_Assigned", "DecisionReason", "Date", "Trainset"):
            self.assertNotIn(banned, cols)
        self.assertEqual(len(cols), 11)  # notebook: "Feature set shape: (4500, 11)"


class TestEncodings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic_dataset(n_trainsets=6, n_days=20, seed=1)

    def test_schema(self):
        self.assertEqual(list(self.df.columns), RAW_COLUMNS)

    def test_balanced_mileage_label_is_not_nan(self):
        # the notebook's classifier map {"Underused":0,"Overused":1} turned "Balanced" into NaN
        enc = encode_features(self.df)
        self.assertTrue(self.df["Mileage_Label"].eq("Balanced").any())
        self.assertFalse(enc["Mileage_Label"].isna().any())
        self.assertEqual(set(enc["Mileage_Label"].unique()) - {0.0, 1.0, 2.0}, set())

    def test_cleaning_slot_missing_means_not_scheduled(self):
        enc = encode_features(self.df)
        expected = (self.df["Cleaning_Slot_Scheduled"] == "Scheduled").astype(float)
        pd.testing.assert_series_equal(enc["Cleaning_Slot_Scheduled"], expected, check_names=False)

    def test_booleans_become_floats(self):
        enc = encode_features(self.df)
        for col in cfg.BOOLEAN_COLUMNS:
            self.assertTrue(set(enc[col].unique()) <= {0.0, 1.0})

    def test_mileage_category_thresholds(self):
        s = pd.Series([0, 119_999, 120_000, 199_999, 200_000, np.nan])
        got = mileage_category(s).tolist()
        self.assertEqual(got[:5], [0.0, 0.0, 1.0, 1.0, 2.0])
        self.assertTrue(np.isnan(got[5]))

    def test_decision_encoding_and_unlabeled_rows(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "FinalDecision"] = "Maybe"
        bad.loc[bad.index[1], "FinalDecision"] = None
        kept = drop_unlabeled_rows(bad, CLASSIFIER_TASK)
        self.assertEqual(len(kept), len(bad) - 2)
        self.assertEqual(set(encode_decision(kept["FinalDecision"]).unique()), {0.0, 1.0})

    def test_missing_feature_column_raises(self):
        with self.assertRaises(ValueError):
            build_task_frame(self.df.drop(columns=["Bay_Position"]), CLASSIFIER_TASK)
        with self.assertRaises(ValueError):
            validate_columns(self.df, ["nope"])

    def test_inference_frame_needs_no_targets(self):
        feats = self.df.drop(columns=["FinalDecision", "Daily_Km_Assigned", "DecisionReason"])
        X, y = build_task_frame(feats, CLASSIFIER_TASK, require_target=False)
        self.assertIsNone(y)
        self.assertEqual(len(X), len(feats))


class TestSplitsAndFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic_dataset(n_trainsets=10, n_days=40, seed=2)

    def test_random_split_is_disjoint_and_complete(self):
        y = encode_decision(self.df["FinalDecision"])
        sp = split_frame(self.df, test_size=0.2, val_size=0.1, stratify=y)
        idx = [set(sp.train.index), set(sp.val.index), set(sp.test.index)]
        self.assertEqual(sum(len(i) for i in idx), len(self.df))
        self.assertFalse(idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
        rates = [encode_decision(p["FinalDecision"]).mean() for p in (sp.train, sp.val, sp.test)]
        self.assertLess(max(rates) - min(rates), 0.05)  # stratified

    def test_chronological_split_never_straddles_a_date(self):
        sp = split_frame(self.df, test_size=0.2, val_size=0.1, strategy="chronological")
        self.assertLess(sp.train["Date"].max(), sp.val["Date"].min())
        self.assertLess(sp.val["Date"].max(), sp.test["Date"].min())
        self.assertEqual(len(sp.train) + len(sp.val) + len(sp.test), len(self.df))

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            split_frame(self.df, test_size=0.2, val_size=0.1, strategy="bogus")

    def test_preprocessor_learns_from_train_only(self):
        sp = split_frame(self.df, test_size=0.2, val_size=0.1)
        X_tr, _ = build_task_frame(sp.train, CLASSIFIER_TASK)
        X_te, _ = build_task_frame(sp.test, CLASSIFIER_TASK)
        pre = build_preprocessor(CLASSIFIER_TASK.numeric_features, CLASSIFIER_TASK.categorical_features, "standard")
        Z_tr = pre.fit_transform(X_tr)
        n_num = len(CLASSIFIER_TASK.numeric_features)
        # scaler statistics equal the TRAIN statistics, not the full-data ones
        scaler = pre.named_transformers_["num"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, X_tr[list(CLASSIFIER_TASK.numeric_features)].fillna(
            X_tr[list(CLASSIFIER_TASK.numeric_features)].median()).mean().to_numpy(), rtol=1e-9)
        np.testing.assert_allclose(Z_tr[:, :n_num].mean(axis=0), 0, atol=1e-9)
        self.assertFalse(np.allclose(pre.transform(X_te)[:, :n_num].mean(axis=0), 0, atol=1e-9))

    def test_unseen_depot_does_not_crash(self):
        X, _ = build_task_frame(self.df, CLASSIFIER_TASK)
        pre = build_preprocessor(CLASSIFIER_TASK.numeric_features, CLASSIFIER_TASK.categorical_features).fit(X)
        X2 = X.head(3).copy()
        X2["Depot"] = "DepotZ"
        self.assertEqual(pre.transform(X2).shape[0], 3)

    def test_all_three_scalers_supported_and_unknown_rejected(self):
        X, _ = build_task_frame(self.df, REGRESSOR_TASK)
        for name in ("standard", "minmax", "robust"):
            Z = build_preprocessor(REGRESSOR_TASK.numeric_features, (), name).fit_transform(X)
            self.assertEqual(Z.shape[1], 11)
            self.assertFalse(np.isnan(Z).any())
        with self.assertRaises(ValueError):
            build_preprocessor(REGRESSOR_TASK.numeric_features, (), "bogus")


if __name__ == "__main__":
    unittest.main()
