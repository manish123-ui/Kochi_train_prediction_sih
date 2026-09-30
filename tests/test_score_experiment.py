"""Score-stacking experiment: preprocessing leakage guards and the tree stage (no PyTorch needed)."""
import unittest

import numpy as np
import pandas as pd

from src.experiments.score_stacking import ScoreExperimentConfig, ScorePreprocessor, run_score_experiment


def make_frame(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "x_num": rng.normal(50, 10, n),
        "x_outlier": np.where(rng.random(n) < 0.01, 1e6, rng.normal(0, 1, n)),
        "x_text_num": rng.normal(5, 2, n).round(2).astype(str),
        "when": pd.date_range("2025-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M"),
        "cat": rng.choice(["a", "b", "c"], n),
        "small_int": rng.integers(0, 12, n),
        "const": 7,
    })
    df.loc[rng.choice(n, 25, replace=False), "x_num"] = np.nan
    df["score"] = 0.3 * df["x_num"].fillna(50) + 4 * df["small_int"] + df["cat"].map({"a": 0, "b": 5, "c": 10}) + rng.normal(0, 2, n)
    return df


class TestScorePreprocessor(unittest.TestCase):
    def setUp(self):
        self.df = make_frame()
        self.train, self.test = self.df.iloc[:350], self.df.iloc[350:].copy()
        self.pre = ScorePreprocessor("score").fit(self.train)

    def test_roles(self):
        names = self.pre.feature_names_
        for dropped in ("id", "const", "score"):
            self.assertNotIn(dropped, names)
        self.assertIn("cat_code", names)
        self.assertIn("x_text_num", names)
        self.assertEqual(self.pre.plan_["small_int"]["kind"], "numeric")

    def test_statistics_come_from_train_only(self):
        self.assertAlmostEqual(self.pre.plan_["x_num"]["median"], float(self.train["x_num"].median()))
        t = self.test.copy()
        t.iloc[0, t.columns.get_loc("x_num")] = np.nan
        filled = self.pre.transform(t)[0, self.pre.feature_names_.index("x_num")]
        self.assertAlmostEqual(filled, float(self.train["x_num"].median()))

    def test_winsorisation_uses_train_bounds(self):
        _, hi = self.pre.plan_["x_outlier"]["bounds"]
        t = self.test.copy()
        t["x_outlier"] = 1e12
        self.assertLessEqual(self.pre.transform(t)[:, self.pre.feature_names_.index("x_outlier")].max(), hi + 1e-9)

    def test_unseen_category_maps_to_minus_one(self):
        t = self.test.copy()
        t["cat"] = "never-seen"
        self.assertEqual(set(self.pre.transform(t)[:, self.pre.feature_names_.index("cat_code")]), {-1.0})


class TestTreeStage(unittest.TestCase):
    def test_tree_only_run_learns_planted_signal(self):
        report = run_score_experiment(make_frame(), ScoreExperimentConfig(use_mlp=False, n_folds=3, rf_trees=40, lgb_rounds=300))
        self.assertIn(report["tree_backend"], {"lightgbm", "random_forest"})
        self.assertGreater(report["metrics"]["tree_test"]["r2"], 0.5)
        self.assertNotIn("mlp_test", report["metrics"])

    def test_missing_target_raises(self):
        with self.assertRaises(ValueError):
            run_score_experiment(make_frame().drop(columns=["score"]), ScoreExperimentConfig(use_mlp=False))


if __name__ == "__main__":
    unittest.main()
