"""Constraints, KPIs and the scheduling scaffold."""
import unittest

import pandas as pd

from src.optimization.constraints import (
    ConstraintSet, ServiceColumnInConstraint, constraints_from_specs,
)
from src.optimization.scheduler import TrainPlanScheduler


def predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["d1", "d1", "d1", "d2", "d2"],
        "Trainset": ["A", "B", "C", "A", "B"],
        "Flag": [0, 1, 0, 0, 0],
        "Predicted_Decision": ["Service", "Service", "Standby", "Service", "Standby"],
        "Predicted_Daily_Km": [500.0, 800.0, 450.0, 300.0, 100.0],
    })


class TestScheduler(unittest.TestCase):
    def test_baseline_plan_passthrough(self):
        plan = TrainPlanScheduler().build_plan(predictions())
        self.assertFalse(plan.optimized)
        self.assertEqual(plan.mode, "ml_passthrough_baseline")
        self.assertEqual(plan.plan["Assigned_Status"].tolist(), ["Service", "Service", "Standby", "Service", "Standby"])
        # standby trains carry 0 planned km in the plan, whatever the regressor said
        self.assertEqual(plan.plan["Planned_Daily_Km"].tolist(), [500.0, 800.0, 0.0, 300.0, 0.0])
        self.assertEqual(plan.kpis["n_service"], 3)
        self.assertEqual(plan.kpis["total_planned_km"], 1600.0)
        self.assertEqual(len(plan.violations), 0)  # nothing configured -> nothing enforced

    def test_optimize_is_explicitly_not_implemented(self):
        self.assertFalse(TrainPlanScheduler.OPTIMIZATION_IMPLEMENTED)
        with self.assertRaises(NotImplementedError):
            TrainPlanScheduler().optimize(predictions())

    def test_requires_prediction_columns(self):
        with self.assertRaises(ValueError):
            TrainPlanScheduler().build_plan(predictions().drop(columns=["Predicted_Decision"]))


class TestConstraints(unittest.TestCase):
    def test_spec_driven_constraints_report_violations(self):
        cs = constraints_from_specs([
            {"type": "service_column_in", "column": "Flag", "allowed": [0]},
            {"type": "service_column_range", "column": "Planned_Daily_Km", "max_value": 700},
            {"type": "service_count", "max_count": 1},
        ])
        v = TrainPlanScheduler(cs).build_plan(predictions()).violations
        by = v.groupby("constraint").size().to_dict()
        self.assertEqual(by["service_column_in[Flag]"], 1)               # B has Flag=1 and is in Service
        self.assertEqual(by["service_column_range[Planned_Daily_Km]"], 1)  # B has 800 km
        self.assertEqual(by["service_count"], 1)                         # d1 has 2 Service trains (> 1)

    def test_constraints_ignore_standby_trains(self):
        cs = ConstraintSet([ServiceColumnInConstraint("Flag", [0])])
        p = predictions()
        p.loc[1, "Predicted_Decision"] = "Standby"  # the only offender goes to standby
        self.assertEqual(len(TrainPlanScheduler(cs).build_plan(p).violations), 0)

    def test_bad_specs_are_rejected(self):
        with self.assertRaises(ValueError):
            constraints_from_specs([{"type": "made_up"}])
        with self.assertRaises(ValueError):
            constraints_from_specs([{"type": "service_count"}])  # needs a bound
        with self.assertRaises(ValueError):
            TrainPlanScheduler(constraints_from_specs(
                [{"type": "service_column_in", "column": "Missing", "allowed": [1]}])).build_plan(predictions())

    def test_no_default_operational_rules(self):
        from src import config as cfg
        self.assertEqual(cfg.OPERATIONAL_CONSTRAINT_SPECS, [])


if __name__ == "__main__":
    unittest.main()
