"""Scheduling interface: predictions -> constrained plan.

WHAT IS IMPLEMENTED
    ``build_plan``  turns model predictions into a plan (the classifier's Service/Standby
                    decision + the regressor's predicted km), checks any configured
                    constraints and computes descriptive KPIs. This is a *baseline
                    pass-through*: it does not reassign, repair or optimise anything.

WHAT IS NOT IMPLEMENTED (future work)
    ``optimize``    a real optimiser (e.g. MILP / CP-SAT / heuristic search) that picks the
                    Service set and km allocation subject to constraints and an Objective.
                    The original notebook contains no such algorithm and no operational rules,
                    so ``optimize`` raises NotImplementedError instead of pretending.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.evaluation.metrics import save_json
from src.optimization.constraints import STATUS_COLUMN, ConstraintSet
from src.optimization.objective import KM_COLUMN, Objective, plan_kpis

logger = logging.getLogger(__name__)

REQUIRED_PREDICTION_COLUMNS = ["Predicted_Decision", "Predicted_Daily_Km"]


@dataclass
class SchedulePlan:
    plan: pd.DataFrame
    violations: pd.DataFrame
    kpis: dict
    mode: str = "ml_passthrough_baseline"
    optimized: bool = False

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.plan.to_csv(directory / "train_plan.csv", index=False)
        self.violations.to_csv(directory / "constraint_violations.csv", index=False)
        save_json({"mode": self.mode, "optimized": self.optimized, "kpis": self.kpis,
                   "n_violations": int(len(self.violations))}, directory / "plan_summary.json")
        return directory


class TrainPlanScheduler:
    OPTIMIZATION_IMPLEMENTED = False

    def __init__(self, constraints: ConstraintSet | None = None):
        self.constraints = constraints or ConstraintSet()

    def build_plan(self, predictions: pd.DataFrame) -> SchedulePlan:
        """Baseline plan straight from the models' outputs (see module docstring)."""
        missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in predictions.columns]
        if missing:
            raise ValueError(f"predictions is missing {missing}; run InductionPipeline.predict first")
        if len(self.constraints) == 0:
            logger.warning("No operational constraints configured - the plan is NOT checked against any rules.")
        plan = predictions.copy()
        plan[STATUS_COLUMN] = plan["Predicted_Decision"]
        plan[KM_COLUMN] = np.where(plan[STATUS_COLUMN] == cfg.SERVICE_LABEL,
                                   plan["Predicted_Daily_Km"].astype(float), 0.0)
        sort_cols = [c for c in (cfg.DATE_COLUMN, cfg.TRAINSET_COLUMN) if c in plan.columns]
        if sort_cols:
            plan = plan.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        return SchedulePlan(plan=plan, violations=self.constraints.check(plan), kpis=plan_kpis(plan))

    def optimize(self, predictions: pd.DataFrame, objective: Objective | None = None) -> SchedulePlan:
        raise NotImplementedError(
            "Scheduling optimisation is future work: the original notebook contains no optimisation "
            "algorithm or operational rules. Use build_plan() for the ML pass-through baseline."
        )
