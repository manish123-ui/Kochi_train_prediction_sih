"""Plan KPIs (implemented) and the objective interface (future work).

The notebook defines no scheduling objective. :func:`plan_kpis` therefore only *describes* a
plan with definitional statistics; it does not say what a good plan is. :class:`Objective` is
the interface a future optimiser will minimise/maximise, and has no implementation yet.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src import config as cfg
from src.optimization.constraints import STATUS_COLUMN

KM_COLUMN = "Planned_Daily_Km"


def plan_kpis(plan: pd.DataFrame) -> dict:
    """Descriptive statistics of a plan (counts, planned km). Not an optimisation objective."""
    is_service = plan[STATUS_COLUMN] == cfg.SERVICE_LABEL
    km_service = plan.loc[is_service, KM_COLUMN].astype(float)
    kpis = {
        "n_rows": int(len(plan)),
        "n_service": int(is_service.sum()),
        "n_standby": int((~is_service).sum()),
        "service_fraction": float(is_service.mean()) if len(plan) else None,
        "total_planned_km": float(plan[KM_COLUMN].astype(float).sum()),
        "mean_km_per_service_train": float(km_service.mean()) if len(km_service) else None,
        "std_km_per_service_train": float(km_service.std(ddof=0)) if len(km_service) else None,
    }
    if cfg.DATE_COLUMN in plan.columns:
        per_date = plan[is_service].groupby(cfg.DATE_COLUMN).size().reindex(plan[cfg.DATE_COLUMN].unique(), fill_value=0)
        kpis.update({
            "n_dates": int(len(per_date)),
            "service_trains_per_date_min": int(per_date.min()),
            "service_trains_per_date_mean": float(per_date.mean()),
            "service_trains_per_date_max": int(per_date.max()),
        })
    return kpis


class Objective(ABC):
    """FUTURE WORK: what a "better" plan means (e.g. mileage balancing, availability).

    Nothing in the original notebook defines this, so no concrete objective ships with the repo.
    Higher score = better plan, by convention.
    """

    name: str = "objective"

    @abstractmethod
    def score(self, plan: pd.DataFrame) -> float:
        """Return a scalar score for ``plan``."""
