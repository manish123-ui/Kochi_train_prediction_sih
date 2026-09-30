"""Operational constraints: a checking interface, with NO Kochi Metro rules hard-coded.

The original notebook contains no operational rules or scheduling logic. Rather than
inventing any, this module provides generic, parameterised constraint types that a domain
expert can configure. With none configured (the default) nothing is enforced and the
scheduler says so.

Constraints are described as dictionaries (in ``config.OPERATIONAL_CONSTRAINT_SPECS`` or a
JSON file passed with ``--constraints``):

    {"type": "service_column_in",    "column": "<col>", "allowed": [<values>]}
    {"type": "service_column_range", "column": "<col>", "min_value": <x>, "max_value": <y>}
    {"type": "service_count",        "min_count": <n>,  "max_count": <m>}   # per Date

The values are supplied by the user, never by this code base. Every constraint only looks at
trains whose ``Assigned_Status`` is "Service" and reports violations. It never repairs a plan
(repairing / optimising is future work, see scheduler.py).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config as cfg

STATUS_COLUMN = "Assigned_Status"


@dataclass(frozen=True)
class Violation:
    constraint: str
    trainset: str | None
    date: str | None
    message: str


def _ident(row: pd.Series) -> tuple[str | None, str | None]:
    t = row.get(cfg.TRAINSET_COLUMN)
    d = row.get(cfg.DATE_COLUMN)
    return (None if t is None else str(t)), (None if d is None else str(d))


def _service_rows(plan: pd.DataFrame) -> pd.DataFrame:
    if STATUS_COLUMN not in plan.columns:
        raise ValueError(f"Plan is missing the {STATUS_COLUMN!r} column")
    return plan[plan[STATUS_COLUMN] == cfg.SERVICE_LABEL]


def _require_column(plan: pd.DataFrame, column: str, constraint: str) -> None:
    if column not in plan.columns:
        raise ValueError(f"Constraint {constraint!r} needs column {column!r}, which the plan does not have")


class Constraint(ABC):
    name: str

    @abstractmethod
    def check(self, plan: pd.DataFrame) -> list[Violation]:
        """Return the violations found in ``plan`` (empty list if satisfied)."""


class ServiceColumnInConstraint(Constraint):
    """Trains in Service must have ``column`` in ``allowed``."""

    def __init__(self, column: str, allowed: list, name: str | None = None):
        self.column, self.allowed = column, list(allowed)
        self.name = name or f"service_column_in[{column}]"

    def check(self, plan):
        _require_column(plan, self.column, self.name)
        bad = _service_rows(plan)
        bad = bad[~bad[self.column].isin(self.allowed)]
        return [Violation(self.name, *_ident(r), f"{self.column}={r[self.column]!r} not in {self.allowed}")
                for _, r in bad.iterrows()]


class ServiceColumnRangeConstraint(Constraint):
    """Trains in Service must have ``column`` within [min_value, max_value] (either bound optional)."""

    def __init__(self, column: str, min_value: float | None = None, max_value: float | None = None,
                 name: str | None = None):
        if min_value is None and max_value is None:
            raise ValueError("service_column_range needs min_value and/or max_value")
        self.column, self.min_value, self.max_value = column, min_value, max_value
        self.name = name or f"service_column_range[{column}]"

    def check(self, plan):
        _require_column(plan, self.column, self.name)
        rows = _service_rows(plan)
        values = pd.to_numeric(rows[self.column], errors="coerce")
        mask = pd.Series(False, index=rows.index)
        if self.min_value is not None:
            mask |= values < self.min_value
        if self.max_value is not None:
            mask |= values > self.max_value
        return [Violation(self.name, *_ident(r), f"{self.column}={r[self.column]} outside "
                          f"[{self.min_value}, {self.max_value}]") for _, r in rows[mask].iterrows()]


class ServiceCountConstraint(Constraint):
    """Number of Service trains per Date (or overall if there is no Date column) within bounds."""

    def __init__(self, min_count: int | None = None, max_count: int | None = None, name: str | None = None):
        if min_count is None and max_count is None:
            raise ValueError("service_count needs min_count and/or max_count")
        self.min_count, self.max_count = min_count, max_count
        self.name = name or "service_count"

    def check(self, plan):
        rows = _service_rows(plan)
        if cfg.DATE_COLUMN in plan.columns:
            counts = rows.groupby(cfg.DATE_COLUMN).size().reindex(plan[cfg.DATE_COLUMN].unique(), fill_value=0)
        else:
            counts = pd.Series({None: len(rows)})
        out = []
        for date, n in counts.items():
            if (self.min_count is not None and n < self.min_count) or (self.max_count is not None and n > self.max_count):
                out.append(Violation(self.name, None, None if date is None else str(date),
                                     f"{n} trains in Service, allowed [{self.min_count}, {self.max_count}]"))
        return out


CONSTRAINT_TYPES = {
    "service_column_in": ServiceColumnInConstraint,
    "service_column_range": ServiceColumnRangeConstraint,
    "service_count": ServiceCountConstraint,
}


class ConstraintSet:
    """A collection of constraints checked together. May be empty."""

    def __init__(self, constraints: list[Constraint] | None = None):
        self.constraints = list(constraints or [])

    def __len__(self) -> int:
        return len(self.constraints)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.constraints]

    def check(self, plan: pd.DataFrame) -> pd.DataFrame:
        violations = [v for c in self.constraints for v in c.check(plan)]
        return pd.DataFrame([v.__dict__ for v in violations],
                            columns=["constraint", "trainset", "date", "message"])


def constraints_from_specs(specs: list[dict] | None) -> ConstraintSet:
    """Build a ConstraintSet from spec dicts (see module docstring)."""
    built = []
    for spec in specs or []:
        spec = dict(spec)
        kind = spec.pop("type", None)
        if kind not in CONSTRAINT_TYPES:
            raise ValueError(f"Unknown constraint type {kind!r}; choose from {sorted(CONSTRAINT_TYPES)}")
        built.append(CONSTRAINT_TYPES[kind](**spec))
    return ConstraintSet(built)


def load_constraint_specs(path: str | Path) -> list[dict]:
    specs = json.loads(Path(path).read_text())
    if not isinstance(specs, list):
        raise ValueError("Constraint file must contain a JSON list of spec objects")
    return specs
