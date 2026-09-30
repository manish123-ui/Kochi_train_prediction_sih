# Kochi Metro Train Scheduling

Machine-learning pipeline for **train induction** at Kochi Metro, refactored from a single Google Colab notebook into a
reproducible Python project:

* a **classifier** that predicts each trainset's `FinalDecision` (**Service** or **Standby**),
* a **regressor** that predicts the daily kilometres assigned (`Daily_Km_Assigned`),
* a **constraint and scheduling scaffold** that turns those predictions into a train plan.

> **Status: research prototype.** The prediction models come from the notebook. The scheduling optimiser does **not** exist yet: the
> original notebook contained none, and this repository does not pretend otherwise (see [Limitations](#limitations) and
> [Implemented vs future work](#implemented-vs-future-work)). No operational rules are hard-coded. The real dataset is private and
> not included; the bundled sample data is **synthetic**.

## Pipeline

```
Data ─► Preprocessing ─► FinalDecision classifier ─► Service / Standby ─┐
                     └─► Daily_Km regressor ──────► Predicted daily km ─┤
                                                                        ▼
                                          Operational constraints (user-defined, none by default)
                                                                        ▼
                                      Scheduling  (baseline pass-through ✔ | optimiser ✘ future work)
                                                                        ▼
                                                              Final train plan
```

Details: [`docs/architecture.md`](docs/architecture.md).

## Repository layout

```
Kochi-Metro-Train-Scheduling/
├── README.md
├── requirements.txt
├── .gitignore
├── data/            README.md, sample/ (synthetic CSV); data/raw/ is git-ignored (put the private CSV there)
├── notebooks/       00 original (unchanged) · 01 EDA · 02 classifier · 03 regression · 04 score experiment
├── src/
│   ├── config.py  main.py
│   ├── data/          loader.py  preprocessing.py
│   ├── prediction/    induction_classifier.py  daily_km_regressor.py  train_classifier.py  train_regressor.py  inference.py  common.py
│   ├── optimization/  constraints.py  objective.py  scheduler.py
│   ├── evaluation/    metrics.py  visualization.py
│   └── experiments/   score_stacking.py
├── tests/           unit + end-to-end tests
├── models/          README.md (trained bundles are git-ignored)
├── outputs/example/ synthetic example plan
└── docs/            architecture.md  methodology.md  data_dictionary.md  experiments.md
```

## Models

| Model | Target | Architecture (from the notebook) | Loss / optimiser |
|---|---|---|---|
| **Classifier** (`InductionMLP`) | `FinalDecision` (Service=1) | MLP 10-100-100-100, LeakyReLU(0.2) / Tanh, 1 logit | BCE-with-logits, Adam 1e-3, batch 16, 20 epochs |
| **Regressor `simple`** | `Daily_Km_Assigned` | MLP 64-32-16, ReLU, Dropout 0.2 | MSE, Adam 1e-2, ReduceLROnPlateau, 500 epochs, early stopping |
| **Regressor `ultra_complex`** | `Daily_Km_Assigned` | 3 parallel paths + combiner, BatchNorm/GELU/Dropout, 1,048,471 parameters | Huber, AdamW 1e-3, OneCycle, 1000 epochs, early stopping |
| **Score experiment** (separate) | `score` in a *different* CSV | LightGBM (RandomForest fallback) 5-fold OOF -> stacked MLP 256-128 -> 50/50 ensemble | see `src/experiments/score_stacking.py` |

## Dataset and features

The induction data has 4,500 rows x 19 columns (shape printed by the notebook). Features include fitness-certificate flags,
job-card counts, branding hours, cumulative mileage and mileage label, cleaning status, bay position and stabling difficulty.
Full table with encodings and the exact feature lists: [`docs/data_dictionary.md`](docs/data_dictionary.md).
`Daily_Km_Assigned` is never a classifier feature (it leaked the label in the notebook).

**Data privacy:** nothing private is committed. Put your CSV at `data/raw/kochi_induction_dataset.csv` (git-ignored) or pass `--data`.

## Installation

```bash
git clone <your-repo-url> Kochi-Metro-Train-Scheduling
cd Kochi-Metro-Train-Scheduling
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # for GPU builds install torch from pytorch.org first
```

## Quick start (synthetic data)

```bash
python -m src.main generate-sample            # writes data/sample/synthetic_kochi_induction_sample.csv
python -m src.main run-all --use-sample --epochs 5
```
`run-all` trains both models, predicts, builds a plan and writes everything to `outputs/latest/`. On synthetic data the metrics are meaningless.

## Training on real data

```bash
python -m src.main train-classifier --data data/raw/kochi_induction_dataset.csv
python -m src.main train-regressor  --data data/raw/kochi_induction_dataset.csv --model simple          # or ultra_complex
```
Useful options: `--epochs N`, `--scaler standard|minmax|robust` (classifier), `--scalers ...` (regressor),
`--split-strategy random|chronological`, `--device cpu|cuda`, `--report-dir`, `--output-dir`. Try `python -m src.main train-regressor --help`.

Artifacts go to `models/` (weights, fitted preprocessor, metadata) and reports to `outputs/latest/` (`*_report.json`, comparison CSV, plots).
The regressor's scaler is chosen on **validation** RMSE; test metrics are computed after and not used for selection.

## Inference and planning

```bash
python -m src.main predict  --input new_day.csv --output outputs/latest/predictions.csv
python -m src.main schedule --input outputs/latest/predictions.csv [--constraints my_constraints.json]
```
`new_day.csv` needs the feature columns only (no targets). Output columns: `P_Service`, `Predicted_Decision`,
`Predicted_Daily_Km_Raw`, `Predicted_Daily_Km` (clipped at 0). `--gate-km` forces 0 km for predicted Standby; it is an **unverified assumption** and off by default.

Python API:

```python
from src.prediction.inference import InductionPipeline
from src.optimization.scheduler import TrainPlanScheduler
from src.optimization.constraints import constraints_from_specs

preds = InductionPipeline.load().predict(df)                    # df = raw rows
plan = TrainPlanScheduler(constraints_from_specs([])).build_plan(preds)
plan.plan, plan.violations, plan.kpis
```

### Constraints
No constraint ships enabled, because the notebook defines none. You can describe your own in JSON; the values are yours:

```json
[
  {"type": "service_column_in",    "column": "<column>", "allowed": ["<value>"]},
  {"type": "service_column_range", "column": "Planned_Daily_Km", "max_value": 0},
  {"type": "service_count",        "min_count": 0, "max_count": 0}
]
```
(`0` and `<...>` are placeholders, not recommendations.) Constraints report violations; they do not repair the plan.

## Score experiment and notebooks

```bash
python -m src.experiments.score_stacking --data path/to/scored.csv --target score          # add --no-mlp for trees only
jupyter notebook notebooks/
```
Notebooks 01-04 are thin wrappers over `src/`. They use your real CSV when present, otherwise the synthetic sample, and are saved **without outputs**.

## Tests

```bash
python -m unittest discover -s tests -v        # also works with pytest
```

## Verification status

What was run while assembling this repository (Python 3.12, pandas 3.0, scikit-learn 1.8, **no PyTorch, no LightGBM, no network** in that environment):

| Check | Result |
|---|---|
| All modules compile; static check for unused imports / undefined names | pass |
| Unit tests: data layer, leakage guards, splits, constraints, scheduler, score preprocessing, tree stage (RandomForest) | **29 pass** |
| PyTorch tests (architectures, training, save/load parity, inference, full pipeline) | **written, skipped (5) - not executed** |
| 1M-parameter count of `ultra_complex` = 1,048,471 (notebook value), via an `nn` stub | matches; stub only counts parameters |
| `generate-sample`, `schedule`, notebooks 01 and 04 (tree-only) executed on synthetic data | pass |
| LightGBM code path (callbacks API) | **not executed** |

**So the PyTorch training/inference code and the LightGBM path have not been run.** Please run `python -m unittest discover -s tests -v` after
installing requirements; the PyTorch tests will then execute and any integration bug will show up there. No model metrics are reported anywhere in this repository except those recorded in the original notebook (labelled as such in [`docs/experiments.md`](docs/experiments.md)).

## Limitations

* **No optimiser.** `TrainPlanScheduler.build_plan` is a pass-through of model outputs; `optimize()` raises `NotImplementedError`. Plans are not guaranteed feasible.
* **No operational rules** are encoded. The repository does not know what makes a train eligible for service.
* **The two models are independent.** The regressor does not use the classifier's output and trains on Standby rows (0 km) too.
* **Weak recorded regression results:** best R^2 in the original notebook was 0.36 (test), with a model selected on the test set.
* **No classifier result exists**: the notebook's classifier cell crashed. The label may be largely rule-derived; check before trusting any accuracy.
* **Random row splits** may let the same trainset on adjacent days appear in both train and test; a chronological option is provided.
* **Private data is required** for real results; the synthetic sample only exercises the code.
* Column meanings in the data dictionary are inferred from names.

## Implemented vs future work

| Implemented | Future work |
|---|---|
| Notebook analysis and cleanup log (`docs/methodology.md`) | Scheduling optimiser (MILP / CP-SAT / heuristics) |
| Leak-free preprocessing, train-only fitting, split strategies | Domain-validated operational constraints |
| FinalDecision classifier: training, evaluation, save/load | Plan objective (`Objective` interface is empty) |
| Daily_Km regressor: two architectures, scaler comparison on validation | Couple classifier and regressor (e.g. train km on Service rows only) |
| Inference pipeline (raw frame -> decision + km) | Calibration and uncertainty for `P_Service` and km |
| Constraint framework (spec-driven, reports violations) | Backtesting against historical decisions |
| Plan builder + KPIs (baseline pass-through) | Hyper-parameter search, multiple seeds / K-fold |
| Score / stacking experiment, leakage-fixed | Run the score experiment on the real file |
| Synthetic data generator, CLI, tests, notebooks | CI, packaging (`pyproject.toml`), experiment tracking, a LICENSE of your choice |
