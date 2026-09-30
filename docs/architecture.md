# Architecture

```
Data                     data/raw/*.csv (private) | data/sample/*.csv (synthetic)        src/data/loader.py
 ↓
Preprocessing            stateless encodings -> split RAW rows -> fit scaler/one-hot     src/data/preprocessing.py
 ↓                       on TRAIN only
 ├─► FinalDecision classifier  (MLP, logit -> P(Service))                                src/prediction/induction_classifier.py
 │        ↓ Service / Standby
 └─► Daily_Km regressor        (MLP "simple" or "ultra_complex")                         src/prediction/daily_km_regressor.py
          ↓ Predicted daily km
   InductionPipeline.predict  -> P_Service, Predicted_Decision, Predicted_Daily_Km       src/prediction/inference.py
 ↓
Operational constraints  generic, user-configured checks; NONE enabled by default        src/optimization/constraints.py
 ↓
Scheduling / optimisation  build_plan = ML pass-through baseline (implemented)           src/optimization/scheduler.py
                           optimize   = NotImplementedError (future work)
 ↓
Final train plan         train_plan.csv, constraint_violations.csv, plan_summary.json
```

## What is real and what is a scaffold
| Stage | Status |
|---|---|
| Data, preprocessing, classifier, regressor, inference | Implemented; extracted from the notebook |
| Constraints | Framework only. The notebook contains no operational rules, so none are defined |
| Scheduler `build_plan` | Baseline: Service/Standby and km are taken from the models; constraints are *checked*, never *enforced* |
| Scheduler `optimize`, `Objective` | Not implemented. The notebook has no optimisation algorithm |

## Module map
| Module | Responsibility |
|---|---|
| `config.py` | paths, column lists, category maps, training configs and presets (single source of truth) |
| `data/loader.py` | read CSV, schema validation, synthetic sample generator |
| `data/preprocessing.py` | `TaskSpec`s, deterministic encodings, `split_frame`, `build_preprocessor` (ColumnTransformer) |
| `prediction/common.py` | seeding, loaders, one shared training loop (`fit_model`), model bundle save/load |
| `prediction/induction_classifier.py` | `InductionMLP`, `InductionClassifier` (raw frame in -> probabilities out) |
| `prediction/daily_km_regressor.py` | `SimpleRegressionMLP`, `UltraComplexRegressionMLP`, `DailyKmRegressor` |
| `prediction/train_*.py` | training entry points (functions + CLI) that write reports/plots |
| `prediction/inference.py` | `InductionPipeline`: both models -> prediction table |
| `optimization/*.py` | constraints, plan KPIs + `Objective` interface, scheduler |
| `evaluation/*.py` | metrics (classification, regression), plots |
| `experiments/score_stacking.py` | the separate `score` experiment (not part of the pipeline) |
| `main.py` | CLI: `generate-sample`, `train-*`, `predict`, `schedule`, `run-all` |

## Design decisions
* **One config, no magic values.** Column lists, maps and hyper-parameters are in `config.py`; model classes and training code take them as arguments.
* **Inference uses the same code path as training.** A trained model is bundled with its fitted preprocessor; `predict` takes a *raw* dataframe and re-applies the encodings, so there is no train/serve skew.
* **Bundles** (`models/<name>/`): `model.pt` (tensors + architecture name/kwargs, loaded with `weights_only=True`), `preprocessor.joblib`, `metadata.json` (config, split sizes, metrics, library versions).
* **Split before fitting.** Only stateless encodings run before the split; imputation medians, scaler statistics and one-hot categories are learned from the training rows.
* **The scheduler never invents behaviour.** Anything not defined by the notebook is an explicit, documented extension point.

## Extending to a real optimiser
1. Define domain rules as `Constraint` subclasses (or spec dicts) in `optimization/constraints.py`, with values supplied by the operator.
2. Implement an `Objective` (e.g. mileage balance across trainsets, number of trains available).
3. Implement `TrainPlanScheduler.optimize` (e.g. MILP with OR-Tools or PuLP, CP-SAT, or local search), using `P_Service` and `Predicted_Daily_Km` as inputs.
4. Validate against historical `FinalDecision` records before using it operationally.
