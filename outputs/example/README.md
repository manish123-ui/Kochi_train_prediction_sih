# Example outputs (SYNTHETIC)

Everything here was produced from the **synthetic** sample data, never from the real Kochi Metro data.

| File | What it is |
|---|---|
| `train_plan.csv` | Output of `TrainPlanScheduler.build_plan`. Command: `python -m src.main schedule --use-sample --from-labels`. `--from-labels` feeds the file's own synthetic labels in as if they were model predictions, to show the plan format. **No model was trained or evaluated for this file.** |
| `plan_summary.json` | KPIs for that plan (counts, planned km). `optimized: false`. |
| `constraint_violations.csv` | Header only: no operational constraints are configured by default, so nothing is checked. |
| `synthetic_km_by_decision.png` | Histogram of synthetic `Daily_Km_Assigned` per decision (0 for Standby by construction). |

Outputs of model training (`classifier_report.json`, `regressor_comparison.csv`, plots, `predictions.csv`) are
not included because PyTorch was not available where this repository was assembled, and numbers are not
invented. Running `python -m src.main run-all --use-sample` creates them in `outputs/latest/`.
