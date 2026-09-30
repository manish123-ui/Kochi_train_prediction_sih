# Experiments

## A. Results recorded in the original notebook
These numbers are copied from the saved outputs of `00_original_colab_notebook.ipynb` (real data, not available to this repo). They were **not
reproduced** here. They came from the notebook's original code, which had the issues listed in `methodology.md` (test-set model selection, no
restored best weights in the simple model), so treat them as historical context; the refactored pipeline will give different numbers.

Setup in both runs: 4,500 rows; 11 features; random split 3,645 train / 405 val / 450 test; seed 42; scalers fitted on train.

### Daily_Km_Assigned regression, test set

| Model (notebook cell) | Scaler | RMSE | MAE | R^2 |
|---|---|---|---|---|
| Simple MLP 64-32-16, MSE, Adam 0.01, 500 epochs, no early stopping (cell 9) | standard | 183.23 | 146.13 | **0.3615** |
| | robust | 184.91 | 147.71 | 0.3497 |
| | minmax | 186.73 | 149.50 | 0.3369 |
| Ultra-complex MLP, 1,048,471 parameters, Huber, AdamW + OneCycle, early stopping (stopped at epochs 158-174) (cell 12) | robust | 196.45 | 123.86 | 0.2660 |
| | minmax | 198.91 | 121.02 | 0.2475 |
| | standard | 204.88 | 124.98 | 0.2016 |

Observations (from these numbers only):
* The 1 M-parameter model had a lower R^2 than the 3 k-parameter one (higher RMSE, lower MAE, consistent with Huber loss), so extra capacity did not help here.
* The three scalers differ by at most about 0.025 R^2 for the simple model, which is small next to the likely variance of a single split.
* One single-row check (cell 11): predicted 469.67 km, actual 478.

### FinalDecision classification
**No result exists.** Cell 8 raised `KeyError: 'cum_mileage'` and never trained. Any accuracy figure would be invented; none is given.

### `score` stacking experiment
**No result exists.** Cell 15 has no output. The data file is not visible in the notebook.

## B. How to produce comparable numbers with this repo
```bash
python -m src.main train-classifier --data data/raw/kochi_induction_dataset.csv
python -m src.main train-regressor  --data data/raw/kochi_induction_dataset.csv --model simple
python -m src.main train-regressor  --data data/raw/kochi_induction_dataset.csv --model ultra_complex
```
Each run writes `outputs/latest/*_report.json` (config, split sizes, validation and test metrics, library versions) and plots.
Report the classifier's accuracy next to `majority_baseline_accuracy`, which is in the same report.

## C. Suggested next experiments (not run)
1. Classifier: random vs `--split-strategy chronological`; drop rule-derived columns to see what remains learnable.
2. Regressor: train only on Service rows; add the classifier's decision/probability as an input; tree baselines (the `score` experiment code can be pointed at this target).
3. Multiple seeds / K-fold to put error bars on the 0.2-0.36 R^2 range.
4. Evaluate any future optimiser against the historical `FinalDecision` records.
