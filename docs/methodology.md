# Methodology and change log

This document records (1) what the original notebook contained, (2) every important change made while turning it into a
repository, and (3) what could not be verified. Cell numbers refer to `notebooks/00_original_colab_notebook.ipynb`
(0-based, 17 cells), which is kept byte-for-byte unchanged.

## 1. Notebook inventory

| Cell | Content | Verdict | Where it went |
|---|---|---|---|
| 0 | TensorFlow/Keras imports (`Dense`, `Sequential`, `L2`, `EarlyStopping`) | unused | dropped |
| 1, 13 | empty | - | dropped |
| 2 | `make_regression` + `nn.Linear` toy (lr=1.0; loss dips to about 1,744 by step 4, then climbs to 387,433 by step 99, i.e. diverges). Uses `torch`/`np` imported in other cells | toy | dropped |
| 3 | First Kochi preprocessing. Its printed output predates its code (shows `Depot` still a string, `Cum_Mileage` capitalised, no `Mileage_Category`); the code reads `cum_mileage` | early draft of cell 8 | folded into `preprocessing.py` |
| 4 | `import torch` | trivial | dropped |
| 5 | class `Models`: not an `nn.Module`, `forward` nested inside `__init__`, no `super().__init__()` | incomplete | architecture recovered in `InductionMLP` (same layer stack as cell 8) |
| 6 | class `spilt` holding one int | empty | dropped |
| 7 | 187 lines: `make_classification` toy, `SimpleMLP`, two split methods | toy / generic tutorial | dropped |
| 8 | **Kochi FinalDecision classifier** (preprocess, scale, MLP, BCE, 20 epochs). **Crashed**: `KeyError: 'cum_mileage'` | real, never completed | `train_classifier.py`, `induction_classifier.py` |
| 9 | **Kochi Daily_Km regression v1**: 11 features, 3 scalers, MLP 64-32-16. Ran; results recorded | real | `train_regressor.py`, `SimpleRegressionMLP` |
| 10 | Inference snippet: defines a different network (`Models`, `input_size=10`), `pd.DataFrame([...])` placeholder. **Crashed** (`state_dict` mismatch) | broken | replaced by `inference.py` |
| 11 | Inference on one test row via `load_trained_model('standard')`. Ran (pred 469.67 vs actual 478) | real | covered by `DailyKmRegressor.predict` |
| 12 | **Kochi Daily_Km regression v2**: same pipeline copy-pasted with `ComplexRegressionModel` (unused) and `UltraComplexRegressionModel` (used). Ran; results recorded | real, duplicate pipeline | merged into `train_regressor.py` (`--model ultra_complex`) |
| 14 | `pdi.head(5)` | display | dropped |
| 15 | **`score` experiment**: LightGBM (RF fallback) K-fold OOF -> stacked PyTorch MLP -> 50/50 ensemble. Different CSV, no outputs, never ran | real code, unrelated to the two tasks | `experiments/score_stacking.py`, notebook 04 |
| 16 | stray markdown "text of linear reg in pytorch" | noise | dropped |

**What the notebook does not contain:** any scheduling or optimisation algorithm, any operational constraint, any link between the
classifier and regressor, any classifier result, any result for the `score` experiment. The repository therefore provides a scaffold
for those parts (see `architecture.md`) and no invented logic.

## 2. Bugs found and fixed

### Data leakage and modelling errors
1. **Target leakage in the classifier (cell 8).** `Daily_Km_Assigned` was left in `X` when predicting `FinalDecision`; the one Standby row the notebook printed has 0 km, so the column can give away the label. It is now excluded (`CLASSIFIER_NUMERIC_FEATURES`; covered by a unit test).
2. **Scaler fitted before the split (cell 8).** `StandardScaler.fit_transform` ran on all rows, then `train_test_split`. Now split first, fit on train only (`ColumnTransformer` in `build_preprocessor`).
3. **Test set used for model selection (cells 9, 12).** The "best" scaler was picked by test R^2. Now picked by validation RMSE; test metrics are reported afterwards.
4. **`Balanced` mileage label became NaN in the classifier (cell 8).** Map `{"Underused":0,"Overused":1}` has no `Balanced`, whereas cells 9/12 map three levels. One 3-level map now serves both tasks. (The printed output of cell 3 shows float `Mileage_Label`, consistent with NaNs.)
5. **`Depot` left as a string (cell 8)** would crash the scaler. It is now one-hot encoded inside the fitted preprocessor. Whether the original author meant to drop it (as the regressor does) is unknown.
6. **Silent `.fillna(0)` after category maps (cell 9).** Unknown categories were quietly turned into 0. Now unknown non-null values become NaN with a logged warning and are median-imputed by the fitted imputer.
7. **No validation split and no stratification in the classifier.** Added a validation split (10 % of non-test rows) and stratification (configurable).
8. **Regressor "best weights" were never restored (cell 9).** Early stopping was inside a `'''` string, so final-epoch weights were used. Early stopping is now a config option (`patience`, default 20 for the simple model) and the best-validation weights are restored.
9. **Shallow-copy "best state" (cell 12, and `.cpu()` on CPU in cell 15).** `state_dict().copy()` keeps referencing tensors that training keeps changing. `fit_model` uses `copy.deepcopy`.
10. **`squeeze()` on a batch of one (cell 8)** collapses the output to a scalar; `squeeze(1)` is used. BatchNorm models cannot train on a trailing batch of one, so the loader drops it.

### Score experiment (cell 15)
11. Medians, winsorisation bounds and category codes were computed on the **whole** frame before splitting -> `ScorePreprocessor`, fitted on train only.
12. `early_stopping_rounds` / `verbose_eval` were removed from `lightgbm.train` in v4 -> callbacks.
13. The feature-importance table used `features + ["lgb_oof_pred"]` (one name too many for the last fold's booster), and `try/except: pass` hid the error -> importance averaged over folds for base features.
14. Only the last fold's booster was saved -> all folds saved.
15. Small-cardinality integer columns were converted to *strings* and then category codes, which orders "10" before "2" -> numeric columns stay numeric.
16. Validation metrics for the MLP/ensemble are optimistic (validation drives early stopping); the report says so and calls test the honest number.

### Engineering
17. **Duplicate pipelines** (cells 9 and 12 differ only in model/loss/scheduler) -> one `train_regressor` driven by `RegressorTrainConfig` presets. **Three copies of the classifier preprocessing** (cells 3, 8 and the regressor's variant) -> `preprocessing.py`.
18. **Unused / broken code removed:** TensorFlow imports, `ComplexRegressionModel` (skip connections add a 256-d tensor to a 128-d tensor; it could never run), `spilt`, the incomplete `Models`, toy cells.
19. **Broken paths** (`/content/...`, `/content/kochi_models`) -> `config.py`; data path configurable; helpful error when the file is missing.
20. **Inference:** replaced cell 10's placeholder with `InductionPipeline`, which loads the fitted preprocessor next to the weights (the notebook reloaded the scaler separately and rebuilt the wrong architecture).
21. **Classifier head:** `Sigmoid` + `BCELoss` -> logit + `BCEWithLogitsLoss` (same function, numerically stable).
22. `weights_only=True` when loading checkpoints; library versions stored in each bundle's `metadata.json`.

## 3. Deliberately preserved from the notebook
Seeds (42), layer stacks of all three MLPs, optimisers/schedulers/losses, batch sizes, epochs, test/val fractions for the regressor, the three scalers, the 11 regressor features, `Mileage_Category` thresholds, winsorisation quantiles, 5-fold OOF stacking and the 0.5/0.5 ensemble.
The regressor still trains on **all** rows including Standby (0 km) and still drops `FinalDecision`, as the notebook did.

## 4. Open questions (not decided by this repo)
* Is `FinalDecision` (largely) a deterministic function of the validity/job-card columns? If so the classifier mostly re-learns rules; check with the EDA notebook before reading any accuracy.
* Should the regressor train only on Service rows, or take the classifier's decision as an input? Untested in the notebook; R^2 of 0.27-0.36 with a bimodal target (0 or hundreds of km) may reflect the mixed population.
* Are random row splits appropriate when the same `Trainset` appears on many days? `split_strategy="chronological"` is provided to compare.
* `Mileage_Category` thresholds vs the real `Cum_Mileage` range.
* `GATE_KM_BY_DECISION` (force 0 km for predicted Standby) is off until verified.
* Whether `Depot`, `Bay_Position`, `Stabling_Difficulty` should be used by the regressor (the notebook dropped them without explanation).

## 5. Verification status of this repository
See the README section "Verification status" for exactly what was executed and what was only syntax-checked.
