# Data dictionary

Source: the 19 column names, the printed `head()` rows and the `(4500, 19)` shape come from the outputs of the original
notebook. **Meanings below are inferred from the column names and are not documented anywhere in the notebook. Confirm them with
whoever owns the data before relying on them.** The real file is private and not in this repository.

## `kochi_induction_dataset.csv` (4,500 rows x 19 columns)

| Column | Example values seen | Role in this repo | Encoding (`src/data/preprocessing.py`) | Meaning (inferred) |
|---|---|---|---|---|
| `Date` | - | identifier, not a feature (used by the optional chronological split) | - | day of the record |
| `Trainset` | - | identifier, not a feature | - | train identifier |
| `Depot` | `DepotA`, `DepotB` | classifier feature only | one-hot, unseen -> all zeros | depot |
| `FinalDecision` | Service, Standby | **classifier target** | Service=1, Standby=0 | induction decision |
| `DecisionReason` | - | dropped (free text; may restate the label) | - | reason text |
| `Rolling_Valid` | 1 | both | bool -> 0/1 | rolling-stock fitness certificate valid |
| `Signalling_Valid` | 1 | both | bool -> 0/1 | signalling fitness valid |
| `Telecom_Valid` | 1 | both | bool -> 0/1 | telecom fitness valid |
| `Any_Fitness_Invalid` | 0 | both | bool -> 0/1 | any fitness certificate invalid |
| `Open_JobCards` | 0, 1 | both | numeric | count of open job cards |
| `Critical_JobCard` | 0 | both | bool -> 0/1 | a critical job card is open |
| `Branding_Hours_Left` | 0, 253, 82 | both | numeric | remaining branding (advertising) hours |
| `Cum_Mileage` | 20745 ... 52808 (rows seen) | both; also source of `Mileage_Category` | numeric | cumulative mileage |
| `Mileage_Label` | Underused, Balanced, Overused | both | 0 / 1 / 2 | mileage balance class |
| `Days_Since_Last_Deep_Clean` | 8-16 (rows seen) | both | numeric | days since last deep clean |
| `Cleaning_Slot_Scheduled` | `Scheduled`, missing | both | `Scheduled`->1, anything else (incl. missing)->0 | cleaning slot booked |
| `Bay_Position` | 2, 15, 20, 23, 29 (rows seen) | classifier only (dropped for the regressor, as in the notebook) | numeric | stabling bay position |
| `Stabling_Difficulty` | Low, Medium, High | classifier only (dropped for the regressor, as in the notebook) | 0 / 1 / 2 | stabling difficulty |
| `Daily_Km_Assigned` | 506, 543, 601, 673, 0, 478 | **regressor target**; never a classifier feature | numeric | kilometres assigned that day (0 on the one Standby row printed) |

## Derived feature

`Mileage_Category` = 0 if `Cum_Mileage` < 120,000; 1 if < 200,000; else 2 (thresholds from the notebook, in `config.py`).
Every `Cum_Mileage` value the notebook printed is below 53,000, so this feature may be constant on the real data
(a warning is logged when it is). Check in `notebooks/01_kochi_metro_eda.ipynb`.

## Feature sets

| Task | Features | Count |
|---|---|---|
| Classifier (`CLASSIFIER_NUMERIC_FEATURES` + `Depot`) | 5 flags, `Open_JobCards`, `Branding_Hours_Left`, `Cum_Mileage`, `Mileage_Label`, `Days_Since_Last_Deep_Clean`, `Cleaning_Slot_Scheduled`, `Bay_Position`, `Stabling_Difficulty`, `Mileage_Category`, one-hot `Depot` | 14 numeric + Depot one-hot |
| Regressor | the 11 columns printed by the notebook ("Feature set shape: (4500, 11)") | 11 |

## `Copy of output_mileage_mtd_scored(1).csv` (score experiment)
Never visible in the notebook: no column names, shape or values were printed. The target is named `score`. The experiment code
therefore detects column types automatically (see `src/experiments/score_stacking.py`).

## Prediction output columns (`InductionPipeline.predict`)
Every input column, plus `P_Service`, `Predicted_Decision`, `Predicted_Daily_Km_Raw`, `Predicted_Daily_Km`.

## Plan columns (`TrainPlanScheduler.build_plan`)
Prediction columns plus `Assigned_Status` (= `Predicted_Decision`) and `Planned_Daily_Km` (0 for Standby).
