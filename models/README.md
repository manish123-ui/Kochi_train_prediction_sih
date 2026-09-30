# Models

Trained artifacts are written here and are git-ignored (no pretrained weights ship with the repo).

```
models/
├── induction_classifier/   # FinalDecision (Service/Standby)
│   ├── model.pt            # weights + architecture name/kwargs (loaded with weights_only=True)
│   ├── preprocessor.joblib # fitted ColumnTransformer (train-split statistics)
│   └── metadata.json       # training report: config, split sizes, metrics, library versions
└── daily_km_regressor/     # Daily_Km_Assigned
    └── (same three files)
```

Create them with:

```bash
python -m src.main train-classifier --data data/raw/kochi_induction_dataset.csv
python -m src.main train-regressor  --data data/raw/kochi_induction_dataset.csv --model simple
```

Security note: `preprocessor.joblib` is a pickle. Only load bundles you trained yourself.
