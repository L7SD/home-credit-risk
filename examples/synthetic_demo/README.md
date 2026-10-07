# Synthetic example

These reports come from 500 synthetic application records with seed 42. The run compared the three default models and the optional XGBoost candidate. Its random forest used 40 trees.

The scores describe this example dataset, not the Home Credit competition data. `metrics.json` records the model settings, split sizes, and package versions.

After setting up the environment described in the root README, run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-xgboost.txt
.venv\Scripts\python.exe -m homecredit demo --output data/example --rows 500 --seed 42
.venv\Scripts\python.exe -m homecredit train --applications data/example/application_train.csv --history-dir data/example --output artifacts/example --trees 40 --seed 42 --refit --importance --xgboost
```

Use empty output directories. The resulting metrics, ROC plot, and importance table correspond to the files in this folder; timestamps may differ.
