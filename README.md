# Home Credit Risk

A reproducible classification workflow for the Home Credit Default Risk dataset. It combines application records with five historical tables, compares models, and saves the fitted preprocessing pipeline for prediction.

## Run the example

Install Python 3.12 and open PowerShell in this folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m homecredit demo --output data/demo
.venv\Scripts\python.exe -m homecredit train --applications data/demo/application_train.csv --history-dir data/demo --output artifacts/demo-run --refit --importance
.venv\Scripts\python.exe -m homecredit predict --model artifacts/demo-run/submission_model.joblib --applications data/demo/application_test.csv --history-dir data/demo --output artifacts/demo-submission.csv
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` in the remaining commands.

The example uses synthetic records. Its scores check the workflow and are not competition benchmarks. Use a new output directory for each run; existing results are not overwritten.

## Use the competition data

Download the files from [Home Credit Default Risk on Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and extract them to `data/raw/`.

The application files are `application_train.csv` and `application_test.csv`. Training requires unique `SK_ID_CURR` values and binary `TARGET` labels. Prediction requires the same application feature columns without `TARGET`.

When using `--history-dir`, these five files are required:

- `bureau.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`

```powershell
.venv\Scripts\python.exe -m homecredit train --applications data/raw/application_train.csv --history-dir data/raw --output artifacts/run-01 --refit
.venv\Scripts\python.exe -m homecredit predict --model artifacts/run-01/submission_model.joblib --applications data/raw/application_test.csv --history-dir data/raw --output artifacts/submission.csv
```

For an application-only baseline, omit `--history-dir` from both commands. To include XGBoost, install `requirements-xgboost.txt` and add `--xgboost` to the training command. `--importance` adds sampled validation permutation importance and may increase runtime substantially.

## Evaluation

Applicants are split into 60% training, 20% validation, and 20% test. Imputation, categorical encoding, scaling, and model fitting use training rows only.

The default candidates are a prior-only dummy classifier, logistic regression, and a random forest. Validation ROC-AUC selects the candidate; only that model is evaluated on the reserved test partition. The decision threshold is fixed at 0.5.

Reports include ROC-AUC, average precision, Brier score, precision, recall, and a confusion matrix. With `--refit`, a separate model is fitted on all labeled rows after evaluation for submission generation. That refit has no independent score in the report.

See [evaluation details](docs/EVALUATION.md) and [test status](docs/TESTING.md).

## Outputs

| File | Contents |
|---|---|
| `metrics.json` | Scores, partition sizes, settings, and package versions |
| `split_assignments.csv` | Applicant IDs and their partitions |
| `evaluation_model.joblib` | Selected pipeline fitted on the training partition |
| `submission_model.joblib` | Optional refit using all labeled rows |
| `test_predictions.csv` | Reserved test labels and probabilities |
| `validation_roc.png` | Candidate ROC curves on validation data |
| `validation_importance.csv` | Optional validation permutation importance |

The prediction command checks feature names and the saved scikit-learn version. Load model files only from trusted sources; `joblib` uses Python serialization.

## Code layout

`features.py` validates inputs and builds applicant-level features. `modeling.py` owns preprocessing, model selection, reports, persistence, and prediction. `demo.py` creates synthetic input files. `__main__.py` exposes the command-line interface.

`notebooks/home_credit_baseline.ipynb` preserves the earlier seven-model exploration. It uses an older preprocessing order and is separate from the current evaluation workflow.

## Limitations

The split is random by applicant, not temporal or external validation. Relative-time filters exclude future history records but do not replace a full audit of when every feature becomes available. Probabilities are not calibrated, and fairness has not been evaluated. This project is for experimentation, not operational lending decisions.

Historical tables are read sequentially using selected columns, but each table must still fit in memory. Dataset files and trained models are excluded from version control.
