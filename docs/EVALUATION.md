# Evaluation design

## Features

Historical records are grouped by applicant before joining application data. The summaries use each applicant's own records and do not learn population statistics or use target labels. Positive relative observation times are excluded.

Payment, credit-to-income, and annuity-to-income ratios use a shared helper. Zero or negative denominators become missing values rather than extreme ratios. Identifier columns are used for joins and output alignment, then removed from model features.

## Training boundary

The partition split happens before fitting preprocessing. Numeric columns use median imputation and scaling; categorical columns use a training-fitted encoder that accepts unseen categories at prediction time. Each candidate gets its own preprocessing/model pipeline.

The earlier notebook calculated medians before validation splitting and aligned categories using the competition test data. Scores from that workflow are not directly comparable to results from the current pipeline.

## Model selection

Validation ROC-AUC selects one candidate. Its final test scores are recorded once. Repeatedly using the reserved test results to choose features or parameters would turn that partition into another validation set.

Average precision complements ROC-AUC on an imbalanced target. Brier score measures probability error. Precision, recall, and the confusion matrix use a fixed threshold of 0.5; matrix rows are actual labels and columns are predicted labels, ordered `[0, 1]`.

Optional permutation importance uses at most 1,000 validation rows. It measures the effect of shuffling each raw feature, not causation.

## Saved models

`evaluation_model.joblib` retains the selected training-partition pipeline. The optional `submission_model.joblib` is a separate clone refitted on all labeled records after scoring. A model bundle stores the expected columns, history-table setting, scikit-learn version, and fit scope.

Record the full environment with `python -m pip freeze` alongside an actual experiment. Keep its split assignments and settings with the results so another run can be compared on the same basis.
