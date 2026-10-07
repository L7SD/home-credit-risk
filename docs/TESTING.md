# Tests

Run the suite from the project root in the Python 3.12 environment:

```sh
python -m unittest discover -s tests -v
```

All 6 tests passed. They cover:

- Applicant-level history aggregation and exclusion of future records.
- Zero-denominator handling and rejection of duplicate IDs or target leakage.
- Disjoint, complete, repeatable train/validation/test partitions.
- Training-only imputation and categorical encoding.
- Synthetic training, model persistence, reload, and prediction.

The end-to-end test uses small synthetic inputs. The example folder also contains reports from a synthetic run with the optional XGBoost candidate; its README records the command settings.

The current pipeline has not been benchmarked on the full competition dataset. The historical notebook's saved scores belong to its earlier preprocessing workflow. Neither the synthetic scores nor those historical scores establish the current pipeline's competition performance.

GitHub Actions runs the core test suite. It does not download Kaggle data or train the optional XGBoost candidate.
