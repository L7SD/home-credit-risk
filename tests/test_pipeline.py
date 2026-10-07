import json
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from homecredit.demo import create_demo
from homecredit.features import (
    ID,
    TARGET,
    TABLES,
    aggregate_history,
    build_dataset,
    model_features,
    safe_ratio,
    validate_applications,
)
from homecredit.modeling import make_pipeline, predict, run_experiment, split_rows


class FeatureTests(unittest.TestCase):
    def test_zero_denominator_is_missing(self):
        ratios = safe_ratio(pd.Series([5.0, 5.0, 5.0]), pd.Series([0.0, -1.0, 2.0]))
        self.assertTrue(ratios.iloc[:2].isna().all())
        self.assertEqual(ratios.iloc[2], 2.5)

    def test_history_stays_with_applicant_and_excludes_future_records(self):
        source = pd.DataFrame(
            {
                ID: [1, 1, 2, 1],
                "DAYS_CREDIT": [-1, -2, -3, 1],
                "CREDIT_DAY_OVERDUE": [0, 0, 5, 100],
                "AMT_CREDIT_SUM": [10, 20, 90, 999],
            }
        )
        prefix, cutoff, agg = TABLES["bureau.csv"]
        result = aggregate_history(source, prefix, cutoff, agg).set_index(ID)
        self.assertEqual(result.loc[1, "bureau_AMT_CREDIT_SUM_sum"], 30)
        self.assertEqual(result.loc[2, "bureau_AMT_CREDIT_SUM_sum"], 90)
        self.assertEqual(result.loc[1, "bureau_record_count"], 2)

    def test_schema_blocks_duplicate_ids_and_leaking_targets(self):
        with self.assertRaises(ValueError):
            validate_applications(
                pd.DataFrame({ID: [1, 1], TARGET: [0, 1]}), training=True
            )
        with self.assertRaises(ValueError):
            validate_applications(
                pd.DataFrame({ID: [1, 2], TARGET: [0, 1]}), training=False
            )
        self.assertEqual(
            list(
                model_features(
                    pd.DataFrame(
                        {ID: [1], TARGET: [0], "SK_ID_PREV": [5], "income": [10]}
                    )
                )
            ),
            ["income"],
        )


class EvaluationTests(unittest.TestCase):
    def test_splits_are_disjoint_complete_and_deterministic(self):
        target = pd.Series([0, 1] * 100)
        splits = split_rows(target, 42)
        self.assertEqual([len(x) for x in splits], [120, 40, 40])
        self.assertEqual(len(set(np.concatenate(splits))), 200)
        for left, right in zip(splits, split_rows(target, 42)):
            np.testing.assert_array_equal(left, right)

    def test_imputer_and_encoder_only_learn_training_rows(self):
        X = pd.DataFrame(
            {"numeric": [1.0, 3.0, np.nan, 5.0], "category": ["known"] * 4}
        )
        pipeline = make_pipeline(LogisticRegression()).fit(X, [0, 1, 0, 1])
        heldout = pd.DataFrame(
            {"numeric": [999999.0, np.nan], "category": ["unseen", "unseen"]}
        )
        prediction = pipeline.predict_proba(heldout)
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        self.assertEqual(numeric.named_steps["impute"].statistics_[0], 3.0)
        categories = (
            pipeline.named_steps["preprocess"]
            .named_transformers_["categorical"]
            .named_steps["encode"]
            .categories_[0]
        )
        self.assertNotIn("unseen", categories)
        self.assertEqual(prediction.shape, (2, 2))
        self.assertTrue(np.isfinite(prediction).all())

    def test_full_synthetic_train_save_reload_predict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_demo(root / "data", rows=150)
            frame = build_dataset(
                root / "data/application_train.csv", root / "data", training=True
            )
            report = run_experiment(
                frame,
                root / "run",
                trees=8,
                use_history=True,
                refit=True,
                data_kind="synthetic-demo",
            )
            self.assertEqual(sum(report["split_rows"].values()), 150)
            self.assertEqual(
                set(report["validation"]),
                {"dummy", "logistic_regression", "random_forest"},
            )
            self.assertEqual(
                json.loads((root / "run/metrics.json").read_text())["data_kind"],
                "synthetic-demo",
            )
            bundle = joblib.load(root / "run/submission_model.joblib")
            self.assertNotIn(ID, bundle["columns"])
            self.assertEqual(bundle["fit_scope"], "all_labeled_rows")
            test = build_dataset(
                root / "data/application_test.csv", root / "data", training=False
            )
            predict(
                root / "run/submission_model.joblib",
                test,
                root / "submission.csv",
                use_history=True,
            )
            output = pd.read_csv(root / "submission.csv")
            self.assertEqual(list(output.columns), [ID, TARGET])
            self.assertTrue(output[TARGET].between(0, 1).all())
            self.assertEqual(output[ID].tolist(), test[ID].tolist())
            with self.assertRaisesRegex(ValueError, "history"):
                predict(
                    root / "run/submission_model.joblib",
                    test,
                    root / "wrong.csv",
                    use_history=False,
                )
            with self.assertRaisesRegex(ValueError, "schema"):
                predict(
                    root / "run/submission_model.joblib",
                    test.drop(columns=["AMT_CREDIT"]),
                    root / "wrong.csv",
                    use_history=True,
                )
            with self.assertRaisesRegex(ValueError, "not empty"):
                run_experiment(frame, root / "run", trees=1)


if __name__ == "__main__":
    unittest.main()
