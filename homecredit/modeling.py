"""Train-only preprocessing and three-way stratified evaluation."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import ID, TARGET, model_features, validate_applications


def make_pipeline(estimator) -> Pipeline:
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    max_categories=40,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def split_rows(target: pd.Series, seed: int = 42):
    if target.value_counts().min() < 10:
        raise ValueError(
            "Need at least 10 rows in each class for a stratified 60/20/20 split"
        )
    development, test = train_test_split(
        np.arange(len(target)), test_size=0.2, random_state=seed, stratify=target
    )
    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=seed,
        stratify=target.iloc[development],
    )
    return train, validation, test


def metrics(target, probabilities) -> dict:
    predicted = probabilities >= 0.5  # Fixed before test evaluation.
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "precision_at_0_5": float(precision_score(target, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(target, predicted, zero_division=0)),
        "confusion_matrix_at_0_5": confusion_matrix(
            target, predicted, labels=[0, 1]
        ).tolist(),
    }


def candidates(seed: int, trees: int, use_xgboost: bool = False) -> dict:
    result = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=3000, C=0.5, solver="liblinear", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=trees,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=seed,
        ),
    }
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ValueError(
                "Install the optional XGBoost dependency before using --xgboost"
            ) from error
        result["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            tree_method="hist",
            eval_metric="auc",
            random_state=seed,
            n_jobs=1,
        )
    return result


def plot_validation(target, probabilities: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(target, proba)
        ax.plot(fpr, tpr, label=f"{name}: {roc_auc_score(target, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Validation ROC — model selection only",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_experiment(
    frame: pd.DataFrame,
    output: Path,
    *,
    seed: int = 42,
    trees: int = 150,
    use_history: bool = False,
    refit: bool = False,
    use_xgboost: bool = False,
    importance: bool = False,
    data_kind: str = "user-provided",
) -> dict:
    validate_applications(frame, training=True)
    if trees < 1:
        raise ValueError("trees must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory is not empty; use a new run directory to preserve results"
        )
    output.mkdir(parents=True, exist_ok=True)
    X, y = model_features(frame), frame[TARGET].astype(int)
    train, validation, test = split_rows(y, seed)
    fitted, scores, validation_predictions = {}, {}, {}
    for name, estimator in candidates(seed, trees, use_xgboost).items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X.iloc[train], y.iloc[train])
        predicted = pipeline.predict_proba(X.iloc[validation])[:, 1]
        fitted[name] = pipeline
        validation_predictions[name] = predicted
        scores[name] = metrics(y.iloc[validation], predicted)
    winner = max(scores, key=lambda name: scores[name]["roc_auc"])
    selected = fitted[winner]
    # Only the selected model is evaluated on the untouched test partition.
    test_predictions = selected.predict_proba(X.iloc[test])[:, 1]
    assignments = pd.DataFrame({ID: frame[ID], "split": ""})
    for label, rows in [("train", train), ("validation", validation), ("test", test)]:
        assignments.iloc[rows, assignments.columns.get_loc("split")] = label
    assignments.to_csv(output / "split_assignments.csv", index=False)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": data_kind,
        "seed": seed,
        "selected_model": winner,
        "selection_metric": "validation ROC-AUC",
        "decision_threshold": 0.5,
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "positive_fraction": {
            name: float(y.iloc[rows].mean())
            for name, rows in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        },
        "feature_count_before_encoding": len(X.columns),
        "history_tables_used": use_history,
        "validation": scores,
        "selected_model_test": metrics(y.iloc[test], test_predictions),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "parameters": {
            name: str(pipeline.named_steps["model"])
            for name, pipeline in fitted.items()
        },
        "limitations": [
            "Random applicant split is not a temporal or external validation.",
            "Probabilities are not calibrated or approved for lending decisions.",
            "Repeated experimentation on this test partition invalidates its holdout status.",
        ],
    }
    bundle = {
        "schema_version": 1,
        "pipeline": selected,
        "columns": list(X.columns),
        "use_history": use_history,
        "sklearn_version": sklearn.__version__,
        "fit_scope": "train_partition",
    }
    joblib.dump(bundle, output / "evaluation_model.joblib")
    pd.DataFrame(
        {
            ID: frame.iloc[test][ID].to_numpy(),
            TARGET: y.iloc[test].to_numpy(),
            "probability": test_predictions,
        }
    ).to_csv(output / "test_predictions.csv", index=False)
    plot_validation(
        y.iloc[validation], validation_predictions, output / "validation_roc.png"
    )
    if importance:
        from sklearn.inspection import permutation_importance

        sample = np.random.default_rng(seed).choice(
            validation, size=min(1000, len(validation)), replace=False
        )
        if y.iloc[sample].nunique() < 2:
            raise ValueError("Importance sample needs both classes")
        result = permutation_importance(
            selected,
            X.iloc[sample],
            y.iloc[sample],
            scoring="roc_auc",
            n_repeats=2,
            random_state=seed,
            n_jobs=1,
        )
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_auc_decrease": result.importances_mean,
                "std": result.importances_std,
            }
        ).sort_values("mean_auc_decrease", ascending=False).to_csv(
            output / "validation_importance.csv", index=False
        )
    if refit:
        full_pipeline = clone(selected).fit(X, y)
        joblib.dump(
            {**bundle, "pipeline": full_pipeline, "fit_scope": "all_labeled_rows"},
            output / "submission_model.joblib",
        )
        report["submission_model_note"] = (
            "Refit on all labeled rows after evaluation. It has no independent score here."
        )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def predict(
    bundle_path: Path, frame: pd.DataFrame, output: Path, *, use_history: bool
) -> None:
    # joblib/pickle artifacts execute code on load. Accept only artifacts you trust.
    bundle = joblib.load(bundle_path)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("sklearn_version") != sklearn.__version__
    ):
        raise ValueError(
            "Model format or scikit-learn version mismatch; use the training environment"
        )
    if bundle["use_history"] != use_history:
        raise ValueError(
            "Prediction must use the same history-table setting as training"
        )
    validate_applications(frame, training=False)
    features = model_features(frame)
    if set(features.columns) != set(bundle["columns"]):
        missing = set(bundle["columns"]) - set(features.columns)
        extra = set(features.columns) - set(bundle["columns"])
        raise ValueError(
            f"Feature schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    probabilities = bundle["pipeline"].predict_proba(features[bundle["columns"]])[:, 1]
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Invalid probabilities")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Prediction output already exists; choose a new path")
    pd.DataFrame({ID: frame[ID], TARGET: probabilities}).to_csv(output, index=False)
