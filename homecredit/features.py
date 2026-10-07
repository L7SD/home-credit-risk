"""Compute applicant-level features without fitting preprocessing statistics."""

from pathlib import Path
import numpy as np
import pandas as pd

ID = "SK_ID_CURR"
TARGET = "TARGET"
# Each table is loaded and discarded separately. Read only columns used below.
TABLES = {
    "bureau.csv": (
        "bureau",
        "DAYS_CREDIT",
        {
            "DAYS_CREDIT": ["mean", "min", "max", "std"],
            "CREDIT_DAY_OVERDUE": ["max"],
            "AMT_CREDIT_SUM": ["sum", "mean"],
        },
    ),
    "previous_application.csv": (
        "prev",
        "DAYS_DECISION",
        {
            "AMT_CREDIT": ["mean", "sum"],
            "AMT_ANNUITY": ["mean"],
            "DAYS_DECISION": ["min", "mean", "max"],
        },
    ),
    "POS_CASH_balance.csv": (
        "pos",
        "MONTHS_BALANCE",
        {
            "CNT_INSTALMENT": ["mean", "std"],
            "CNT_INSTALMENT_FUTURE": ["mean"],
            "MONTHS_BALANCE": ["min", "max"],
        },
    ),
    "credit_card_balance.csv": (
        "cc",
        "MONTHS_BALANCE",
        {
            "AMT_BALANCE": ["mean", "max", "std"],
            "AMT_PAYMENT_TOTAL_CURRENT": ["mean", "sum"],
        },
    ),
    "installments_payments.csv": (
        "inst",
        "DAYS_ENTRY_PAYMENT",
        {
            "AMT_PAYMENT": ["mean", "sum", "std"],
            "AMT_INSTALMENT": ["mean"],
            "PAYMENT_RATIO": ["mean", "std"],
        },
    ),
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return missing values for zero or negative denominators."""
    return (numerator / denominator.where(denominator > 0)).replace(
        [np.inf, -np.inf], np.nan
    )


def validate_applications(frame: pd.DataFrame, *, training: bool) -> None:
    if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Applications require unique, nonmissing SK_ID_CURR values")
    values = pd.to_numeric(frame[ID], errors="raise")
    if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
        raise ValueError("SK_ID_CURR must contain positive integer identifiers")
    if not len(frame):
        raise ValueError("Applications are empty")
    if training:
        if (
            TARGET not in frame
            or frame[TARGET].isna().any()
            or set(frame[TARGET].unique()) != {0, 1}
        ):
            raise ValueError(
                "Training TARGET must contain both 0 and 1, without missing values"
            )
    elif TARGET in frame:
        raise ValueError("Prediction input must not contain TARGET")


def aggregate_history(
    frame: pd.DataFrame, prefix: str, time_column: str, aggregations: dict
) -> pd.DataFrame:
    """Aggregate each applicant's records up to the application cutoff."""
    required = {ID, time_column, *aggregations}
    if prefix == "inst":
        required = (required - {"PAYMENT_RATIO"}) | {"AMT_PAYMENT", "AMT_INSTALMENT"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{prefix}: missing columns {sorted(missing)}")
    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (
        frame[ID].isna().any()
        or not np.isfinite(frame[ID]).all()
        or (frame[ID] <= 0).any()
        or (frame[ID] % 1 != 0).any()
    ):
        raise ValueError(f"{prefix}: applicant IDs must be positive integers")
    # Positive relative times represent records after the application cutoff.
    frame = frame.loc[frame[time_column].le(0)].copy()
    if prefix == "inst":
        frame["PAYMENT_RATIO"] = safe_ratio(
            frame["AMT_PAYMENT"], frame["AMT_INSTALMENT"]
        )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    grouped = frame.groupby(ID, sort=False)
    result = grouped.agg(aggregations)
    result.columns = [f"{prefix}_{column}_{stat}" for column, stat in result.columns]
    result[f"{prefix}_record_count"] = grouped.size()
    return result.reset_index()


def build_dataset(
    application_csv: str | Path,
    history_dir: str | Path | None = None,
    *,
    training: bool,
) -> pd.DataFrame:
    frame = pd.read_csv(application_csv)
    validate_applications(frame, training=training)
    frame[ID] = pd.to_numeric(frame[ID], errors="raise").astype("int64")
    if "DAYS_EMPLOYED" in frame:
        frame["DAYS_EMPLOYED"] = frame["DAYS_EMPLOYED"].replace(365243, np.nan)
    for name, numerator, denominator in [
        ("credit_income_ratio", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("annuity_income_ratio", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    ]:
        if {numerator, denominator} <= set(frame.columns):
            frame[name] = safe_ratio(frame[numerator], frame[denominator])
    if history_dir is not None:
        for filename, (prefix, time_column, aggregation) in TABLES.items():
            needed = {ID, time_column, *aggregation} - {"PAYMENT_RATIO"}
            history = pd.read_csv(
                Path(history_dir) / filename, usecols=lambda column: column in needed
            )
            if not needed <= set(history.columns):
                raise ValueError(
                    f"{filename}: missing columns {sorted(needed - set(history.columns))}"
                )
            history[ID] = pd.to_numeric(history[ID], errors="raise")
            if (
                history[ID].isna().any()
                or not np.isfinite(history[ID]).all()
                or (history[ID] <= 0).any()
                or (history[ID] % 1 != 0).any()
            ):
                raise ValueError(f"{filename}: applicant IDs must be positive integers")
            history = history.loc[history[ID].isin(frame[ID])]
            summary = aggregate_history(history, prefix, time_column, aggregation)
            frame = frame.merge(
                summary, how="left", on=ID, validate="one_to_one", sort=False
            )
            del history, summary
    return frame.replace([np.inf, -np.inf], np.nan)


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier and target columns before fitting a model."""
    return frame.drop(
        columns=[c for c in frame if c == TARGET or c.startswith("SK_ID_")]
    )
