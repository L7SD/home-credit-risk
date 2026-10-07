"""Create synthetic CSVs for local examples and tests."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .features import ID, TARGET


def create_demo(destination: Path, rows: int = 500, seed: int = 42) -> None:
    destination = Path(destination)
    if rows < 100:
        raise ValueError("Use at least 100 demo rows")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Demo destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count = rows + max(30, rows // 5)
    income = rng.lognormal(11.8, 0.4, count)
    ratio = rng.uniform(0.5, 8, count)
    risk = 1 / (1 + np.exp(-(-3 + 0.55 * ratio)))
    frame = pd.DataFrame(
        {
            ID: np.arange(100001, 100001 + count),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": income * ratio,
            "AMT_ANNUITY": income * rng.uniform(0.03, 0.3, count),
            "DAYS_EMPLOYED": -rng.integers(20, 8000, count).astype(float),
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Pensioner", "Commercial associate"], count
            ),
            "EXT_SOURCE_2": np.clip(1 - risk + rng.normal(0, 0.12, count), 0, 1),
        }
    )
    frame.loc[rng.choice(count, count // 12, replace=False), "EXT_SOURCE_2"] = np.nan
    frame.loc[rng.choice(count, count // 20, replace=False), "DAYS_EMPLOYED"] = 365243
    train = frame.iloc[:rows].copy()
    train[TARGET] = rng.binomial(1, risk[:rows])
    train.to_csv(destination / "application_train.csv", index=False)
    frame.iloc[rows:].to_csv(destination / "application_test.csv", index=False)
    ids = np.repeat(frame[ID].to_numpy(), 3)
    n = len(ids)
    pd.DataFrame(
        {
            ID: ids,
            "DAYS_CREDIT": -rng.integers(1, 2000, n),
            "CREDIT_DAY_OVERDUE": rng.integers(0, 30, n),
            "AMT_CREDIT_SUM": rng.uniform(1000, 150000, n),
        }
    ).to_csv(destination / "bureau.csv", index=False)
    pd.DataFrame(
        {
            ID: ids,
            "AMT_CREDIT": rng.uniform(1000, 150000, n),
            "AMT_ANNUITY": rng.uniform(100, 10000, n),
            "DAYS_DECISION": -rng.integers(1, 2000, n),
        }
    ).to_csv(destination / "previous_application.csv", index=False)
    pd.DataFrame(
        {
            ID: ids,
            "CNT_INSTALMENT": rng.integers(6, 60, n),
            "CNT_INSTALMENT_FUTURE": rng.integers(0, 6, n),
            "MONTHS_BALANCE": -rng.integers(1, 48, n),
        }
    ).to_csv(destination / "POS_CASH_balance.csv", index=False)
    pd.DataFrame(
        {
            ID: ids,
            "AMT_BALANCE": rng.uniform(0, 20000, n),
            "AMT_PAYMENT_TOTAL_CURRENT": rng.uniform(0, 3000, n),
            "MONTHS_BALANCE": -rng.integers(1, 48, n),
        }
    ).to_csv(destination / "credit_card_balance.csv", index=False)
    installments = rng.uniform(100, 5000, n)
    pd.DataFrame(
        {
            ID: ids,
            "AMT_INSTALMENT": installments,
            "AMT_PAYMENT": installments * rng.uniform(0.5, 1.1, n),
            "DAYS_ENTRY_PAYMENT": -rng.integers(1, 500, n),
        }
    ).to_csv(destination / "installments_payments.csv", index=False)
    (destination / "SYNTHETIC_DATA.json").write_text(
        json.dumps(
            {"kind": "synthetic-demo", "seed": seed, "training_rows": rows}, indent=2
        )
        + "\n"
    )
