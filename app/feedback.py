import os
import pandas as pd
from pathlib import Path
from app.constants import FEATURE_COLUMNS

DEFAULT_TRAINING_CSV = os.environ.get("TRAINING_CSV_PATH", "data/training_data.csv")

def _ensure_dir(path: str):
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def _detect_label_format(existing_df: pd.DataFrame, label_col: str = "Creditability"):
    if label_col not in existing_df.columns:
        return "int"
    series = existing_df[label_col].dropna()
    if series.empty:
        return "int"
    try:
        pd.to_numeric(series).astype(int)
        return "int"
    except Exception:
        return "str"

def save_feedback_row(input_row: dict, label: str, csv_path: str = DEFAULT_TRAINING_CSV):
    _ensure_dir(csv_path)
    csv_p = Path(csv_path)

    row = {col: input_row.get(col, pd.NA) for col in FEATURE_COLUMNS}

    if csv_p.exists():
        existing = pd.read_csv(csv_p, sep=';')
        label_format = _detect_label_format(existing, label_col="Creditability")
        if label_format == "int":
            mapping = {"Good": 1, "Bad": 0, "good": 1, "bad": 0}
            credit_val = mapping.get(label, label)
            try:
                credit_val = int(credit_val)
            except Exception:
                credit_val = str(credit_val)
        else:
            credit_val = str(label)

        expected_columns = existing.columns.tolist()
        if "Creditability" not in expected_columns:
            expected_columns = ["Creditability"] + [c for c in expected_columns if c != "Creditability"]
        df_row = pd.DataFrame([{}])
        for c in expected_columns:
            if c == "Creditability":
                df_row[c] = [credit_val]
            elif c in FEATURE_COLUMNS:
                df_row[c] = [row.get(c, pd.NA)]
            else:
                df_row[c] = [pd.NA]

        df_row.to_csv(csv_p, mode="a", header=False, index=False, sep=';')
    else:
        mapping = {"Good": 1, "Bad": 0, "good": 1, "bad": 0}
        credit_val = mapping.get(label, label)

        cols = ["Creditability"] + FEATURE_COLUMNS
        values = [credit_val] + [row[c] for c in FEATURE_COLUMNS]
        df_row = pd.DataFrame([values], columns=cols)
        df_row.to_csv(csv_p, mode="w", header=True, index=False, sep=';')
