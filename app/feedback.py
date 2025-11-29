import os
import pandas as pd
from pathlib import Path
from app.constants import FEATURE_COLUMNS

DEFAULT_TRAINING_CSV = os.environ.get("TRAINING_CSV_PATH", "data/training_data.csv")

def _ensure_dir(path: str):
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def save_feedback_row(input_row: dict, label: str, csv_path: str = DEFAULT_TRAINING_CSV):
    _ensure_dir(csv_path)
    df_row = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)
    df_row["label"] = label
    if Path(csv_path).exists():
        existing = pd.read_csv(csv_path)
        for c in df_row.columns:
            if c not in existing.columns:
                existing[c] = pd.NA
        for c in existing.columns:
            if c not in df_row.columns:
                df_row[c] = pd.NA
        df_row = df_row[existing.columns.tolist()]
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
