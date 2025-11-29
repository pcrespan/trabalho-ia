# create_preprocessor.py
import joblib
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA = Path("../../data/german.csv")
OUT_ROOT = Path("../models")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, sep=";")
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

TARGET = "Creditability"
if TARGET not in df.columns:
    lower = {c.lower(): c for c in df.columns}
    if TARGET.lower() in lower:
        TARGET = lower[TARGET.lower()]
    else:
        raise RuntimeError(f"Target '{TARGET}' not found in CSV columns: {df.columns.tolist()}")

X = df.drop(columns=[TARGET])

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = X.select_dtypes(include=[float, int]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

preprocessor.fit(X)

joblib.dump(preprocessor, Path("preprocessor.joblib"))
joblib.dump(preprocessor, OUT_ROOT / "preprocessor.joblib")
print("Saved preprocessor.joblib to project root and models/preprocessor.joblib")
