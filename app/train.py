# app/train.py
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from app.utils import find_preprocessor_path, load_preprocessor
from app.constants import FEATURE_COLUMNS

DEFAULT_TRAINING_CSV = os.environ.get("TRAINING_CSV_PATH", "../data/german.csv")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "../train_pipeline/models"))

def _ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "regression").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "ensemble").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "mlp").mkdir(parents=True, exist_ok=True)

def _load_training_df(csv_path: str):
    df = pd.read_csv(csv_path, sep=';')
    return df

def _prepare_Xy(df: pd.DataFrame):
    df = df.copy()
    if "Creditability" not in df.columns:
        raise ValueError("Training CSV must contain a 'Creditability' column")
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df["Creditability"].astype(str).copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le

def train_and_persist_models(training_csv_path: str = DEFAULT_TRAINING_CSV):
    _ensure_models_dir()

    preproc_path = find_preprocessor_path()
    preprocessor = load_preprocessor(preproc_path)

    df = _load_training_df(training_csv_path)
    X, y, label_encoder = _prepare_Xy(df)

    X_proc = preprocessor.transform(X)

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_proc, y)
    logistic_path = MODELS_DIR / "regression" / "logistic.pkl"
    joblib.dump(logistic, logistic_path)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_proc, y)
    rf_path = MODELS_DIR / "ensemble" / "random_forest.pkl"
    joblib.dump(rf, rf_path)

    mlp = MLPClassifier(hidden_layer_sizes=(128, ), max_iter=400)
    mlp.fit(X_proc, y)
    mlp_path = MODELS_DIR / "mlp" / "mlp.pkl"
    joblib.dump(mlp, mlp_path)

    le_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, le_path)

    return {
        "logistic": str(logistic_path),
        "random_forest": str(rf_path),
        "mlp": str(mlp_path),
        "label_encoder": str(le_path),
    }
