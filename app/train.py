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
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

def _ensure_models_dir():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

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
    joblib.dump(logistic, os.path.join(MODELS_DIR, "logistic.pkl"))

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_proc, y)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    mlp = MLPClassifier(hidden_layer_sizes=(128, ), max_iter=400)
    mlp.fit(X_proc, y)
    joblib.dump(mlp, os.path.join(MODELS_DIR, "mlp.pkl"))

    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return {
        "logistic": os.path.join(MODELS_DIR, "logistic.pkl"),
        "random_forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
        "mlp": os.path.join(MODELS_DIR, "mlp.pkl"),
        "label_encoder": os.path.join(MODELS_DIR, "label_encoder.pkl"),
    }
