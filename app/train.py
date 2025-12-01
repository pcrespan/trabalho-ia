import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
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
    y_raw = df["Creditability"].copy()
    try:
        y = pd.to_numeric(y_raw, errors='raise').astype(int)
        le = None
    except Exception:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    return X, y, le

def train_and_persist_models(training_csv_path: str = DEFAULT_TRAINING_CSV, random_state: int = 42):
    _ensure_models_dir()
    preproc_path = find_preprocessor_path()
    preprocessor = load_preprocessor(preproc_path)
    df = _load_training_df(training_csv_path)
    X, y, label_encoder = _prepare_Xy(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train_proc = preprocessor.fit_transform(X_train) if hasattr(preprocessor, "fit_transform") else preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    histories = {}
    logistic = SGDClassifier(loss="log_loss", max_iter=1, tol=None, warm_start=True, random_state=random_state)
    n_epochs = 30
    train_loss_log = []
    test_loss_log = []
    classes = np.unique(y_train)
    for epoch in range(n_epochs):
        logistic.max_iter = 1
        logistic.partial_fit(X_train_proc, y_train, classes=classes)
        if hasattr(logistic, "predict_proba"):
            y_train_proba = logistic.predict_proba(X_train_proc)
            y_test_proba = logistic.predict_proba(X_test_proc)
        else:
            y_train_proba = np.vstack([1 - logistic.predict_proba(X_train_proc)[:, -1], logistic.predict_proba(X_train_proc)[:, -1]]).T if hasattr(logistic, "predict_proba") else np.clip(logistic._predict_proba_lr(X_train_proc),1e-15,1-1e-15)
            y_test_proba = np.clip(logistic._predict_proba_lr(X_test_proc),1e-15,1-1e-15)
        try:
            tl = log_loss(y_train, y_train_proba)
        except Exception:
            tl = np.nan
        try:
            tel = log_loss(y_test, y_test_proba)
        except Exception:
            tel = np.nan
        train_loss_log.append(tl)
        test_loss_log.append(tel)
    logistic_path = MODELS_DIR / "regression" / "logistic.pkl"
    joblib.dump(logistic, logistic_path)
    histories["Logistic"] = {"train_loss": train_loss_log, "test_loss": test_loss_log}
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, min_samples_split=10, max_features="sqrt", warm_start=True, random_state=random_state)
    max_trees = 100
    step = 10
    rf_train_loss = []
    rf_test_loss = []
    for n in range(10, max_trees + 1, step):
        rf.n_estimators = n
        rf.fit(X_train_proc, y_train)
        if hasattr(rf, "predict_proba"):
            y_train_proba = rf.predict_proba(X_train_proc)
            y_test_proba = rf.predict_proba(X_test_proc)
            try:
                rf_train_loss.append(log_loss(y_train, y_train_proba))
            except Exception:
                rf_train_loss.append(np.nan)
            try:
                rf_test_loss.append(log_loss(y_test, y_test_proba))
            except Exception:
                rf_test_loss.append(np.nan)
        else:
            rf_train_loss.append(np.nan)
            rf_test_loss.append(np.nan)
    rf_path = MODELS_DIR / "ensemble" / "random_forest.pkl"
    joblib.dump(rf, rf_path)
    histories["RandomForest"] = {"train_loss": rf_train_loss, "test_loss": rf_test_loss, "n_estimators": list(range(10, max_trees + 1, step))}
    if label_encoder is not None:
        le_path = MODELS_DIR / "label_encoder.pkl"
        joblib.dump(label_encoder, le_path)
    logistic_loaded = joblib.load(logistic_path)
    rf_loaded = joblib.load(rf_path)
    models = {"Logistic": logistic_loaded, "RandomForest": rf_loaded}
    accuracies = {}
    for name, m in models.items():
        try:
            y_pred = m.predict(X_test_proc)
            accuracies[name] = accuracy_score(y_test, y_pred)
        except Exception:
            accuracies[name] = None
    return {"models": {k: str((MODELS_DIR / ("regression" if k=="Logistic" else "ensemble" if k=="RandomForest" else "mlp") / ( "logistic.pkl" if k=="Logistic" else "random_forest.pkl" if k=="RandomForest" else "mlp.pkl")) ) for k in models.keys()}, "histories": histories, "accuracies": accuracies}
