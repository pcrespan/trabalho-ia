import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.base_model import BaseModel
from pathlib import Path

class EnsembleModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5, min_samples_split=10, max_features="sqrt")

    def fit(self, X_train, y_train, X_val, y_val):
        print(self)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        print("Accuracy:", accuracy_score(y_val, preds))
        print(classification_report(y_val, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        inst = cls.__new__(cls)

        if not isinstance(payload, dict):
            inst.model = payload
            return inst

        for key in ("model", "estimator", "sk_model"):
            if key in payload:
                inst.model = payload[key]
                return inst

        for v in payload.values():
            if hasattr(v, "predict"):
                inst.model = v
                return inst

        raise ValueError("Could not locate a scikit-learn estimator inside the loaded payload.")

    def __str__(self):
        return "EnsembleModel"
