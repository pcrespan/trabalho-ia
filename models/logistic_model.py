import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.base_model import BaseModel
from pathlib import Path

class LogisticModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        print(self)
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

        inst.model = payload
        return inst

    def __str__(self):
        return "LogisticModel"
