import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.base_model import BaseModel

class EnsembleModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)

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

    def __str__(self):
        return "EnsembleModel"
