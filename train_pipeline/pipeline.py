import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from models.logistic_model import LogisticModel
from models.ensemble_model import EnsembleModel
from models.mlp_model import MLPModel

df = pd.read_csv("../data/german.csv", sep=";")
target = "Creditability"

X = df.drop(columns=[target])
y = df[target]

categorical = X.select_dtypes(include=["object"]).columns
numerical = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_processed = preprocessor.fit_transform(X)
X_processed = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

os.makedirs("models", exist_ok=True)

log_model = LogisticModel()
os.makedirs("models/regression", exist_ok=True)
log_model.fit(X_train, y_train, X_val, y_val)
log_model.save("models/regression/logistic.pkl")

ens_model = EnsembleModel()
os.makedirs("models/ensemble", exist_ok=True)
ens_model.fit(X_train, y_train, X_val, y_val)
ens_model.save("models/ensemble/random_forest.pkl")

mlp = MLPModel(input_dim=X_train.shape[1])
os.makedirs("models/mlp", exist_ok=True)
mlp.fit(X_train, y_train, X_val, y_val)
mlp.save("models/mlp/mlp.pt")

print("Training completed.")
