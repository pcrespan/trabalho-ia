# app/constants.py
from pathlib import Path

PROJECT_ROOT = Path.cwd()
MODEL_DIR = Path("models")
TRAIN_PIPELINE_DIR = Path("train_pipeline") / "models"

PREPROCESSOR_CANDIDATES = [
    MODEL_DIR / "preprocessor" / "preprocessor.joblib",
    MODEL_DIR / "preprocessor.joblib",
    Path("preprocessor.joblib"),
    TRAIN_PIPELINE_DIR.parent / "preprocessor.joblib",  # train_pipeline/preprocessor.joblib
    TRAIN_PIPELINE_DIR / "preprocessor.joblib",
]

FEATURE_COLUMNS = [
    "Account_Balance",
    "Duration_of_Credit_monthly",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Credit_Amount",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Instalment_per_cent",
    "Sex_Marital_Status",
    "Guarantors",
    "Duration_in_Current_address",
    "Most_valuable_available_asset",
    "Age_years",
    "Concurrent_Credits",
    "Type_of_apartment",
    "No_of_Credits_at_this_Bank",
    "Occupation",
    "No_of_dependents",
    "Telephone",
    "Foreign_Worker",
]

CATEGORICAL_OPTIONS = {
    "Account_Balance": [("No account", 1), ("None (No balance)", 2), ("Some balance", 3)],
    "Payment_Status_of_Previous_Credit": [("Some Problems", 1), ("Paid Up", 2), ("No Problems", 3)],
    "Purpose": [
        ("New car", 0),
        ("Used car", 1),
        ("Furniture/Equipment", 2),
        ("Radio/TV", 3),
        ("Domestic appliances", 4),
        ("Repairs", 5),
        ("Education", 6),
        ("Vacation", 7),
        ("Retraining", 8),
        ("Business", 9),
        ("Other", 10),
    ],
    "Value_Savings_Stocks": [("None", 1), ("Below 100 DM", 2), ("[100,1000] DM", 3), ("Above 1000 DM", 4)],
    "Length_of_current_employment": [("<1 year", 1), ("[1,4) years", 2), ("[4,7) years", 3), (">=7 years", 4)],
    "Sex_Marital_Status": [("Male Divorced/Single", 1), ("Male Married/Widowed", 2), ("Female", 3)],
    "Guarantors": [("None", 1), ("Yes", 2)],
    "Most_valuable_available_asset": [("None", 1), ("Car", 2), ("Real estate", 3), ("Savings/Stocks", 4)],
    "Concurrent_Credits": [("Other Banks or Dept Stores", 1), ("None", 2)],
    "Type_of_apartment": [("Unknown", 1), ("Rent", 2), ("Own", 3)],
    "No_of_Credits_at_this_Bank": [("1", 1), ("More than 1", 2)],
    "Occupation": [("Unskilled", 1), ("Skilled", 2), ("Highly skilled", 3), ("Other", 4)],
    "Telephone": [("No", 1), ("Yes", 2)],
    "Foreign_Worker": [("No", 1), ("Yes", 2)],
}

NUMERIC_RANGES = {
    "Duration_of_Credit_monthly": (1, 500),
    "Credit_Amount": (0, 1_000_000),
    "Instalment_per_cent": (0, 100),
    "Duration_in_Current_address": (0, 100),
    "Age_years": (16, 120),
    "No_of_dependents": (0, 50),
}

# explicit model file candidates (look in models/ first, then train_pipeline/models/)
MODEL_FILENAMES = {
    "logistic": [
        MODEL_DIR / "regression" / "logistic.pkl",
        MODEL_DIR / "regression" / "logistic.joblib",
        MODEL_DIR / "regression" / "model.joblib",
        TRAIN_PIPELINE_DIR / "regression" / "logistic.pkl",
        TRAIN_PIPELINE_DIR / "logistic.pkl",
    ],
    "ensemble": [
        MODEL_DIR / "ensemble" / "random_forest.pkl",
        MODEL_DIR / "ensemble" / "random_forest.joblib",
        MODEL_DIR / "ensemble" / "model.joblib",
        TRAIN_PIPELINE_DIR / "ensemble" / "random_forest.pkl",
        TRAIN_PIPELINE_DIR / "random_forest.pkl",
    ],
    #"mlp": [
    #    MODEL_DIR / "mlp" / "mlp.pt",
    #    MODEL_DIR / "mlp" / "model.pt",
    #    TRAIN_PIPELINE_DIR / "mlp" / "mlp.pt",
    #    TRAIN_PIPELINE_DIR / "mlp.pt",
    #],
}
