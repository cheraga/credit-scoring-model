
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = ROOT / "data" / "processed" / "test_set.csv"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load test data
# --------------------------------------------------

test_df = pd.read_csv(TEST_FILE)

X_test = test_df.drop(columns=["credit_risk"])
y_test = test_df["credit_risk"]

# --------------------------------------------------
# Models actually saved by the current project
# --------------------------------------------------

models = {
    "Tuned XGBoost": MODELS_DIR / "best_credit_scoring_model.joblib",
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "XGBoost": MODELS_DIR / "xgboost.joblib",
}

results = []

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

for model_name, model_path in models.items():

    if not model_path.exists():
        print(f"WARNING: Missing model: {model_path}")
        continue

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    })

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = METRICS_DIR / "final_model_comparison.csv"

results_df.to_csv(output_file, index=False)

print("\n=== FINAL MODEL COMPARISON ===")
print(results_df.to_string(index=False))

print(f"\nSaved: {output_file}")
