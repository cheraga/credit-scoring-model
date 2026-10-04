
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = "/content/credit-scoring-model"

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(
    f"{BASE}/models/best_credit_scoring_model.joblib"
)

# --------------------------------------------------
# Load test data
# --------------------------------------------------

test_df = pd.read_csv(
    f"{BASE}/data/processed/test_set.csv"
)

X_test = test_df.drop(columns=["credit_risk"])
y_test = test_df["credit_risk"]

# --------------------------------------------------
# Get probabilities
# --------------------------------------------------

bad_probability = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Thresholds
# --------------------------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

results = []

# --------------------------------------------------
# Evaluate each threshold
# --------------------------------------------------

for threshold in thresholds:

    y_pred = (
        bad_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    })

# --------------------------------------------------
# Results dataframe
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("=" * 90)
print("THRESHOLD ANALYSIS")
print("=" * 90)

print(
    results_df.to_string(index=False)
)

# --------------------------------------------------
# Best threshold by F1
# --------------------------------------------------

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print("\nBest threshold according to F1:")
print(
    best_f1.to_string()
)

# --------------------------------------------------
# Best threshold by Recall
# --------------------------------------------------

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

print("\nBest threshold according to Recall:")
print(
    best_recall.to_string()
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df.to_csv(
    f"{BASE}/results/metrics/threshold_analysis.csv",
    index=False
)

print(
    "\nSaved:"
    f" {BASE}/results/metrics/threshold_analysis.csv"
)

print(
    "\nThreshold analysis completed successfully."
)
