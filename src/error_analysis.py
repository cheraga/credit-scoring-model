
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report
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
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Good Credit",
            "Bad Credit"
        ]
    )
)

# --------------------------------------------------
# Create analysis dataframe
# --------------------------------------------------

analysis_df = X_test.copy()

analysis_df["actual"] = y_test.values
analysis_df["predicted"] = y_pred
analysis_df["bad_probability"] = y_prob

# --------------------------------------------------
# Error type
# --------------------------------------------------

def classify_result(row):

    actual = row["actual"]
    predicted = row["predicted"]

    if actual == 0 and predicted == 0:
        return "True Negative"

    elif actual == 1 and predicted == 1:
        return "True Positive"

    elif actual == 0 and predicted == 1:
        return "False Positive"

    elif actual == 1 and predicted == 0:
        return "False Negative"

    return "Unknown"


analysis_df["result"] = analysis_df.apply(
    classify_result,
    axis=1
)

# --------------------------------------------------
# Error counts
# --------------------------------------------------

print("\nPrediction categories:")

print(
    analysis_df["result"].value_counts()
)

# --------------------------------------------------
# False Negatives
# --------------------------------------------------

false_negatives = analysis_df[
    analysis_df["result"] == "False Negative"
].copy()

print(
    "\nNumber of False Negatives:",
    len(false_negatives)
)

print(
    "\nFalse Negatives:"
)

print(
    false_negatives[
        ["actual", "predicted", "bad_probability"]
    ].to_string()
)

# --------------------------------------------------
# False Positives
# --------------------------------------------------

false_positives = analysis_df[
    analysis_df["result"] == "False Positive"
].copy()

print(
    "\nNumber of False Positives:",
    len(false_positives)
)

print(
    "\nFalse Positives:"
)

print(
    false_positives[
        ["actual", "predicted", "bad_probability"]
    ].to_string()
)

# --------------------------------------------------
# Save complete analysis
# --------------------------------------------------

analysis_df.to_csv(
    f"{BASE}/results/metrics/error_analysis.csv",
    index=False
)

false_negatives.to_csv(
    f"{BASE}/results/metrics/false_negatives.csv",
    index=False
)

false_positives.to_csv(
    f"{BASE}/results/metrics/false_positives.csv",
    index=False
)

print(
    "\nSaved:"
    f"\n- {BASE}/results/metrics/error_analysis.csv"
    f"\n- {BASE}/results/metrics/false_negatives.csv"
    f"\n- {BASE}/results/metrics/false_positives.csv"
)

print("\nError analysis completed successfully.")
