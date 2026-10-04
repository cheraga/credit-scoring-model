
import pandas as pd
import numpy as np
import joblib

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
# Select one applicant
# --------------------------------------------------

applicant_index = 0

applicant = X_test.iloc[[applicant_index]]

actual = y_test.iloc[applicant_index]

# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(applicant)[0]

probabilities = model.predict_proba(applicant)[0]

good_probability = probabilities[0]
bad_probability = probabilities[1]

# --------------------------------------------------
# Display result
# --------------------------------------------------

print("=" * 60)
print("CREDIT SCORING PREDICTION")
print("=" * 60)

print(f"\nApplicant index: {applicant_index}")

print("\nActual class:")
print(
    "Good Credit" if actual == 0 else "Bad Credit"
)

print("\nPredicted class:")
print(
    "Good Credit" if prediction == 0 else "Bad Credit"
)

print(f"\nGood Credit probability: {good_probability:.2%}")
print(f"Bad Credit probability:  {bad_probability:.2%}")

print("\nPrediction completed successfully.")
