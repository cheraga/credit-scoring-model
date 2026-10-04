
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

BASE = "/content/credit-scoring-model"

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(
    f"{BASE}/models/best_credit_scoring_model.joblib"
)

print("Model loaded successfully.")

# --------------------------------------------------
# Load test data
# --------------------------------------------------

test_df = pd.read_csv(
    f"{BASE}/data/processed/test_set.csv"
)

X_test = test_df.drop(columns=["credit_risk"])
y_test = test_df["credit_risk"]

# --------------------------------------------------
# Select applicant
# --------------------------------------------------

applicant_index = 0

X_applicant = X_test.iloc[[applicant_index]]
y_actual = y_test.iloc[applicant_index]

# --------------------------------------------------
# Extract pipeline components
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# --------------------------------------------------
# Transform applicant
# --------------------------------------------------

X_transformed = preprocessor.transform(X_applicant)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out(
    X_test.columns
)

# --------------------------------------------------
# SHAP Explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(
    X_transformed
)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 1:
    applicant_shap = shap_values
else:
    applicant_shap = shap_values[0]

# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(X_applicant)[0]

probabilities = model.predict_proba(X_applicant)[0]

good_probability = probabilities[0]
bad_probability = probabilities[1]

# --------------------------------------------------
# Display prediction
# --------------------------------------------------

print("=" * 70)
print("INDIVIDUAL CREDIT RISK EXPLANATION")
print("=" * 70)

print(f"\nApplicant index: {applicant_index}")

print(
    "\nActual class:",
    "Good Credit" if y_actual == 0 else "Bad Credit"
)

print(
    "Predicted class:",
    "Good Credit" if prediction == 0 else "Bad Credit"
)

print(f"\nGood Credit probability: {good_probability:.2%}")
print(f"Bad Credit probability:  {bad_probability:.2%}")

# --------------------------------------------------
# SHAP contributions
# --------------------------------------------------

explanation_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": applicant_shap
})

explanation_df["abs_shap"] = (
    explanation_df["shap_value"].abs()
)

explanation_df = explanation_df.sort_values(
    "abs_shap",
    ascending=False
)

print("\nTop factors affecting this applicant:")
print(
    explanation_df.head(15)[
        ["feature", "shap_value"]
    ].to_string(index=False)
)

# --------------------------------------------------
# Separate positive / negative contributions
# --------------------------------------------------

positive = explanation_df[
    explanation_df["shap_value"] > 0
].head(10)

negative = explanation_df[
    explanation_df["shap_value"] < 0
].head(10)

print("\nFactors with POSITIVE SHAP contribution:")
print(
    positive[
        ["feature", "shap_value"]
    ].to_string(index=False)
)

print("\nFactors with NEGATIVE SHAP contribution:")
print(
    negative[
        ["feature", "shap_value"]
    ].to_string(index=False)
)

# --------------------------------------------------
# Save explanation
# --------------------------------------------------

explanation_df[
    ["feature", "shap_value"]
].to_csv(
    f"{BASE}/results/metrics/applicant_0_shap.csv",
    index=False
)

# --------------------------------------------------
# SHAP Waterfall Plot
# --------------------------------------------------

# Create Explanation object
base_value = explainer.expected_value

if isinstance(base_value, np.ndarray):
    base_value = base_value[0]

shap_explanation = shap.Explanation(
    values=applicant_shap,
    base_values=base_value,
    data=X_transformed[0],
    feature_names=feature_names
)

plt.figure()

shap.plots.waterfall(
    shap_explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{BASE}/results/figures/applicant_0_shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved:"
    f" {BASE}/results/metrics/applicant_0_shap.csv"
)

print(
    "Saved:"
    f" {BASE}/results/figures/applicant_0_shap_waterfall.png"
)

print("\nIndividual SHAP explanation completed successfully.")
