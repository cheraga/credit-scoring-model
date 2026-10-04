
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

# --------------------------------------------------
# Detect pipeline / estimator
# --------------------------------------------------

print("Model type:")
print(type(model))

if hasattr(model, "named_steps"):

    print("\nPipeline steps:")
    print(model.named_steps)

    # Get preprocessing step
    preprocessing = None
    estimator = None

    for name, step in model.named_steps.items():

        if hasattr(step, "transform"):
            preprocessing = step

        if hasattr(step, "feature_importances_"):
            estimator = step

    if estimator is None:
        raise ValueError(
            "Could not find an estimator with feature_importances_."
        )

    # Transform features
    if preprocessing is not None:
        X_transformed = preprocessing.transform(X_test)

        # Get feature names after preprocessing
        try:
            feature_names = preprocessing.get_feature_names_out(
                X_test.columns
            )
        except Exception:
            feature_names = [
                f"feature_{i}"
                for i in range(X_transformed.shape[1])
            ]

    else:
        X_transformed = X_test
        feature_names = X_test.columns

else:

    estimator = model
    X_transformed = X_test

    if hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_
    else:
        feature_names = X_test.columns


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

if not hasattr(estimator, "feature_importances_"):
    raise ValueError(
        "The model does not provide feature_importances_."
    )

importance = estimator.feature_importances_

print("\nNumber of feature names:", len(feature_names))
print("Number of importances:", len(importance))

# Safety check
if len(feature_names) != len(importance):

    print("\nWARNING:")
    print("Feature names and importance lengths do not match.")

    feature_names = [
        f"feature_{i}"
        for i in range(len(importance))
    ]

# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

feature_importance.to_csv(
    f"{BASE}/results/metrics/feature_importance.csv",
    index=False
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

top_n = min(20, len(feature_importance))

plot_data = feature_importance.head(top_n).sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_data["feature"],
    plot_data["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    f"Top {top_n} Feature Importances - Credit Scoring"
)

plt.tight_layout()

plt.savefig(
    f"{BASE}/results/figures/feature_importance.png",
    dpi=300
)

plt.show()

print(
    f"\nSaved: "
    f"{BASE}/results/metrics/feature_importance.csv"
)

print(
    f"Saved: "
    f"{BASE}/results/figures/feature_importance.png"
)

print("\nFeature importance analysis completed successfully.")
