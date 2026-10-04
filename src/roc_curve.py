import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

BASE = "/content/credit-scoring-model"

model = joblib.load(
    f"{BASE}/models/best_credit_scoring_model.joblib"
)

test_df = pd.read_csv(
    f"{BASE}/data/processed/test_set.csv"
)

X_test = test_df.drop(columns=["credit_risk"])
y_test = test_df["credit_risk"]

# Probability for positive class
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Credit Scoring - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{BASE}/results/figures/roc_curve.png",
    dpi=300
)

plt.show()

print(f"ROC-AUC: {auc:.4f}")
print(f"Saved: {BASE}/results/figures/roc_curve.png")
