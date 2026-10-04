import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

BASE = "/content/credit-scoring-model"

model = joblib.load(
    f"{BASE}/models/best_credit_scoring_model.joblib"
)

test_df = pd.read_csv(
    f"{BASE}/data/processed/test_set.csv"
)

X_test = test_df.drop(columns=["credit_risk"])
y_test = test_df["credit_risk"]

y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(
    recall,
    precision,
    label=f"Average Precision = {ap:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Credit Scoring - Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{BASE}/results/figures/precision_recall_curve.png",
    dpi=300
)

plt.show()

print(f"Average Precision: {ap:.4f}")
print(
    f"Saved: {BASE}/results/figures/precision_recall_curve.png"
)
