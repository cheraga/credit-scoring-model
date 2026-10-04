import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BASE = "/content/credit-scoring-model"

# Load model
model = joblib.load(
    f"{BASE}/models/best_credit_scoring_model.joblib"
)

# Load test data
test_df = pd.read_csv(
    f"{BASE}/data/processed/test_set.csv"
)

# Last column is assumed to be target
X_test = test_df.drop(columns=["credit_risk"])
y_test = test_df["credit_risk"]

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Good Credit", "Bad Credit"]
)

disp.plot()
plt.title("Credit Scoring - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{BASE}/results/figures/confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    f"Saved: {BASE}/results/figures/confusion_matrix.png"
)
