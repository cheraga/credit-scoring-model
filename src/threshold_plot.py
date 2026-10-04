
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/content/credit-scoring-model")

input_file = ROOT / "results" / "metrics" / "threshold_cv_summary.csv"
output_file = ROOT / "results" / "figures" / "threshold_metrics.png"

df = pd.read_csv(input_file)

plt.figure(figsize=(8, 5))

plt.plot(
    df["threshold"],
    df["accuracy_mean"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    df["threshold"],
    df["precision_mean"],
    marker="o",
    label="Precision"
)

plt.plot(
    df["threshold"],
    df["recall_mean"],
    marker="o",
    label="Recall"
)

plt.plot(
    df["threshold"],
    df["f1_mean"],
    marker="o",
    linewidth=3,
    label="F1 Score"
)

plt.axvline(
    x=0.40,
    linestyle="--",
    label="Selected threshold = 0.40"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Cross-Validated Score")
plt.title("Threshold Selection Using 5-Fold Cross-Validation")
plt.xticks(df["threshold"])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Saved: {output_file}")
