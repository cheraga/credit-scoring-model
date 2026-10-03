from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "best_credit_scoring_model.joblib"
)

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "test_set.csv"
)

METRICS_DIR = (
    ROOT
    / "results"
    / "metrics"
)

FIGURES_DIR = (
    ROOT
    / "results"
    / "figures"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("Loading model...")

    model = joblib.load(
        MODEL_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    X_test = test.drop(
        columns=["credit_risk"]
    )

    y_test = test["credit_risk"]

    # Predictions
    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    metrics = {
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

        "roc_auc": roc_auc_score(
            y_test,
            y_prob
        ),
    }

    print("\nTest Metrics")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    # Save metrics
    metrics_path = (
        METRICS_DIR
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_path = (
        METRICS_DIR
        / "classification_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay(
        confusion_matrix=cm
    ).plot(
        ax=ax
    )

    ax.set_title(
        "Credit Scoring Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {metrics['roc_auc']:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Credit Scoring ROC Curve"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "roc_curve.png",
        dpi=300
    )

    plt.close(fig)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
