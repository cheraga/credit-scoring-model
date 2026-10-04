from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.feature_engineering import add_features
from src.preprocessing import build_preprocessor


ROOT = Path("/content/credit-scoring-model")

METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

N_SPLITS = 5
RANDOM_STATE = 42


def build_pipeline(X, classifier):
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def main():

    print("=" * 90)
    print("CROSS-VALIDATED THRESHOLD ANALYSIS")
    print("=" * 90)

    # 1. Load data
    X, y = load_data()

    print("\nOriginal dataset:")
    print(X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    # 2. Feature engineering
    X = add_features(X)

    print("\nAfter feature engineering:")
    print(X.shape)

    # 3. SAME train/test split used by train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTest set will remain untouched.")

    # 4. 5-fold CV ONLY on training set
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    # 5. Cross-validation
    for fold, (train_idx, validation_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1
    ):

        print("\n" + "-" * 70)
        print(f"Fold {fold}/{N_SPLITS}")
        print("-" * 70)

        X_fold_train = X_train.iloc[train_idx]
        X_fold_validation = X_train.iloc[validation_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_validation = y_train.iloc[validation_idx]

        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pipeline = build_pipeline(
            X_fold_train,
            classifier
        )

        pipeline.fit(
            X_fold_train,
            y_fold_train
        )

        probabilities = pipeline.predict_proba(
            X_fold_validation
        )[:, 1]

        print(
            f"Validation samples: {len(validation_idx)}"
        )

        for threshold in THRESHOLDS:

            metrics = evaluate_threshold(
                y_fold_validation,
                probabilities,
                threshold
            )

            fold_results.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    **metrics,
                }
            )

    # 6. Save fold results
    fold_results_df = pd.DataFrame(fold_results)

    fold_results_path = (
        METRICS_DIR / "threshold_cv_folds.csv"
    )

    fold_results_df.to_csv(
        fold_results_path,
        index=False
    )

    # 7. Mean and standard deviation
    summary = (
        fold_results_df
        .groupby("threshold")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),

            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),

            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),

            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )

    print("\n" + "=" * 90)
    print("CROSS-VALIDATION THRESHOLD RESULTS")
    print("=" * 90)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # 8. Best threshold by F1
    best_f1 = summary.loc[
        summary["f1_mean"].idxmax()
    ]

    print("\n" + "=" * 90)
    print("BEST THRESHOLD BY MEAN F1")
    print("=" * 90)

    print(best_f1.to_string())

    # 9. Best threshold by Recall
    best_recall = summary.loc[
        summary["recall_mean"].idxmax()
    ]

    print("\n" + "=" * 90)
    print("BEST THRESHOLD BY MEAN RECALL")
    print("=" * 90)

    print(best_recall.to_string())

    # 10. Save summary
    summary_path = (
        METRICS_DIR / "threshold_cv_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # 11. Save selected threshold
    selected_threshold = float(
        best_f1["threshold"]
    )

    threshold_info = pd.DataFrame(
        [{
            "selection_metric": "mean_f1",
            "selected_threshold": selected_threshold,

            "cv_accuracy_mean":
                float(best_f1["accuracy_mean"]),

            "cv_accuracy_std":
                float(best_f1["accuracy_std"]),

            "cv_precision_mean":
                float(best_f1["precision_mean"]),

            "cv_precision_std":
                float(best_f1["precision_std"]),

            "cv_recall_mean":
                float(best_f1["recall_mean"]),

            "cv_recall_std":
                float(best_f1["recall_std"]),

            "cv_f1_mean":
                float(best_f1["f1_mean"]),

            "cv_f1_std":
                float(best_f1["f1_std"]),
        }]
    )

    threshold_info_path = (
        METRICS_DIR / "selected_threshold.csv"
    )

    threshold_info.to_csv(
        threshold_info_path,
        index=False
    )

    print("\n" + "=" * 90)
    print("SELECTED THRESHOLD")
    print("=" * 90)

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Mean CV F1: "
        f"{best_f1['f1_mean']:.4f}"
    )

    print(
        f"Std CV F1: "
        f"{best_f1['f1_std']:.4f}"
    )

    print("\n" + "=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(fold_results_path)
    print(summary_path)
    print(threshold_info_path)

    print("\nTest set remains untouched.")
    print("Cross-validated threshold analysis completed.")


if __name__ == "__main__":
    main()
