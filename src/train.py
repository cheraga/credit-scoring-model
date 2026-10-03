from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from src.data_loader import load_data
from src.feature_engineering import add_features
from src.preprocessing import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"
PROCESSED_DIR = ROOT / "data" / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(X, classifier):
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():

    print("=" * 60)
    print("CREDIT SCORING MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    X, y = load_data()

    print("\nDataset shape:")
    print(X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    # ---------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------

    X = add_features(X)

    # ---------------------------------------------------------
    # 3. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------------------------------
    # 4. Cross-validation configuration
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
        "roc_auc": make_scorer(roc_auc_score),
    }

    # ---------------------------------------------------------
    # 5. Define models
    # ---------------------------------------------------------

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison_results = []

    # ---------------------------------------------------------
    # 6. Train baseline models
    # ---------------------------------------------------------

    for name, classifier in models.items():

        print("\n" + "-" * 60)
        print(f"Training: {name}")
        print("-" * 60)

        pipeline = build_pipeline(
            X_train,
            classifier
        )

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        row = {
            "model": name,
            "accuracy_mean": cv_results[
                "test_accuracy"
            ].mean(),
            "accuracy_std": cv_results[
                "test_accuracy"
            ].std(),

            "precision_mean": cv_results[
                "test_precision"
            ].mean(),

            "recall_mean": cv_results[
                "test_recall"
            ].mean(),

            "f1_mean": cv_results[
                "test_f1"
            ].mean(),

            "roc_auc_mean": cv_results[
                "test_roc_auc"
            ].mean(),
        }

        comparison_results.append(row)

        pipeline.fit(X_train, y_train)

        model_path = MODELS_DIR / f"{name}.joblib"

        joblib.dump(
            pipeline,
            model_path
        )

        print(f"Saved: {model_path}")

    # ---------------------------------------------------------
    # 7. Save model comparison
    # ---------------------------------------------------------

    comparison_df = pd.DataFrame(
        comparison_results
    )

    comparison_path = (
        METRICS_DIR / "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nSaved model comparison: "
        f"{comparison_path}"
    )

    # ---------------------------------------------------------
    # 8. XGBoost hyperparameter tuning
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 60)

    xgb_pipeline = build_pipeline(
        X_train,
        XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    )

    param_distributions = {
        "classifier__n_estimators": [
            200,
            300,
            400,
            600,
        ],

        "classifier__max_depth": [
            2,
            3,
            4,
            5,
            6,
        ],

        "classifier__learning_rate": [
            0.02,
            0.05,
            0.1,
        ],

        "classifier__subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "classifier__colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
    }

    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train
    )

    best_model = search.best_estimator_

    # ---------------------------------------------------------
    # 9. Save best model
    # ---------------------------------------------------------

    best_model_path = (
        MODELS_DIR
        / "best_credit_scoring_model.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path
    )

    print(
        f"\nBest model saved: "
        f"{best_model_path}"
    )

    # ---------------------------------------------------------
    # 10. Save best parameters
    # ---------------------------------------------------------

    best_params_path = (
        METRICS_DIR
        / "best_params.json"
    )

    with open(
        best_params_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            search.best_params_,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # 11. Save best CV score
    # ---------------------------------------------------------

    best_cv_path = (
        METRICS_DIR
        / "best_cv_score.json"
    )

    with open(
        best_cv_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "best_cv_roc_auc":
                    float(search.best_score_)
            },
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # 12. Save test set
    # ---------------------------------------------------------

    test_set = X_test.copy()
    test_set["credit_risk"] = y_test.values

    test_path = (
        PROCESSED_DIR
        / "test_set.csv"
    )

    test_set.to_csv(
        test_path,
        index=False
    )

    print(
        f"Test set saved: {test_path}"
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
