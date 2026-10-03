
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


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

FIGURES_DIR = (
    ROOT
    / "results"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("Loading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    X_test = test.drop(
        columns=["credit_risk"]
    )

    preprocessor = (
        model.named_steps["preprocessor"]
    )

    classifier = (
        model.named_steps["classifier"]
    )

    X_transformed = (
        preprocessor.transform(X_test)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    if isinstance(
        shap_values,
        list
    ):
        values = shap_values[1]
    else:
        values = shap_values

    plt.figure()

    shap.summary_plot(
        values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "shap_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SHAP figure saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
