import numpy as np
import pandas as pd


def add_features(X: pd.DataFrame):

    X = X.copy()

    X["credit_amount_per_month"] = (
        X["credit_amount"]
        / X["duration_months"].replace(
            0,
            np.nan
        )
    ).fillna(0)

    X["age_credit_ratio"] = (
        X["age"]
        / X["credit_amount"].replace(
            0,
            np.nan
        )
    )

    X["age_credit_ratio"] = (
        X["age_credit_ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    X["credit_duration_interaction"] = (
        X["credit_amount"]
        * X["duration_months"]
    )

    return X
