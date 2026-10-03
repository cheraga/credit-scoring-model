from ucimlrepo import fetch_ucirepo


DATASET_ID = 144


FEATURE_NAMES = [
    "checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "maintenance_people",
    "telephone",
    "foreign_worker",
]


def load_data():

    dataset = fetch_ucirepo(
        id=DATASET_ID
    )

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    X.columns = FEATURE_NAMES

    target = y.iloc[:, 0].astype(int).map({
        1: 0,
        2: 1
    })

    target.name = "credit_risk"

    return X, target


if __name__ == "__main__":

    X, y = load_data()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())
