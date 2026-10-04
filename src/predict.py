import joblib
import pandas as pd

MODEL_PATH = "models/credit_scoring_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_credit(applicant):

    df = pd.DataFrame([applicant])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    label = "Bad Credit" if prediction == 1 else "Good Credit"

    return {
        "prediction": label,
        "bad_credit_probability": float(probability)
    }
