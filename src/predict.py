"""Load saved models and predict for a single property."""
from functools import lru_cache

import joblib
import pandas as pd

from src import config as C
from src.data_utils import add_features


@lru_cache(maxsize=1)
def load_models():
    classifier = joblib.load(C.MODELS_DIR / "classifier.joblib")
    regressor = joblib.load(C.MODELS_DIR / "regressor.joblib")
    return classifier, regressor


def predict_property(property_data: dict) -> dict:
    classifier, regressor = load_models()
    row = add_features(pd.DataFrame([property_data]))
    good_prob = float(classifier.predict_proba(row)[0, 1])
    current_price = float(property_data["Price_in_Lakhs"])
    future_price = float(regressor.predict(row)[0]) * current_price  # model predicts the ratio
    return {
        "good_investment": good_prob >= 0.5,
        "good_probability": good_prob,
        "current_price_lakhs": current_price,
        "future_price_lakhs": future_price,
        "expected_gain_pct": (future_price - current_price) / current_price * 100,
        "growth_rate_used": C.CITY_GROWTH_RATES.get(property_data["City"], C.BASE_ANNUAL_GROWTH),
    }
