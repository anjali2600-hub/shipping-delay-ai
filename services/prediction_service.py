"""
prediction_service.py  (real-data version, Amazon Delivery Dataset)
Loads the trained models and predicts delivery time, delay minutes,
delay category, delay probability and risk level for one delivery.
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "amazon")
LABEL_CONFIG_PATH = os.path.join(BASE_DIR, "models", "label_config.json")

# Load everything once, when the app starts
regressor = joblib.load(os.path.join(MODEL_DIR, "best_regressor.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "best_classifier.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
    MODEL_INFO = json.load(f)
with open(LABEL_CONFIG_PATH) as f:
    LABEL_CONFIG = json.load(f)

NUMERIC = MODEL_INFO["numeric_features"]
CATEGORICAL = MODEL_INFO["categorical_features"]
FEATURES = NUMERIC + CATEGORICAL

# Values that exist in the dataset (the form will offer these)
OPTIONS = {
    "weather": ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"],
    "traffic": ["Low", "Medium", "High", "Jam"],
    "vehicle": ["motorcycle", "scooter", "van"],
    "area": ["Metropolitan", "Urban", "Semi-Urban", "Other"],
    "category": [
        "Apparel", "Books", "Clothing", "Cosmetics", "Electronics", "Grocery",
        "Home", "Jewelry", "Kitchen", "Outdoors", "Pet Supplies", "Shoes",
        "Skincare", "Snacks", "Sports", "Toys",
    ],
}


def _build_row(raw: dict) -> pd.DataFrame:
    """Turns the user's input into one row with the same columns as training."""
    order_time = raw.get("order_time")
    order_dt = pd.to_datetime(order_time) if order_time else datetime.now()
    hour = order_dt.hour
    dow = order_dt.weekday()

    row = {
        "Agent_Age": float(raw.get("agent_age", 30)),
        "Agent_Rating": float(raw.get("agent_rating", 4.7)),
        "distance_km": float(raw["distance_km"]),
        "order_hour": hour,
        "day_of_week": dow,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_peak_hour": 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0,
        "Weather": raw.get("weather_condition", "Sunny"),
        "Traffic": raw.get("traffic_level", "Medium"),
        "Vehicle": raw.get("vehicle", "motorcycle"),
        "Area": raw.get("area", "Metropolitan"),
        "Category": raw.get("category", "Electronics"),
    }
    return pd.DataFrame([row])[FEATURES]


def _format_minutes(total_minutes: float) -> str:
    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes % 60))
    if hours > 0:
        return f"{hours} hour(s) {minutes} minute(s)"
    return f"{minutes} minute(s)"
def predict_shipment(raw_input: dict) -> dict:
    """Main entry point: raw form input in, full prediction result out."""
    X = _build_row(raw_input)

    # Delivery time (regression)
    predicted_time = max(10.0, float(regressor.predict(X)[0]))

    # Delay = predicted time minus the expected time used when labelling
    distance = float(X["distance_km"].iloc[0])
    expected_time = LABEL_CONFIG["intercept"] + LABEL_CONFIG["slope"] * distance
    predicted_delay = max(0.0, predicted_time - expected_time)

    # Delay category and probabilities (classification)
    proba = classifier.predict_proba(X)[0]
    class_index = int(proba.argmax())
    delay_category = str(label_encoder.inverse_transform([class_index])[0])

    on_time_index = int(label_encoder.transform(["On Time"])[0])
    delay_probability = (1.0 - float(proba[on_time_index])) * 100
    category_confidence = float(proba.max()) * 100

    # Risk level: chance that the delay is 45+ minutes (Moderate or Severe)
    serious_indices = label_encoder.transform(["Moderate Delay", "Severe Delay"])
    serious_probability = float(sum(proba[i] for i in serious_indices))
    if delay_category == "Severe Delay" or serious_probability >= 0.60:
        risk_level = "HIGH"
    elif delay_category == "Moderate Delay" or serious_probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "predicted_delivery_time_minutes": round(predicted_time, 1),
        "predicted_delivery_time_formatted": _format_minutes(predicted_time),
        "expected_time_minutes": round(expected_time, 1),
        "predicted_delay_minutes": round(predicted_delay, 1),
        "delay_category": delay_category,
        "delay_probability_percent": round(delay_probability, 1),
        "category_confidence_percent": round(category_confidence, 1),
        "serious_delay_probability_percent": round(serious_probability * 100, 1),
        "risk_level": risk_level,
        "will_be_delayed": delay_category != "On Time",
        "models_used": {
            "regression": MODEL_INFO["best_regressor"],
            "classification": MODEL_INFO["best_classifier"],
        },
    }


if __name__ == "__main__":
    scenarios = {
        "Bad conditions (jam, cloudy, evening rush)": {
            "distance_km": 8.5,
            "weather_condition": "Cloudy",
            "traffic_level": "Jam",
            "vehicle": "motorcycle",
            "area": "Metropolitan",
            "category": "Electronics",
            "agent_age": 30,
            "agent_rating": 4.5,
            "order_time": "2026-10-19 18:30:00",
        },
        "Good conditions (low traffic, sunny, late morning)": {
            "distance_km": 3.0,
            "weather_condition": "Sunny",
            "traffic_level": "Low",
            "vehicle": "scooter",
            "area": "Urban",
            "category": "Books",
            "agent_age": 28,
            "agent_rating": 4.9,
            "order_time": "2026-10-19 11:00:00",
        },
    }

    for name, sample in scenarios.items():
        print(f"\n{name}")
        for key, value in predict_shipment(sample).items():
            print(f"  {key}: {value}")