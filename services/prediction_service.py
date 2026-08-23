"""
prediction_service.py
Loads the best trained models and produces a full prediction for a
single shipment, given raw (unencoded) user/form input.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime

MODELS_DIR = "models"

# --- Load everything once, at import time (not per-request) ---
label_encoders = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
delay_status_encoder = joblib.load(f"{MODELS_DIR}/delay_status_encoder.pkl")
best_model_info = joblib.load(f"{MODELS_DIR}/best_model_info.pkl")

best_regression_model = joblib.load(f"{MODELS_DIR}/best_model_regression.pkl")
best_classification_model = joblib.load(f"{MODELS_DIR}/best_model_classification.pkl")

REGRESSION_MODEL_NAME = best_model_info["regression"]
CLASSIFICATION_MODEL_NAME = best_model_info["classification"]

# Same score maps used in feature_engineering.py -- must stay identical
TRAFFIC_SCORE_MAP = {"Low": 1, "Medium": 2, "High": 3, "Severe": 4}
WEATHER_SCORE_MAP = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3, "Heavy Rain": 4, "Storm": 5}
ROUTE_SCORE_MAP = {"Highway": 1, "Mixed": 2, "Rural Road": 3, "City Road": 4}

RISK_LEVEL_MAP = {
    "On Time": "LOW",
    "Slight Delay": "LOW",
    "Moderate Delay": "MEDIUM",
    "Severe Delay": "HIGH",
}


def _safe_label_encode(column_name: str, value: str) -> int:
    """
    Encodes a categorical value using the saved training-time encoder.
    Falls back to the most common training-time class if the user
    submits a category the model has never seen (keeps the app from
    crashing on unexpected input).
    """
    encoder = label_encoders[column_name]
    value = str(value)
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    else:
        # Fallback: encode as the first known class (index 0)
        return int(encoder.transform([encoder.classes_[0]])[0])


def _build_feature_row(raw_input: dict) -> pd.DataFrame:
    """
    Takes raw user input and produces a single-row DataFrame with
    EXACTLY the same columns, in the same order, as the training data.
    """
    distance_km = float(raw_input["distance_km"])

    dispatch_time_str = raw_input.get("dispatch_time")
    if dispatch_time_str:
        dispatch_dt = pd.to_datetime(dispatch_time_str)
    else:
        dispatch_dt = datetime.now()

    dispatch_hour = dispatch_dt.hour
    is_peak_hour = 1 if (7 <= dispatch_hour <= 10 or 17 <= dispatch_hour <= 20) else 0

    if distance_km <= 100:
        distance_category = "Short"
    elif distance_km <= 500:
        distance_category = "Medium"
    elif distance_km <= 1000:
        distance_category = "Long"
    else:
        distance_category = "Very Long"

    traffic_level = raw_input["traffic_level"]
    weather_condition = raw_input["weather_condition"]
    route_type = raw_input["route_type"]

    traffic_score = TRAFFIC_SCORE_MAP.get(traffic_level, 2)
    weather_score = WEATHER_SCORE_MAP.get(weather_condition, 1)
    route_score = ROUTE_SCORE_MAP.get(route_type, 2)

    row = {
        "origin": _safe_label_encode("origin", raw_input["origin"]),
        "destination": _safe_label_encode("destination", raw_input["destination"]),
        "distance_km": distance_km,
        "weather_condition": _safe_label_encode("weather_condition", weather_condition),
        "temperature": float(raw_input.get("temperature", 25.0)),
        "rainfall": float(raw_input.get("rainfall", 0.0)),
        "traffic_level": _safe_label_encode("traffic_level", traffic_level),
        "route_type": _safe_label_encode("route_type", route_type),
        "package_weight": float(raw_input.get("package_weight", 5.0)),
        "shipment_type": _safe_label_encode("shipment_type", raw_input.get("shipment_type", "Standard")),
        "historical_delay_rate": float(raw_input.get("historical_delay_rate", 0.2)),
        "estimated_delivery_time": (distance_km / 45) * 60,  # same formula as training
        "traffic_score": traffic_score,
        "weather_score": weather_score,
        "route_score": route_score,
        "traffic_distance_interaction": traffic_score * distance_km,
        "weather_traffic_interaction": weather_score * traffic_score,
        "dispatch_hour": dispatch_hour,
        "is_peak_hour": is_peak_hour,
        "distance_category": _safe_label_encode("distance_category", distance_category),
    }

    df = pd.DataFrame([row])
    # Enforce exact column order used during training
    df = df[feature_columns]
    return df


def predict_shipment(raw_input: dict) -> dict:
    """
    Main entry point. Takes raw form input (dict) and returns a
    complete prediction result dict, ready to hand to the frontend.
    """
    X = _build_feature_row(raw_input)

    # --- Delivery time / delay minutes (regression) ---
    predicted_delay_minutes = float(best_regression_model.predict(X)[0])
    predicted_delay_minutes = max(0.0, predicted_delay_minutes)

    estimated_base_minutes = X["estimated_delivery_time"].iloc[0]
    predicted_total_minutes = estimated_base_minutes + predicted_delay_minutes

    # --- Delay category (classification) ---
    if CLASSIFICATION_MODEL_NAME == "xgboost_clf":
        pred_encoded = best_classification_model.predict(X)[0]
        delay_category = delay_status_encoder.inverse_transform([pred_encoded])[0]
    else:
        delay_category = best_classification_model.predict(X)[0]

    # --- Delay probability (confidence in the predicted category) ---
    delay_probability = None
    if hasattr(best_classification_model, "predict_proba"):
        proba = best_classification_model.predict_proba(X)[0]
        delay_probability = float(np.max(proba)) * 100  # % confidence

    risk_level = RISK_LEVEL_MAP.get(delay_category, "MEDIUM")

    def format_minutes(total_minutes: float) -> str:
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        if hours > 0:
            return f"{hours} hour(s) {minutes} minute(s)"
        return f"{minutes} minute(s)"

    return {
        "predicted_delivery_time_minutes": round(predicted_total_minutes, 1),
        "predicted_delivery_time_formatted": format_minutes(predicted_total_minutes),
        "predicted_delay_minutes": round(predicted_delay_minutes, 1),
        "delay_category": delay_category,
        "delay_probability_percent": round(delay_probability, 1) if delay_probability else None,
        "risk_level": risk_level,
        "will_be_delayed": delay_category != "On Time",
        "models_used": {
            "regression": REGRESSION_MODEL_NAME,
            "classification": CLASSIFICATION_MODEL_NAME,
        },
    }


if __name__ == "__main__":
    # Quick manual test
    sample_input = {
        "origin": "Delhi",
        "destination": "Mumbai",
        "distance_km": 1400,
        "weather_condition": "Heavy Rain",
        "temperature": 28,
        "rainfall": 35,
        "traffic_level": "High",
        "route_type": "City Road",
        "package_weight": 12,
        "shipment_type": "Express",
        "historical_delay_rate": 0.3,
        "dispatch_time": "2026-08-20 09:00:00",
    }

    result = predict_shipment(sample_input)
    print("Sample prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")