"""
feature_engineering.py
Creates derived features from raw shipment data to help the models
pick up on interactions between traffic, weather, and distance.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Traffic level as a numeric severity score (helps tree models split cleanly)
    traffic_score_map = {"Low": 1, "Medium": 2, "High": 3, "Severe": 4}
    df["traffic_score"] = df["traffic_level"].map(traffic_score_map)

    # Weather severity score
    weather_score_map = {
        "Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3, "Heavy Rain": 4, "Storm": 5
    }
    df["weather_score"] = df["weather_condition"].map(weather_score_map)

    # Route difficulty score
    route_score_map = {"Highway": 1, "Mixed": 2, "Rural Road": 3, "City Road": 4}
    df["route_score"] = df["route_type"].map(route_score_map)

    # Interaction features — these often matter more than the raw values alone
    df["traffic_distance_interaction"] = df["traffic_score"] * df["distance_km"]
    df["weather_traffic_interaction"] = df["weather_score"] * df["traffic_score"]

    # Dispatch hour (peak-hour dispatch tends to correlate with delay)
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["is_peak_hour"] = df["dispatch_hour"].apply(
        lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 20) else 0
    )

    # Distance buckets (short / medium / long haul)
    df["distance_category"] = pd.cut(
        df["distance_km"],
        bins=[0, 100, 500, 1000, np.inf],
        labels=["Short", "Medium", "Long", "Very Long"]
    )

    return df