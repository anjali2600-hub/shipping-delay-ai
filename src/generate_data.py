"""
generate_data.py
Generates a realistic synthetic shipping/delivery dataset for the
AI-Based Shipping Delay Classification and Mitigation System.

Delay-status thresholds are calibrated from the empirical distribution
of generated delay_minutes (percentile-based) rather than fixed arbitrary
cutoffs, to ensure a reasonably balanced spread across categories.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N_RECORDS = 3000

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Raipur", "Ahmedabad", "Jaipur"
]

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Heavy Rain", "Storm", "Fog"]
ROUTE_TYPES = ["Highway", "City Road", "Rural Road", "Mixed"]
SHIPMENT_TYPES = ["Standard", "Express", "Same-Day"]
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Severe"]

WEATHER_DELAY_FACTOR = {
    "Clear": 0, "Cloudy": 5, "Rain": 20, "Heavy Rain": 45, "Storm": 70, "Fog": 30
}
TRAFFIC_DELAY_FACTOR = {
    "Low": 0, "Medium": 15, "High": 40, "Severe": 80
}
ROUTE_DELAY_FACTOR = {
    "Highway": 0, "Mixed": 10, "City Road": 20, "Rural Road": 15
}


def random_dispatch_time():
    base = datetime(2025, 1, 1)
    offset_days = random.randint(0, 364)
    offset_hours = random.randint(6, 22)
    return base + timedelta(days=offset_days, hours=offset_hours)


def generate_row(shipment_id):
    origin = random.choice(CITIES)
    destination = random.choice([c for c in CITIES if c != origin])

    distance_km = round(np.random.uniform(20, 2000), 1)
    weather_condition = random.choice(WEATHER_CONDITIONS)
    traffic_level = random.choice(TRAFFIC_LEVELS)
    route_type = random.choice(ROUTE_TYPES)
    shipment_type = random.choice(SHIPMENT_TYPES)

    temperature = round(np.random.uniform(10, 45), 1)
    rainfall = 0.0
    if weather_condition in ("Rain", "Heavy Rain", "Storm"):
        rainfall = round(np.random.uniform(5, 80), 1)
    elif weather_condition == "Cloudy":
        rainfall = round(np.random.uniform(0, 5), 1)

    package_weight = round(np.random.uniform(0.5, 50), 1)
    historical_delay_rate = round(np.random.uniform(0.0, 0.6), 2)

    dispatch_time = random_dispatch_time()

    base_travel_minutes = (distance_km / 45) * 60

    weather_delay = WEATHER_DELAY_FACTOR[weather_condition] * np.random.uniform(0.7, 1.3)
    traffic_delay = TRAFFIC_DELAY_FACTOR[traffic_level] * np.random.uniform(0.7, 1.3)
    route_delay = ROUTE_DELAY_FACTOR[route_type] * np.random.uniform(0.7, 1.3)
    distance_delay = (distance_km / 100) * np.random.uniform(1.0, 3.0)
    history_delay = historical_delay_rate * 60 * np.random.uniform(0.5, 1.5)

    priority_reduction = {"Standard": 0, "Express": -5, "Same-Day": -10}[shipment_type]

    noise = np.random.normal(0, 10)

    delay_minutes = max(
        0,
        weather_delay + traffic_delay + route_delay + distance_delay + history_delay + priority_reduction + noise
    )
    delay_minutes = round(delay_minutes, 1)

    estimated_delivery_time = round(base_travel_minutes, 1)
    actual_delivery_time = round(base_travel_minutes + delay_minutes, 1)

    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "distance_km": distance_km,
        "weather_condition": weather_condition,
        "temperature": temperature,
        "rainfall": rainfall,
        "traffic_level": traffic_level,
        "route_type": route_type,
        "package_weight": package_weight,
        "shipment_type": shipment_type,
        "dispatch_time": dispatch_time.strftime("%Y-%m-%d %H:%M:%S"),
        "historical_delay_rate": historical_delay_rate,
        "estimated_delivery_time": estimated_delivery_time,
        "actual_delivery_time": actual_delivery_time,
        "delay_minutes": delay_minutes,
    }


def classify_delay(delay_minutes, t1, t2, t3):
    if delay_minutes <= t1:
        return "On Time"
    elif delay_minutes <= t2:
        return "Slight Delay"
    elif delay_minutes <= t3:
        return "Moderate Delay"
    else:
        return "Severe Delay"


def main():
    rows = [generate_row(i + 1) for i in range(N_RECORDS)]
    df = pd.DataFrame(rows)

    t1 = df["delay_minutes"].quantile(0.25)
    t2 = df["delay_minutes"].quantile(0.60)
    t3 = df["delay_minutes"].quantile(0.90)

    df["delay_status"] = df["delay_minutes"].apply(lambda d: classify_delay(d, t1, t2, t3))

    output_path = "data/sample_shipping_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} records -> {output_path}")
    print(f"\nCalibrated thresholds (minutes): On Time <= {t1:.1f} | "
          f"Slight <= {t2:.1f} | Moderate <= {t3:.1f} | Severe > {t3:.1f}")
    print("\nDelay status distribution:")
    print(df["delay_status"].value_counts())
    print("\nSample rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()