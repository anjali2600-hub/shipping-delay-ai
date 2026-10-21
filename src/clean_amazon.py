"""
clean_amazon.py
Cleans the raw Amazon delivery dataset and adds basic features
(distance, time features). Saves data/processed/amazon_clean.csv
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/amazon_delivery.csv"
OUT_PATH = "data/processed/amazon_clean.csv"


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two lat/long points."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def in_india(lat, lon):
    """Rough bounding box of India."""
    return lat.between(6, 38) & lon.between(68, 98)


def main():
    df = pd.read_csv(RAW_PATH)
    n_start = len(df)
    print(f"Rows at start: {n_start}")

    # 1. Strip spaces from text columns; turn the text 'NaN' into real missing values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "NaN": np.nan, "": np.nan})

    df["Area"] = df["Area"].replace({"Metropolitian": "Metropolitan"})
    df["Vehicle"] = df["Vehicle"].str.lower()

    df = df.drop_duplicates(subset="Order_ID")

    # 2. Missing Weather / Traffic: drop these rows (small share of the data)
    before = len(df)
    df = df.dropna(subset=["Weather", "Traffic"])
    print(f"Dropped rows with missing Weather/Traffic: {before - len(df)}")

    # 3. Agent rating: valid range is 1-5, raw max is 6.0
    bad_rating = int((df["Agent_Rating"] > 5).sum())
    df.loc[df["Agent_Rating"] > 5, "Agent_Rating"] = np.nan
    df["Agent_Rating"] = df["Agent_Rating"].fillna(df["Agent_Rating"].median())
    print(f"Ratings above 5 treated as missing and filled with median: {bad_rating}")

    # 4. Coordinates: keep only points inside India's rough bounding box
    valid = in_india(df["Store_Latitude"], df["Store_Longitude"]) & in_india(
        df["Drop_Latitude"], df["Drop_Longitude"]
    )
    print(f"Dropped rows with invalid coordinates: {int((~valid).sum())}")
    df = df[valid].copy()

    # 5. Distance between store and drop point
    df["distance_km"] = haversine_km(
        df["Store_Latitude"], df["Store_Longitude"],
        df["Drop_Latitude"], df["Drop_Longitude"],
    ).round(2)

    # 6. Time features
    order_dt = pd.to_datetime(df["Order_Date"] + " " + df["Order_Time"], errors="coerce")
    pickup_dt = pd.to_datetime(df["Order_Date"] + " " + df["Pickup_Time"], errors="coerce")
    ok = order_dt.notna() & pickup_dt.notna()
    print(f"Dropped rows with unreadable date/time: {int((~ok).sum())}")
    df = df[ok].copy()
    order_dt = order_dt[ok]
    pickup_dt = pickup_dt[ok]

    df["order_hour"] = order_dt.dt.hour
    df["day_of_week"] = order_dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["order_hour"].apply(
        lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 21) else 0
    )

    # Minutes between order and pickup (add 24h if pickup is after midnight)
    prep = (pickup_dt - order_dt).dt.total_seconds() / 60
    df["prep_minutes"] = prep.where(prep >= 0, prep + 1440)

    keep = [
        "Order_ID", "Agent_Age", "Agent_Rating", "Weather", "Traffic",
        "Vehicle", "Area", "Category", "distance_km", "order_hour",
        "day_of_week", "is_weekend", "is_peak_hour", "prep_minutes",
        "Delivery_Time",
    ]
    df = df[keep]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nRows at end: {len(df)} ({len(df) / n_start * 100:.1f}% of original)")
    print(f"Saved to {OUT_PATH}")

    print("\nDistance (km):")
    print(df["distance_km"].describe().round(2).to_string())
    print("\nDelivery_Time (minutes):")
    print(df["Delivery_Time"].describe().round(1).to_string())
    print("\nprep_minutes:")
    print(df["prep_minutes"].describe().round(1).to_string())

    for col in ["Weather", "Traffic", "Vehicle", "Area"]:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())

    print("\nCategory:", sorted(df["Category"].unique()))


if __name__ == "__main__":
    main()