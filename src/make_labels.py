"""
make_labels.py
Builds the delay label for the Amazon data. The dataset has no promised
delivery time, so "expected time" is estimated from deliveries made in the
best conditions (low traffic, sunny weather) as a straight-line function of
distance.   delay = actual delivery time - expected time.
"""

import json
import os
import numpy as np
import pandas as pd

IN_PATH = "data/processed/amazon_clean.csv"
OUT_PATH = "data/processed/amazon_labeled.csv"
CONFIG_PATH = "models/label_config.json"

# Minutes later than expected. Early deliveries count as On Time.
THRESHOLDS = [15, 45, 90]
LABELS = ["On Time", "Slight Delay", "Moderate Delay", "Severe Delay"]

df = pd.read_csv(IN_PATH)
print(f"Rows: {len(df)}")

# 1. Expected time from best-condition deliveries
ideal = df[(df["Traffic"] == "Low") & (df["Weather"] == "Sunny")]
slope, intercept = np.polyfit(ideal["distance_km"], ideal["Delivery_Time"], 1)
print(f"\nBest-condition rows (Low traffic + Sunny): {len(ideal)}")
print(f"Expected time = {intercept:.1f} + {slope:.2f} * distance_km")

df["expected_time"] = (intercept + slope * df["distance_km"]).round(1)
df["delay_raw"] = (df["Delivery_Time"] - df["expected_time"]).round(1)
df["delay_minutes"] = df["delay_raw"].clip(lower=0)

print("\ndelay_raw (negative = faster than expected):")
print(df["delay_raw"].describe().round(1).to_string())

# 2. Delay categories
bins = [-np.inf] + THRESHOLDS + [np.inf]
df["delay_status"] = pd.cut(df["delay_raw"], bins=bins, labels=LABELS)

counts = df["delay_status"].value_counts().reindex(LABELS)
print(f"\nThresholds (minutes): On Time <= {THRESHOLDS[0]} | "
      f"Slight <= {THRESHOLDS[1]} | Moderate <= {THRESHOLDS[2]} | Severe > {THRESHOLDS[2]}")
print("\nDelay category counts:")
for label, n in counts.items():
    print(f"  {label}: {n} ({n / len(df) * 100:.1f}%)")

# 3. Quick sanity check: average delay by traffic and weather
print("\nAverage delay_raw by Traffic:")
print(df.groupby("Traffic")["delay_raw"].mean().round(1).sort_values().to_string())
print("\nAverage delay_raw by Weather:")
print(df.groupby("Weather")["delay_raw"].mean().round(1).sort_values().to_string())

# 4. Save labeled data + the settings the live app will need later
df.to_csv(OUT_PATH, index=False)

os.makedirs("models", exist_ok=True)
with open(CONFIG_PATH, "w") as f:
    json.dump(
        {
            "intercept": float(intercept),
            "slope": float(slope),
            "thresholds": THRESHOLDS,
            "labels": LABELS,
        },
        f,
        indent=2,
    )

print(f"\nSaved to {OUT_PATH}")
print(f"Saved label settings to {CONFIG_PATH}")