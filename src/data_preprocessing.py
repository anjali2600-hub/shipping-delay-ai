"""
data_preprocessing.py
Loads the raw shipment CSV, cleans it, encodes categorical variables,
and produces train/test splits ready for model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from feature_engineering import add_engineered_features

RAW_DATA_PATH = "data/sample_shipping_data.csv"
PROCESSED_DIR = "data/processed"

CATEGORICAL_COLS = [
    "origin", "destination", "weather_condition", "traffic_level",
    "route_type", "shipment_type", "distance_category"
]

DROP_COLS = ["shipment_id", "dispatch_time", "actual_delivery_time"]

TARGET_REGRESSION = "delay_minutes"
TARGET_CLASSIFICATION = "delay_status"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/label_encoders.pkl")

    return df, encoders


def prepare_datasets():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_raw_data()
    df = clean_data(df)
    df = add_engineered_features(df)
    df, encoders = encode_features(df)

    df.to_csv(f"{PROCESSED_DIR}/processed_shipping_data.csv", index=False)

    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
    ]

    X = df[feature_cols]
    y_reg = df[TARGET_REGRESSION]
    y_clf = df[TARGET_CLASSIFICATION]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    joblib.dump(feature_cols, "models/feature_columns.pkl")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_reg_train": y_reg_train, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_test": y_clf_test,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    data = prepare_datasets()
    print("Preprocessing complete.")
    print(f"Feature columns ({len(data['feature_cols'])}):")
    print(data["feature_cols"])
    print(f"\nTrain shape: {data['X_train'].shape}")
    print(f"Test shape: {data['X_test'].shape}")