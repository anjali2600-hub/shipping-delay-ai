"""
train_amazon.py
Trains and compares models on the labeled Amazon delivery data.
  Regression target:     Delivery_Time (minutes)
  Classification target: delay_status (On Time / Slight / Moderate / Severe)
Only the best model of each kind is saved, to keep the repo small.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

DATA_PATH = "data/processed/amazon_labeled.csv"
MODEL_DIR = "models/amazon"
LABELS = ["On Time", "Slight Delay", "Moderate Delay", "Severe Delay"]

NUMERIC = ["Agent_Age", "Agent_Rating", "distance_km", "order_hour",
           "day_of_week", "is_weekend", "is_peak_hour"]
CATEGORICAL = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
FEATURES = NUMERIC + CATEGORICAL


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y_reg = df["Delivery_Time"]
    y_clf = df["delay_status"]

    label_encoder = LabelEncoder().fit(LABELS)

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    yc_train_enc = label_encoder.transform(yc_train)
    yc_test_enc = label_encoder.transform(yc_test)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # ---------------- Regression ----------------
    reg_models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1),
    }

    print("\nTraining regression models...")
    reg_rows, reg_fitted = [], {}
    for name, model in reg_models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, yr_train)
        preds = pipe.predict(X_test)
        mse = mean_squared_error(yr_test, preds)
        reg_rows.append({
            "model": name,
            "MAE": mean_absolute_error(yr_test, preds),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(yr_test, preds),
        })
        reg_fitted[name] = pipe
        print(f"  done: {name}")

    reg_results = pd.DataFrame(reg_rows).sort_values("RMSE")
    best_reg = reg_results.iloc[0]["model"]

    print("\n" + "=" * 60)
    print("REGRESSION COMPARISON (predicting Delivery_Time in minutes)")
    print("=" * 60)
    print(reg_results.round(3).to_string(index=False))
    print(f"\nBest regression model: {best_reg} (lowest RMSE)")

    # ---------------- Classification ----------------
    clf_models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="mlogloss"),
    }
    sample_weights = compute_sample_weight("balanced", yc_train_enc)

    print("\nTraining classification models...")
    clf_rows, clf_fitted, clf_preds = [], {}, {}
    for name, model in clf_models.items():
        pipe = make_pipeline(model)
        if name == "xgboost":
            pipe.fit(X_train, yc_train_enc, model__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, yc_train_enc)
        preds = pipe.predict(X_test)
        clf_rows.append({
            "model": name,
            "Accuracy": accuracy_score(yc_test_enc, preds),
            "Precision": precision_score(yc_test_enc, preds, average="weighted", zero_division=0),
            "Recall": recall_score(yc_test_enc, preds, average="weighted", zero_division=0),
            "F1_weighted": f1_score(yc_test_enc, preds, average="weighted", zero_division=0),
            "F1_macro": f1_score(yc_test_enc, preds, average="macro", zero_division=0),
        })
        clf_fitted[name] = pipe
        clf_preds[name] = preds
        print(f"  done: {name}")

    clf_results = pd.DataFrame(clf_rows).sort_values("F1_macro", ascending=False)
    best_clf = clf_results.iloc[0]["model"]

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPARISON (predicting delay_status)")
    print("=" * 60)
    print(clf_results.round(3).to_string(index=False))
    print(f"\nBest classification model: {best_clf} (highest macro F1)")

    order = label_encoder.transform(LABELS)
    cm = confusion_matrix(yc_test_enc, clf_preds[best_clf], labels=order)
    print(f"\nConfusion matrix for {best_clf} (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    print(f"\nPer-class report for {best_clf}:")
    print(classification_report(
        yc_test_enc, clf_preds[best_clf], labels=order,
        target_names=LABELS, zero_division=0, digits=3))

    # ---------------- Save best models ----------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(reg_fitted[best_reg], f"{MODEL_DIR}/best_regressor.pkl", compress=3)
    joblib.dump(clf_fitted[best_clf], f"{MODEL_DIR}/best_classifier.pkl", compress=3)
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")

    info = {
        "best_regressor": best_reg,
        "best_classifier": best_clf,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "regression_results": reg_results.round(4).to_dict("records"),
        "classification_results": clf_results.round(4).to_dict("records"),
    }
    with open(f"{MODEL_DIR}/model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved best models and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()