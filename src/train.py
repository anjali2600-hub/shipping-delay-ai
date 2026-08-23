"""
train.py
Trains regression models (delay_minutes) and classification models
(delay_status), saving each trained model to models/.
"""

import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from data_preprocessing import prepare_datasets

MODELS_DIR = "models"


def train_regression_models(X_train, y_train):
    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["linear_regression"] = lr

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["random_forest_reg"] = rf

    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models["xgboost_reg"] = xgb

    return models


def train_classification_models(X_train, y_train):
    models = {}

    logreg = LogisticRegression(max_iter=1000, multi_class="multinomial")
    logreg.fit(X_train, y_train)
    models["logistic_regression"] = logreg

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["random_forest_clf"] = rf

    # XGBClassifier needs numeric labels, not strings -> encode them
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    joblib.dump(label_encoder, f"{MODELS_DIR}/delay_status_encoder.pkl")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train_encoded)
    models["xgboost_clf"] = xgb

    return models


def main():
    print("Loading and preparing data...")
    data = prepare_datasets()

    print("\nTraining regression models (predicting delay_minutes)...")
    reg_models = train_regression_models(data["X_train"], data["y_reg_train"])
    for name, model in reg_models.items():
        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")
        print(f"  Saved {name}.pkl")

    print("\nTraining classification models (predicting delay_status)...")
    clf_models = train_classification_models(data["X_train"], data["y_clf_train"])
    for name, model in clf_models.items():
        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")
        print(f"  Saved {name}.pkl")

    print("\nAll models trained and saved to models/")


if __name__ == "__main__":
    main()