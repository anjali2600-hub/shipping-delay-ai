"""
evaluate.py
Evaluates all trained regression and classification models on the
held-out test set, prints comparison tables, and saves the best
model of each type as best_model_regression.pkl / best_model_classification.pkl
(and a combined best_model.pkl for convenience).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from data_preprocessing import prepare_datasets

MODELS_DIR = "models"

REGRESSION_MODELS = ["linear_regression", "random_forest_reg", "xgboost_reg"]
CLASSIFICATION_MODELS = ["logistic_regression", "random_forest_clf", "xgboost_clf"]


def evaluate_regression(models_data, X_test, y_test):
    results = []
    for name in REGRESSION_MODELS:
        model = joblib.load(f"{MODELS_DIR}/{name}.pkl")
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results.append({
            "model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2
        })

    df = pd.DataFrame(results).sort_values("RMSE")
    return df


def evaluate_classification(X_test, y_test):
    label_encoder = joblib.load(f"{MODELS_DIR}/delay_status_encoder.pkl")
    y_test_encoded = label_encoder.transform(y_test)

    results = []
    confusion_matrices = {}

    for name in CLASSIFICATION_MODELS:
        model = joblib.load(f"{MODELS_DIR}/{name}.pkl")

        if name == "xgboost_clf":
            # xgboost was trained on encoded integer labels
            preds_encoded = model.predict(X_test)
            preds = label_encoder.inverse_transform(preds_encoded)
        else:
            preds = model.predict(X_test)  # trained on original string labels

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        results.append({
            "model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1
        })

        confusion_matrices[name] = (
            confusion_matrix(y_test, preds, labels=label_encoder.classes_),
            label_encoder.classes_
        )

    df = pd.DataFrame(results).sort_values("F1", ascending=False)
    return df, confusion_matrices


def main():
    print("Loading data and preparing test sets...")
    data = prepare_datasets()
    X_test = data["X_test"]
    y_reg_test = data["y_reg_test"]
    y_clf_test = data["y_clf_test"]

    print("\n" + "=" * 60)
    print("REGRESSION MODEL COMPARISON (predicting delay_minutes)")
    print("=" * 60)
    reg_results = evaluate_regression(data, X_test, y_reg_test)
    print(reg_results.to_string(index=False))

    best_reg_name = reg_results.iloc[0]["model"]
    print(f"\nBest regression model: {best_reg_name} (lowest RMSE)")

    print("\n" + "=" * 60)
    print("CLASSIFICATION MODEL COMPARISON (predicting delay_status)")
    print("=" * 60)
    clf_results, conf_matrices = evaluate_classification(X_test, y_clf_test)
    print(clf_results.to_string(index=False))

    best_clf_name = clf_results.iloc[0]["model"]
    print(f"\nBest classification model: {best_clf_name} (highest F1)")

    print(f"\nConfusion matrix for {best_clf_name}:")
    cm, labels = conf_matrices[best_clf_name]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    # Save the best model of each type
    best_reg_model = joblib.load(f"{MODELS_DIR}/{best_reg_name}.pkl")
    joblib.dump(best_reg_model, f"{MODELS_DIR}/best_model_regression.pkl")

    best_clf_model = joblib.load(f"{MODELS_DIR}/{best_clf_name}.pkl")
    joblib.dump(best_clf_model, f"{MODELS_DIR}/best_model_classification.pkl")

    # Save a small metadata file so predict.py/prediction_service.py know
    # which underlying model type was chosen (needed for xgboost label decoding)
    joblib.dump(
        {"regression": best_reg_name, "classification": best_clf_name},
        f"{MODELS_DIR}/best_model_info.pkl"
    )

    print(f"\nSaved best_model_regression.pkl ({best_reg_name})")
    print(f"Saved best_model_classification.pkl ({best_clf_name})")
    print("Saved best_model_info.pkl (tracks which model types were chosen)")


if __name__ == "__main__":
    main()