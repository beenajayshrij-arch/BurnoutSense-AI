"""
Model evaluation and risk categorization for burnout prediction.

Steps covered:
- R2, MAE, RMSE on the test set for each model
- Train R2 vs Test R2 (overfitting check)
- 5-fold cross-validation score
- Select the best model based on test R2
- Tercile-based risk categorization (Low / Medium / High)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import load_clean_data, engineer_features
from train_model import compute_feature_importance, split_data, train_models


def evaluate_model(name, model, X_train, X_test, y_train, y_test, X_full, y_full):
    """
    Compute R2, MAE, RMSE on the test set, train vs test R2 (overfitting check),
    and 5-fold cross-validation score for a single model.
    Returns a dict of metrics.
    """
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    test_r2 = r2_score(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring="r2")

    print(f"\n[evaluate_model] === {name} ===")
    print(f"  Test R2:        {test_r2:.4f}")
    print(f"  Train R2:       {train_r2:.4f}  (train vs test gap: {train_r2 - test_r2:.4f})")
    print(f"  MAE:            {mae:.4f}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  5-fold CV R2:   mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    return {
        "name": name,
        "test_r2": test_r2,
        "train_r2": train_r2,
        "mae": mae,
        "rmse": rmse,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def evaluate_all_models(models, X_train, X_test, X_train_scaled, X_test_scaled,
                         y_train, y_test, X, X_scaled, y):
    """
    Evaluate all trained models and return a results DataFrame plus the
    name of the best model (highest test R2).
    """
    results = []

    results.append(evaluate_model(
        "Linear Regression", models["Linear Regression"],
        X_train_scaled, X_test_scaled, y_train, y_test, X_scaled, y
    ))

    results.append(evaluate_model(
        "Random Forest", models["Random Forest"],
        X_train, X_test, y_train, y_test, X, y
    ))
    results.append(evaluate_model(
        "XGBoost", models["XGBoost"],
        X_train, X_test, y_train, y_test, X, y
    ))

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False).reset_index(drop=True)
    print("\n[evaluate_all_models] Summary (sorted by Test R2):")
    print(results_df)

    best_model_name = results_df.iloc[0]["name"]
    print(f"\n[evaluate_all_models] Best model selected: {best_model_name}")

    return results_df, best_model_name


def compute_risk_thresholds(y):
    """
    Compute tercile thresholds (33rd and 66th percentile) from the actual
    Burn Rate distribution, used to classify Low / Medium / High risk.
    """
    low_thresh, high_thresh = y.quantile([0.33, 0.66])
    print(f"[compute_risk_thresholds] Tercile thresholds: "
          f"low<{low_thresh:.4f}, medium<{high_thresh:.4f}, high>={high_thresh:.4f}")
    return low_thresh, high_thresh


def classify_risk(burn_rate, low_thresh, high_thresh):
    """
    Classify a single burn rate value into 'Low', 'Medium', or 'High' risk
    using the given tercile thresholds.
    """
    if burn_rate < low_thresh:
        return "Low"
    elif burn_rate < high_thresh:
        return "Medium"
    else:
        return "High"


if __name__ == "__main__":
    df = load_clean_data()
    df, X, X_scaled, y, feature_cols, encoders, scaler = engineer_features(df)

    compute_feature_importance(X, y)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, X_scaled, y)
    models = train_models(X_train, X_train_scaled, y_train)

    results_df, best_model_name = evaluate_all_models(
        models, X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, X, X_scaled, y
    )

    low_thresh, high_thresh = compute_risk_thresholds(y)

    best_model = models[best_model_name]
    X_test_for_best = X_test_scaled if best_model_name == "Linear Regression" else X_test
    predictions = best_model.predict(X_test_for_best)
    risk_levels = [classify_risk(p, low_thresh, high_thresh) for p in predictions]

    risk_counts = pd.Series(risk_levels).value_counts()
    print(f"\n[main] Risk level distribution on test set predictions:")
    print(risk_counts)