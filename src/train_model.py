"""
Feature selection, train/test split, and model training for burnout prediction.

Steps covered:
- Fit a preliminary RandomForestRegressor to extract feature importances
- Print a ranked feature importance table
- train_test_split(X, y, test_size=0.2, random_state=42)
- Train Linear Regression (on scaled features), Random Forest and XGBoost (on unscaled features)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import load_clean_data, engineer_features


def compute_feature_importance(X, y):
    """
    Fit a preliminary RandomForestRegressor and extract feature importances.
    Prints a ranked table of features by importance and returns it as a DataFrame.
    """
    prelim_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    prelim_rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": prelim_rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("[compute_feature_importance] Ranked feature importances:")
    print(importance_df)
    return importance_df


def split_data(X, X_scaled, y, test_size=0.2, random_state=42):
    """
    Perform an 80/20 train/test split. Splits both the unscaled feature set
    (for RF/XGBoost) and the scaled feature set (for Linear Regression) using
    the same random_state so row alignment with y is consistent.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    print(f"[split_data] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"[split_data] y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    print(f"[split_data] X_train_scaled shape: {X_train_scaled.shape}, "
          f"X_test_scaled shape: {X_test_scaled.shape}")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def train_models(X_train, X_train_scaled, y_train):
    """
    Train three models on the training set only:
    - Linear Regression (on scaled features)
    - Random Forest (n_estimators=100, max_depth=10)
    - XGBoost (learning_rate=0.1, n_estimators=100, max_depth=6)

    Returns a dict of {model_name: fitted_model}.
    """
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train_scaled, y_train)
    print("[train_models] Linear Regression trained.")

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    print("[train_models] Random Forest trained.")

    xgb = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    print("[train_models] XGBoost trained.")

    return {
        "Linear Regression": lin_reg,
        "Random Forest": rf,
        "XGBoost": xgb,
    }


if __name__ == "__main__":
    df = load_clean_data()
    df, X, X_scaled, y, feature_cols, encoders, scaler = engineer_features(df)

    compute_feature_importance(X, y)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, X_scaled, y)

    models = train_models(X_train, X_train_scaled, y_train)

    print("[main] All models trained successfully:", list(models.keys()))