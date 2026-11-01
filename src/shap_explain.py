"""
SHAP explainability for the burnout prediction model.

Steps covered:
- Use shap.TreeExplainer for Random Forest / XGBoost (tree-based models),
  or shap.Explainer as a generic fallback for other model types.
- Compute per-feature SHAP values for each employee in the test set.
- get_top_features(): returns the top N features by absolute SHAP value,
  along with their sign (positive = risk-increasing, negative = risk-reducing).
"""
import numpy as np
import pandas as pd
import shap


def get_shap_explainer(model, X_train):
    """
    Build a SHAP explainer for the given model.
    Uses TreeExplainer for tree-based models (Random Forest, XGBoost),
    since it's exact and fast for these model types.
    Falls back to the generic shap.Explainer for anything else
    (e.g. Linear Regression), which picks an appropriate algorithm automatically.
    """
    model_type = type(model).__name__
    if model_type in ("RandomForestRegressor", "XGBRegressor"):
        explainer = shap.TreeExplainer(model)
        print(f"[get_shap_explainer] Using shap.TreeExplainer for {model_type}")
    else:
        explainer = shap.Explainer(model, X_train)
        print(f"[get_shap_explainer] Using generic shap.Explainer for {model_type}")
    return explainer


def compute_shap_values(explainer, X_test):
    """
    Compute per-feature SHAP values for every row in X_test.
    Returns a DataFrame of the same shape as X_test, with SHAP values
    instead of feature values.
    """
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)
    print(f"[compute_shap_values] Computed SHAP values with shape: {shap_df.shape}")
    return shap_df


def get_top_features(shap_values_row, n=2):
    """
    Given a single row of SHAP values (a Series indexed by feature name),
    return the top N features by absolute SHAP value, along with their sign.

    Returns a list of tuples: [(feature_name, shap_value, direction), ...]
    where direction is "risk-increasing" if shap_value > 0,
    or "risk-reducing" if shap_value < 0.
    """
    abs_sorted = shap_values_row.abs().sort_values(ascending=False)
    top_features = []
    for feature_name in abs_sorted.index[:n]:
        value = shap_values_row[feature_name]
        direction = "risk-increasing" if value > 0 else "risk-reducing"
        top_features.append((feature_name, value, direction))
    return top_features


def explain_test_set(model, X_train, X_test, n_top_features=2):
    """
    Full SHAP explainability pipeline: builds the explainer, computes SHAP
    values for the whole test set, and extracts the top N features for each
    row. Returns (shap_df, list_of_top_features_per_row).
    """
    explainer = get_shap_explainer(model, X_train)
    shap_df = compute_shap_values(explainer, X_test)

    all_top_features = []
    for idx in shap_df.index:
        top_feats = get_top_features(shap_df.loc[idx], n=n_top_features)
        all_top_features.append(top_feats)

    print(f"[explain_test_set] Extracted top {n_top_features} features for {len(all_top_features)} employees.")
    return shap_df, all_top_features


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from feature_engineering import load_clean_data, engineer_features
    from train_model import split_data, train_models
    from evaluate import evaluate_all_models

    df = load_clean_data()
    df, X, X_scaled, y, feature_cols, encoders, scaler = engineer_features(df)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, X_scaled, y)
    models = train_models(X_train, X_train_scaled, y_train)

    results_df, best_model_name = evaluate_all_models(
        models, X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, X, X_scaled, y
    )
    print(f"[main] Running SHAP explainability on best model: {best_model_name}")

    best_model = models[best_model_name]
    X_train_for_best = X_train_scaled if best_model_name == "Linear Regression" else X_train
    X_test_for_best = X_test_scaled if best_model_name == "Linear Regression" else X_test

    shap_df, all_top_features = explain_test_set(best_model, X_train_for_best, X_test_for_best)

    print("\n[main] Example: top features for the first 5 test employees:")
    for i, top_feats in enumerate(all_top_features[:5]):
        print(f"  Row {i}: {top_feats}")