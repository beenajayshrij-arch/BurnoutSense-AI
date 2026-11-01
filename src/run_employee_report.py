import pandas as pd

from feature_engineering import load_clean_data, engineer_features
from train_model import compute_feature_importance, split_data, train_models
from evaluate import evaluate_all_models, compute_risk_thresholds, classify_risk
from shap_explain import explain_test_set
from action_mapping import get_action
from llm_reason import generate_reason


def build_employee_reports(n=5):
    """
    Trains the model, evaluates it, and returns a list of report dicts
    for the first `n` employees in the test set. Set n=None for all.
    """
    df = load_clean_data()
    df, X, X_scaled, y, feature_cols, encoders, scaler = engineer_features(df)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, X_scaled, y)
    models = train_models(X_train, X_train_scaled, y_train)

    results_df, best_model_name = evaluate_all_models(
        models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, X, X_scaled, y
    )
    low_thresh, high_thresh = compute_risk_thresholds(y)
    best_model = models[best_model_name]

    X_test_for_best = X_test_scaled if best_model_name == "Linear Regression" else X_test
    predictions = best_model.predict(X_test_for_best)
    risk_levels = [classify_risk(p, low_thresh, high_thresh) for p in predictions]

    _, top_features_list = explain_test_set(best_model, X_train, X_test, n_top_features=2)

    total = len(risk_levels) if n is None else min(n, len(risk_levels))
    reports = []
    for i in range(total):
        risk_level = risk_levels[i]
        employee_top_features = top_features_list[i]
        top_feature = employee_top_features[0][0]
        direction = employee_top_features[0][2]
        second_feature = employee_top_features[1][0] if len(employee_top_features) > 1 else top_feature

        action = get_action(top_feature)
        reason = generate_reason(risk_level, top_feature, second_feature)

        reports.append({
            "employee_id": i,
            "risk_level": risk_level,
            "top_driver": top_feature,
            "direction": direction,
            "action": action,
            "reason": reason,
        })

    return reports


if __name__ == "__main__":
    reports = build_employee_reports(n=5)
    for r in reports:
        print(r)