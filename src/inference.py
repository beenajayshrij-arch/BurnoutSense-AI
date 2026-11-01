"""
Real-time inference for a single new employee.

Loads the trained model artifacts saved by pipeline.py (models/burnout_model.pkl)
and predicts burnout risk for one new employee's raw HR-entered data, WITHOUT
retraining anything. This is what lets HR add a brand-new employee and get an
instant risk score, SHAP explanation, action, and reason.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
import shap

from evaluate import classify_risk
from shap_explain import get_top_features
from action_mapping import get_combined_action
from llm_reason import generate_reason

MODEL_PATH = "models/burnout_model.pkl"


def load_artifacts(path=MODEL_PATH):
    """Load the saved model + encoders + scaler + thresholds from disk."""
    return joblib.load(path)


def predict_new_employee(raw_input, artifacts=None):
    """
    raw_input: dict with keys:
        'Gender'                 - 'Female' or 'Male'
        'Company Type'           - 'Product' or 'Service'
        'WFH Setup Available'    - 'Yes' or 'No'
        'Designation'            - int, 0-5
        'Resource Allocation'    - float
        'Mental Fatigue Score'   - float
        'Date of Joining'        - string, 'YYYY-MM-DD'

    Returns a dict: risk_score, risk_level, top_feature, reason, action
    """
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts["model"]
    model_name = artifacts["model_name"]
    scaler = artifacts["scaler"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]
    low_thresh = artifacts["low_thresh"]
    high_thresh = artifacts["high_thresh"]
    X_background = artifacts["X_train_background"]

    # Encode exactly the same way training data was encoded
    row = {
        "Gender": encoders["Gender"].transform([raw_input["Gender"]])[0],
        "Company Type": encoders["Company Type"].transform([raw_input["Company Type"]])[0],
        "WFH Setup Available": encoders["WFH Setup Available"].transform([raw_input["WFH Setup Available"]])[0],
        "Designation": raw_input["Designation"],
        "Resource Allocation": raw_input["Resource Allocation"],
        "Mental Fatigue Score": raw_input["Mental Fatigue Score"],
        "Days_of_Service": (pd.Timestamp.now() - pd.to_datetime(raw_input["Date of Joining"])).days,
    }

    X_new = pd.DataFrame([row])[feature_cols]
    X_new_for_model = pd.DataFrame(scaler.transform(X_new), columns=feature_cols) if model_name == "Linear Regression" else X_new

    # Predict
    risk_score = float(model.predict(X_new_for_model)[0])
    risk_level = classify_risk(risk_score, low_thresh, high_thresh)

    # SHAP explanation for just this one employee
    if model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_background)
    shap_values = explainer.shap_values(X_new_for_model)
    shap_row = pd.Series(shap_values[0], index=feature_cols)

    top_feats = get_top_features(shap_row, n=2)
    top_feature = top_feats[0][0]
    second_feature = top_feats[1][0] if len(top_feats) > 1 else top_feats[0][0]

    action = get_combined_action(top_feature, second_feature)
    reason_text = generate_reason(risk_level, top_feature, second_feature)
    reason = f"(Risk score: {risk_score:.2f}) {reason_text}"

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "top_feature": top_feature,
        "reason": reason,
        "action": action,
    }


if __name__ == "__main__":
    sample_input = {
        "Gender": "Male",
        "Company Type": "Service",
        "WFH Setup Available": "No",
        "Designation": 3,
        "Resource Allocation": 7.5,
        "Mental Fatigue Score": 8.2,
        "Date of Joining": "2021-03-15",
    }
    result = predict_new_employee(sample_input)
    print("[inference] Test prediction for a sample new employee:")
    for k, v in result.items():
        print(f"  {k}: {v}")