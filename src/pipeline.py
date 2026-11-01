
"""
End-to-end pipeline for Employee Burnout Risk Prediction System.

Steps:
1. Load and clean employee data.
2. Assign readable employee IDs.
3. Perform feature engineering.
4. Train and evaluate ML models.
5. Select the best model.
6. Generate burnout risk predictions.
7. Generate SHAP explanations.
8. Generate recommendations.
9. Save predictions and model artifacts.

Output:
data/burnout_predictions.csv
models/burnout_model.pkl
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

from data_cleaning import load_data, clean_data
from feature_engineering import load_clean_data, engineer_features
from train_model import split_data, train_models
from evaluate import (
    evaluate_all_models,
    compute_risk_thresholds,
    classify_risk
)
from shap_explain import (
    get_shap_explainer,
    compute_shap_values,
    get_top_features
)
from action_mapping import get_combined_action
from llm_reason import generate_reason


def run_pipeline():

    # --------------------------------------------------
    # STEP 1: LOAD AND CLEAN DATA
    # --------------------------------------------------

    print("[pipeline] Step 1: Loading and cleaning raw data...")

    df_raw = load_data()

    # Assign readable employee IDs
    # Example: EMP0001, EMP0002, EMP0003

    df_raw["Employee ID"] = [
        f"EMP{i + 1:04d}" for i in range(len(df_raw))
    ]

    # Clean employee dataset
    df_clean = clean_data(df_raw)

    # Save cleaned dataset
    df_clean.to_csv(
        "data/employee_burnout_clean.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # --------------------------------------------------

    print("[pipeline] Step 2: Engineering features...")

    df_clean = load_clean_data(
        "data/employee_burnout_clean.csv"
    )

    df_eng, X, X_scaled, y, feature_cols, encoders, scaler = (
        engineer_features(df_clean)
    )

    # --------------------------------------------------
    # STEP 3: TRAINING MODELS
    # --------------------------------------------------

    print("[pipeline] Step 3: Training models...")

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    ) = split_data(X, X_scaled, y)

    models = train_models(
        X_train,
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------
    # STEP 4: MODEL EVALUATION
    # --------------------------------------------------

    print("[pipeline] Step 4: Evaluating models...")

    results_df, best_model_name = evaluate_all_models(
        models,
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X,
        X_scaled,
        y
    )

    best_model = models[best_model_name]

    print(
        f"[pipeline] Best model selected: {best_model_name}"
    )

    # Select appropriate input data for the model

    X_train_for_best = (
        X_train_scaled
        if best_model_name == "Linear Regression"
        else X_train
    )

    X_test_for_best = (
        X_test_scaled
        if best_model_name == "Linear Regression"
        else X_test
    )

    # --------------------------------------------------
    # STEP 5: RISK THRESHOLDS AND MODEL SAVING
    # --------------------------------------------------

    low_thresh, high_thresh = compute_risk_thresholds(y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "scaler": scaler,
            "encoders": encoders,
            "feature_cols": feature_cols,
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
            "X_train_background": X_train_for_best.sample(
                min(200, len(X_train_for_best)),
                random_state=42
            ),
        },
        "models/burnout_model.pkl"
    )

    print(
        "[pipeline] Saved trained model artifacts "
        "to models/burnout_model.pkl"
    )

    # --------------------------------------------------
    # STEP 6: GENERATE PREDICTIONS
    # --------------------------------------------------

    print("[pipeline] Step 5: Generating predictions...")

    risk_scores = best_model.predict(X_test_for_best)

    risk_levels = [
        classify_risk(score, low_thresh, high_thresh)
        for score in risk_scores
    ]

    # --------------------------------------------------
    # STEP 7: SHAP EXPLANATIONS
    # --------------------------------------------------

    print(
        "[pipeline] Step 6: Computing SHAP explanations..."
    )

    explainer = get_shap_explainer(
        best_model,
        X_train_for_best
    )

    shap_df = compute_shap_values(
        explainer,
        X_test_for_best
    )

    # --------------------------------------------------
    # STEP 8: ACTIONS AND REASONS
    # --------------------------------------------------

    print(
        "[pipeline] Step 7: Mapping actions and "
        "generating explanations..."
    )

    records = []
    reason_cache = {}

    for i, idx in enumerate(X_test.index):

        # Get employee ID from the original indexed dataset
        employee_id = df_eng.loc[idx, "Employee ID"]

        # Identify top contributing features
        top_feats = get_top_features(
            shap_df.loc[idx],
            n=2
        )

        top_feature = top_feats[0][0]

        second_feature = (
            top_feats[1][0]
            if len(top_feats) > 1
            else top_feats[0][0]
        )

        risk_level = risk_levels[i]

        # Generate recommendations
        action = get_combined_action(
            top_feature,
            second_feature
        )

        # Cache repeated explanations
        cache_key = (
            risk_level,
            top_feature,
            second_feature
        )

        if cache_key not in reason_cache:

            reason_cache[cache_key] = generate_reason(
                risk_level,
                top_feature,
                second_feature
            )

        base_reason = reason_cache[cache_key]

        # Get feature values
        raw_top_val = X_test.loc[idx, top_feature]
        raw_second_val = X_test.loc[idx, second_feature]

        reason = (
            f"(Risk score: {risk_scores[i]:.2f}) "
            f"{base_reason} "
            f"Specifically, their {top_feature} was "
            f"{raw_top_val:.1f} and "
            f"{second_feature} was {raw_second_val:.1f}."
        )

        # Store prediction with permanent employee ID
        records.append(
            {
                "employee_id": employee_id,
                "risk_score": round(
                    float(risk_scores[i]), 4
                ),
                "risk_level": risk_level,
                "top_feature": top_feature,
                "reason": reason,
                "action": action,
            }
        )

        if (i + 1) % 500 == 0:

            print(
                f"[pipeline] Processed "
                f"{i + 1}/{len(X_test.index)} employees..."
            )

    print(
        f"[pipeline] Made {len(reason_cache)} unique "
        f"LLM calls instead of {len(X_test.index)}."
    )

    # --------------------------------------------------
    # STEP 9: SAVE PREDICTIONS
    # --------------------------------------------------

    output_df = pd.DataFrame(records)

    output_path = "data/burnout_predictions.csv"

    output_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n[pipeline] Done. Saved {len(output_df)} "
        f"predictions to {output_path}"
    )

    print("[pipeline] Risk level distribution:")

    print(
        output_df["risk_level"].value_counts()
    )

    print("[pipeline] Sample employee IDs:")

    print(
        output_df["employee_id"].head()
    )

    return output_df


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    run_pipeline()
