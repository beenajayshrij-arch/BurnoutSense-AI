"""
Feature engineering for the Employee Burnout dataset.

Steps covered:
- Label-encode Gender, Company Type, WFH Setup Available (binary 0/1)
- Leave Designation as a raw ordinal integer (no encoding)
- Derive Days_of_Service from Date of Joining
- Build two parallel feature sets:
    - unscaled features (for Random Forest / XGBoost)
    - scaled features (for Linear Regression only, via StandardScaler)
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


import os

def load_clean_data(path=None):
    """Load the cleaned employee burnout CSV into a DataFrame."""
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "data", "employee_burnout_clean.csv")
    df = pd.read_csv(path)
    print(f"[load_clean_data] Loaded cleaned data with shape: {df.shape}")
    return df


def encode_categoricals(df):
    """
    Label-encode Gender, Company Type, and WFH Setup Available into 0/1.
    Designation is intentionally left untouched since it is already an
    ordinal integer (0-5) and should not be one-hot or label encoded.
    """
    df = df.copy()
    encoders = {}
    for col in ["Gender", "Company Type", "WFH Setup Available"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"[encode_categoricals] Encoded '{col}': {list(le.classes_)} -> {list(range(len(le.classes_)))}")
    return df, encoders


def add_days_of_service(df):
    """
    Derive Days_of_Service as the number of days between Date of Joining
    and the current date.
    """
    df = df.copy()
    df["Days_of_Service"] = (
        pd.Timestamp.now() - pd.to_datetime(df["Date of Joining"])
    ).dt.days
    print(f"[add_days_of_service] Added 'Days_of_Service' column. "
          f"Range: {df['Days_of_Service'].min()} to {df['Days_of_Service'].max()}")
    return df


def build_feature_sets(df):
    """
    Build the feature matrix X and target y.

    Returns:
        X (unscaled)      - for Random Forest / XGBoost
        X_scaled          - StandardScaler-transformed copy, for Linear Regression only
        y                 - target (Burn Rate)
        feature_cols      - list of feature column names used
        scaler            - fitted StandardScaler (kept for later use/inference)
    """
    feature_cols = [
        "Gender",
        "Company Type",
        "WFH Setup Available",
        "Designation",
        "Resource Allocation",
        "Mental Fatigue Score",
        "Days_of_Service",
    ]

    X = df[feature_cols].copy()
    y = df["Burn Rate"].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols,
        index=X.index,
    )

    print(f"[build_feature_sets] Unscaled feature set shape: {X.shape}")
    print(f"[build_feature_sets] Scaled feature set shape (Linear Regression only): {X_scaled.shape}")
    print(f"[build_feature_sets] Target shape: {y.shape}")

    return X, X_scaled, y, feature_cols, scaler


def engineer_features(df):
    """
    Full feature engineering pipeline: encode categoricals, add Days_of_Service,
    and build both unscaled and scaled feature sets.
    """
    df, encoders = encode_categoricals(df)
    df = add_days_of_service(df)
    X, X_scaled, y, feature_cols, scaler = build_feature_sets(df)
    return df, X, X_scaled, y, feature_cols, encoders, scaler


if __name__ == "__main__":
    df = load_clean_data()
    df, X, X_scaled, y, feature_cols, encoders, scaler = engineer_features(df)
    print("[main] Feature engineering complete.")
    print(f"[main] Final engineered dataframe shape: {df.shape}")
    df.to_csv("data/employee_burnout_engineered.csv", index=False)
    print("[main] Saved engineered data to data/employee_burnout_engineered.csv")