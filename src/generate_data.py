"""Generates a synthetic Employee Burnout dataset matching the Kaggle schema."""
import pandas as pd
import numpy as np

def generate_synthetic_data(n=22000, seed=42):
    """Generate synthetic employee burnout data with realistic distributions
    and ~1-2% MCAR missingness in Burn Rate, Mental Fatigue Score, and Resource Allocation."""
    rng = np.random.default_rng(seed)

    employee_id = [f"EMP{i:06d}" for i in range(1, n + 1)]
    start_dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 9, size=n), unit="D"
    )
    gender = rng.choice(["Male", "Female"], size=n)
    company_type = rng.choice(["Service", "Product"], size=n)
    wfh = rng.choice(["Yes", "No"], size=n)
    designation = rng.integers(0, 6, size=n)

    resource_allocation = np.clip(rng.normal(5.5, 2.0, size=n), 1, 10)
    mental_fatigue = np.clip(rng.normal(5.0, 1.8, size=n), 0, 10)

    burn_rate = (
        0.35
        + 0.04 * mental_fatigue
        + 0.02 * resource_allocation
        + 0.01 * designation
        + rng.normal(0, 0.05, size=n)
    )
    burn_rate = np.clip(burn_rate, 0, 1)

    df = pd.DataFrame({
        "Employee ID": employee_id,
        "Date of Joining": start_dates,
        "Gender": gender,
        "Company Type": company_type,
        "WFH Setup Available": wfh,
        "Designation": designation,
        "Resource Allocation": resource_allocation,
        "Mental Fatigue Score": mental_fatigue,
        "Burn Rate": burn_rate,
    })

    # Inject ~1-2% MCAR missingness
    for col in ["Burn Rate", "Mental Fatigue Score", "Resource Allocation"]:
        missing_frac = rng.uniform(0.01, 0.02)
        missing_idx = rng.choice(df.index, size=int(len(df) * missing_frac), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/employee_burnout.csv", index=False)
    print(f"Generated dataset with shape: {df.shape}")
    print(df.head())