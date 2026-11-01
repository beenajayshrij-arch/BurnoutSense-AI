"""
Data understanding and cleaning for the Employee Burnout dataset.

Steps covered:
- Null counts per column
- MCAR check (does missingness in a column correlate with other features?)
- Skewness check on Burn Rate (log transform only if skew > 1)
- Correlation matrix + multicollinearity flag (> 0.6)
- Median imputation for Mental Fatigue Score / Resource Allocation
- Drop rows with missing target (Burn Rate) — never impute the target
"""
import pandas as pd
import numpy as np


def load_data(path="data/employee_burnout.csv"):
    """Load the raw employee burnout CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[load_data] Loaded raw data with shape: {df.shape}")
    return df


def check_nulls(df):
    """Print null counts per column and return the null-count Series."""
    nulls = df.isnull().sum()
    print("[check_nulls] Null counts per column:")
    print(nulls)
    return nulls


def check_mcar(df, cols_with_missing):
    """
    For each column with missing values, check whether the *fact* that a value
    is missing correlates with other numeric features. If correlations are all
    low (~near 0), this supports the missingness being MCAR (missing completely
    at random) rather than systematically related to other variables.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("[check_mcar] Checking correlation between missingness indicator and numeric features:")
    for col in cols_with_missing:
        indicator = df[col].isnull().astype(int)
        print(f"  -- Missingness in '{col}' vs other numeric columns --")
        for other_col in numeric_cols:
            if other_col == col:
                continue
            valid = df[other_col].notnull()
            if valid.sum() > 1:
                corr = np.corrcoef(indicator[valid], df.loc[valid, other_col].fillna(df[other_col].median()))[0, 1]
                flag = "  <-- possible non-random pattern" if abs(corr) > 0.1 else ""
                print(f"     corr(missing_{col}, {other_col}) = {corr:.4f}{flag}")


def check_skew(df, target_col="Burn Rate"):
    """
    Compute skewness of the target variable. Only recommend a log transform
    if skew > 1 (strong right skew). Returns the skew value.
    """
    skew_val = df[target_col].skew()
    print(f"[check_skew] Skewness of '{target_col}': {skew_val:.4f}")
    if skew_val > 1:
        print("  -> Skew > 1: a log transform would be recommended.")
    else:
        print("  -> Skew <= 1: no log transform needed.")
    return skew_val


def check_correlations(df, threshold=0.6):
    """
    Compute the full correlation matrix for numeric columns and print any
    feature pairs with |correlation| > threshold, flagged for multicollinearity.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    print("[check_correlations] Full correlation matrix:")
    print(corr_matrix)

    print(f"[check_correlations] Feature pairs with |correlation| > {threshold}:")
    flagged = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                flagged.append((cols[i], cols[j], val))
                print(f"  {cols[i]} <-> {cols[j]}: {val:.4f}")
    if not flagged:
        print("  None found.")
    return corr_matrix, flagged


def clean_data(df):
    """
    Clean the dataset:
    - Median-impute Mental Fatigue Score and Resource Allocation
      (median chosen over mean because these features can contain outliers,
       which would skew a mean-based imputation; verify visually with a boxplot).
    - Drop rows where Burn Rate (target) is missing — the target is never imputed,
      since imputing labels would inject fabricated ground truth into training/eval.
    """
    df = df.copy()

    for col in ["Mental Fatigue Score", "Resource Allocation"]:
        median_val = df[col].median()
        n_missing = df[col].isnull().sum()
        df[col] = df[col].fillna(median_val)
        print(f"[clean_data] Imputed {n_missing} missing values in '{col}' with median={median_val:.4f}")

    before = df.shape[0]
    df = df.dropna(subset=["Burn Rate"])
    after = df.shape[0]
    print(f"[clean_data] Dropped {before - after} rows with missing target 'Burn Rate'. "
          f"Shape now: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_data()
    check_nulls(df)
    check_mcar(df, ["Mental Fatigue Score", "Resource Allocation", "Burn Rate"])
    check_skew(df)
    check_correlations(df)
    df_clean = clean_data(df)
    print(f"[main] Final cleaned shape: {df_clean.shape}")
    df_clean.to_csv("data/employee_burnout_clean.csv", index=False)
    print("[main] Saved cleaned data to data/employee_burnout_clean.csv")