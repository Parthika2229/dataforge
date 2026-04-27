"""
InsOps · Automated Data Cleaning Engine
Replicates InsOps-style intelligent data transformation.
"""
import pandas as pd
import numpy as np
import re


def standardize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Lowercase + replace spaces/special chars with underscores."""
    old_cols = df.columns.tolist()
    new_cols = [
        re.sub(r"[^a-z0-9_]", "_", col.lower().strip().replace(" ", "_"))
        for col in old_cols
    ]
    # Deduplicate
    seen = {}
    final_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)
    df.columns = final_cols
    renamed = [(o, n) for o, n in zip(old_cols, final_cols) if o != n]
    report = [f"Renamed column `{o}` → `{n}`" for o, n in renamed]
    return df, report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    report = [f"Removed {removed} duplicate row(s)."] if removed else ["No duplicate rows found."]
    return df, report


def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> tuple[pd.DataFrame, list]:
    """
    Numerical  → mean / median / drop
    Categorical → mode imputation
    """
    report = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Numerical
    for col in num_cols:
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue
        if strategy == "drop":
            df = df.dropna(subset=[col])
            report.append(f"Dropped {n_miss} rows with missing `{col}`.")
        elif strategy == "median":
            val = df[col].median()
            df[col] = df[col].fillna(val)
            report.append(f"Imputed {n_miss} missing values in `{col}` with median ({val:.4g}).")
        else:  # mean
            val = df[col].mean()
            df[col] = df[col].fillna(val)
            report.append(f"Imputed {n_miss} missing values in `{col}` with mean ({val:.4g}).")

    # Categorical
    for col in cat_cols:
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue
        mode_val = df[col].mode()
        if len(mode_val) == 0:
            df[col] = df[col].fillna("unknown")
            report.append(f"Imputed {n_miss} missing in `{col}` with 'unknown' (no mode).")
        else:
            df[col] = df[col].fillna(mode_val[0])
            report.append(f"Imputed {n_miss} missing in `{col}` with mode '{mode_val[0]}'.")

    if not report:
        report.append("No missing values to impute.")
    return df, report


def cap_outliers_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Winsorize numerical columns at 1.5 * IQR."""
    report = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_out:
            df[col] = df[col].clip(lower, upper)
            report.append(f"Capped {n_out} outliers in `{col}` to [{lower:.4g}, {upper:.4g}].")
    if not report:
        report.append("No outliers detected (IQR method).")
    return df, report


def one_hot_encode(df: pd.DataFrame, max_cardinality: int = 20) -> tuple[pd.DataFrame, list]:
    """One-hot encode categorical columns with ≤ max_cardinality unique values."""
    report = []
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    to_encode = []
    for col in cat_cols:
        n = df[col].nunique()
        if n <= max_cardinality:
            to_encode.append(col)
        else:
            report.append(f"Skipped encoding `{col}` — high cardinality ({n} unique values).")

    if to_encode:
        df = pd.get_dummies(df, columns=to_encode, drop_first=False, dtype=int)
        report.append(f"One-hot encoded: {', '.join(to_encode)}.")
    else:
        report.append("No low-cardinality categorical columns to encode.")

    return df, report


def clean_pipeline(
    df: pd.DataFrame,
    remove_dupes: bool = True,
    num_strategy: str = "mean",
    handle_outliers: bool = False,
    encode_cat: bool = True,
    max_cardinality: int = 20,
) -> tuple[pd.DataFrame, list]:
    """
    Master cleaning pipeline — runs all steps in order.
    Returns (cleaned_df, full_report_list).
    """
    full_report = []

    # 1. Standardize column names
    df, r = standardize_column_names(df)
    full_report.extend(r)

    # 2. Remove duplicates
    if remove_dupes:
        df, r = remove_duplicates(df)
        full_report.extend(r)

    # 3. Impute missing values
    df, r = impute_missing(df, strategy=num_strategy)
    full_report.extend(r)

    # 4. Cap outliers
    if handle_outliers:
        df, r = cap_outliers_iqr(df)
        full_report.extend(r)

    # 5. One-hot encode
    if encode_cat:
        df, r = one_hot_encode(df, max_cardinality=max_cardinality)
        full_report.extend(r)

    return df, full_report
