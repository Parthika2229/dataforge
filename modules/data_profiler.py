"""
DataForge · Deep Data Profiler
Generates comprehensive statistical profile for every column.
"""
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a rich per-column profile DataFrame covering:
    dtype, missing, unique, mean, std, min, max, skewness, kurtosis,
    mode, entropy, IQR, CV (coefficient of variation).
    """
    rows = []
    for col in df.columns:
        s    = df[col]
        dtype = str(s.dtype)
        n     = len(s)
        n_miss = s.isnull().sum()
        n_uniq = s.nunique()

        row = {
            "Column":     col,
            "dtype":      dtype,
            "Count":      n,
            "Missing":    n_miss,
            "Missing %":  round(n_miss / n * 100, 1),
            "Unique":     n_uniq,
            "Unique %":   round(n_uniq / n * 100, 1),
        }

        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean) > 0:
                row["Mean"]      = round(clean.mean(), 4)
                row["Std"]       = round(clean.std(), 4)
                row["Min"]       = round(clean.min(), 4)
                row["25%"]       = round(clean.quantile(0.25), 4)
                row["Median"]    = round(clean.median(), 4)
                row["75%"]       = round(clean.quantile(0.75), 4)
                row["Max"]       = round(clean.max(), 4)
                row["IQR"]       = round(clean.quantile(0.75) - clean.quantile(0.25), 4)
                row["Skewness"]  = round(float(scipy_stats.skew(clean)), 4)
                row["Kurtosis"]  = round(float(scipy_stats.kurtosis(clean)), 4)
                cv = (clean.std() / clean.mean() * 100) if clean.mean() != 0 else None
                row["CV %"]      = round(cv, 2) if cv is not None else None
                # Outlier count (IQR method)
                Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
                IQR    = Q3 - Q1
                n_out  = ((clean < Q1 - 1.5*IQR) | (clean > Q3 + 1.5*IQR)).sum()
                row["Outliers"]  = int(n_out)
                row["Outliers %"]= round(n_out / n * 100, 1)
                # Distribution shape hint
                skew = abs(row["Skewness"])
                if skew < 0.5:
                    row["Shape"] = "Normal"
                elif skew < 1.0:
                    row["Shape"] = "Mild skew"
                else:
                    row["Shape"] = "Heavy skew"
            else:
                for k in ["Mean","Std","Min","25%","Median","75%","Max","IQR",
                          "Skewness","Kurtosis","CV %","Outliers","Outliers %","Shape"]:
                    row[k] = None
        else:
            # Categorical columns
            mode_val = s.mode()
            row["Mode"]       = str(mode_val.iloc[0]) if len(mode_val) > 0 else None
            row["Mode freq %"]= round(s.value_counts(normalize=True).iloc[0] * 100, 1) \
                                if len(s.dropna()) > 0 else None
            # Shannon entropy
            vc = s.value_counts(normalize=True)
            entropy = -np.sum(vc * np.log2(vc + 1e-10))
            row["Entropy"]    = round(float(entropy), 3)

        rows.append(row)

    return pd.DataFrame(rows).set_index("Column")


def flag_issues(profile_df: pd.DataFrame) -> list[dict]:
    """Scan profile for potential data quality issues."""
    issues = []
    for col, row in profile_df.iterrows():
        miss_pct = row.get("Missing %", 0) or 0
        if miss_pct > 70:
            issues.append({"col": col, "level": "error",
                           "msg": f"`{col}` has {miss_pct}% missing — consider dropping"})
        elif miss_pct > 30:
            issues.append({"col": col, "level": "warn",
                           "msg": f"`{col}` has {miss_pct}% missing — imputation needed"})

        uniq_pct = row.get("Unique %", 0) or 0
        if uniq_pct == 100 and row.get("dtype", "") == "object":
            issues.append({"col": col, "level": "warn",
                           "msg": f"`{col}` has all unique values — likely an ID column"})

        if row.get("Unique", 0) == 1:
            issues.append({"col": col, "level": "error",
                           "msg": f"`{col}` is constant — zero predictive value"})

        out_pct = row.get("Outliers %", 0) or 0
        if out_pct > 15:
            issues.append({"col": col, "level": "warn",
                           "msg": f"`{col}` has {out_pct}% outliers — check for data errors"})

        shape = row.get("Shape", "")
        if shape == "Heavy skew":
            issues.append({"col": col, "level": "info",
                           "msg": f"`{col}` is heavily skewed — use median imputation"})

    return issues
