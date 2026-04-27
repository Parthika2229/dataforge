"""
InsOps · Smart Insights Engine
Generates automated data quality insights and a quality score.
"""
import pandas as pd
import numpy as np
from scipy import stats


def generate_insights(clean_df: pd.DataFrame, raw_df: pd.DataFrame) -> list[dict]:
    """
    Returns a list of insight dicts: {"msg": str, "level": "ok"|"warn"|"error"}
    """
    insights = []
    num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

    # 1. Missing value check on raw df
    missing_pct = raw_df.isnull().mean() * 100
    high_miss = missing_pct[missing_pct > 30]
    for col, pct in high_miss.items():
        insights.append({"msg": f"`{col}` had {pct:.1f}% missing values (imputed)", "level": "warn"})

    very_high = missing_pct[missing_pct > 70]
    for col in very_high.index:
        insights.append({"msg": f"`{col}` has >70% missing — consider dropping this column", "level": "error"})

    # 2. Constant columns
    for col in clean_df.columns:
        if clean_df[col].nunique() <= 1:
            insights.append({"msg": f"`{col}` is constant (1 unique value) — may not be useful", "level": "warn"})

    # 3. Outlier detection (Z-score)
    for col in num_cols:
        try:
            z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
            n_extreme = (z_scores > 3).sum()
            if n_extreme > 0:
                pct = n_extreme / len(clean_df) * 100
                level = "error" if pct > 10 else "warn"
                insights.append({"msg": f"`{col}` has {n_extreme} extreme outliers ({pct:.1f}% of rows)", "level": level})
        except Exception:
            pass

    # 4. Near-duplicate columns (high correlation)
    if len(num_cols) >= 2:
        try:
            corr = clean_df[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [(c, r, upper.loc[r, c]) for c in upper.columns for r in upper.index
                         if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.95]
            for c1, c2, v in high_corr[:3]:
                insights.append({"msg": f"`{c1}` & `{c2}` are highly correlated ({v:.2f}) — potential redundancy", "level": "warn"})
        except Exception:
            pass

    # 5. Class imbalance check (for low-cardinality columns in raw)
    for col in raw_df.select_dtypes(include="object").columns:
        vc = raw_df[col].value_counts(normalize=True)
        if len(vc) == 2 and vc.iloc[0] > 0.90:
            insights.append({"msg": f"`{col}` is heavily imbalanced ({vc.iloc[0]*100:.0f}% one class)", "level": "warn"})

    # 6. Positive signals
    remaining_miss = clean_df.isnull().sum().sum()
    if remaining_miss == 0:
        insights.append({"msg": "No missing values in cleaned dataset ✓", "level": "ok"})

    dupe_count = clean_df.duplicated().sum()
    if dupe_count == 0:
        insights.append({"msg": "No duplicate rows in cleaned dataset ✓", "level": "ok"})

    if len(num_cols) > 0:
        insights.append({"msg": f"{len(num_cols)} numerical feature(s) available for modelling ✓", "level": "ok"})

    return insights


def compute_quality_score(df: pd.DataFrame) -> int:
    """
    Compute a 0-100 data quality score based on:
    - Completeness  (40 pts)
    - Uniqueness    (20 pts)
    - Consistency   (20 pts) — column name standardization proxy
    - Validity      (20 pts) — no all-constant columns
    """
    score = 0

    # Completeness
    miss_pct = df.isnull().mean().mean()
    score += int((1 - miss_pct) * 40)

    # Uniqueness
    dup_pct = df.duplicated().sum() / max(len(df), 1)
    score += int((1 - dup_pct) * 20)

    # Consistency (snake_case names = all standardized)
    import re
    bad_names = sum(1 for c in df.columns if re.search(r"[^a-z0-9_]", c))
    consistency = 1 - bad_names / max(len(df.columns), 1)
    score += int(consistency * 20)

    # Validity (no all-constant columns)
    constant_cols = sum(1 for c in df.columns if df[c].nunique() <= 1)
    validity = 1 - constant_cols / max(len(df.columns), 1)
    score += int(validity * 20)

    return min(score, 100)
