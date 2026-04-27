"""
DataForge · Anomaly Detection Module
Uses Isolation Forest + Z-Score + IQR to flag suspicious rows.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


TEAL   = "#00d4aa"
CORAL  = "#ff6b6b"
PURPLE = "#7c3aed"
BG     = "#0a0e1a"
PANEL  = "#111827"


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Run Isolation Forest on all numerical columns.
    Returns original df with added columns: anomaly_score, is_anomaly.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        return df.copy()

    X = df[num_cols].fillna(df[num_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    preds  = iso.fit_predict(X_scaled)
    scores = iso.score_samples(X_scaled)

    result = df.copy()
    result["anomaly_score"] = np.round(-scores, 4)   # higher = more anomalous
    result["is_anomaly"]    = (preds == -1).astype(int)
    return result


def zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Flag rows where any numerical column exceeds Z-score threshold."""
    from scipy import stats
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    result = df.copy()
    result["zscore_anomaly"] = 0

    for col in num_cols:
        if col in ("anomaly_score", "is_anomaly", "zscore_anomaly"):
            continue
        z = np.abs(stats.zscore(df[col].fillna(df[col].median())))
        result.loc[z > threshold, "zscore_anomaly"] = 1

    return result


def anomaly_summary(df_flagged: pd.DataFrame) -> dict:
    """Compute summary stats from flagged dataframe."""
    total   = len(df_flagged)
    if "is_anomaly" not in df_flagged.columns:
        return {}
    flagged = df_flagged["is_anomaly"].sum()
    pct     = round(flagged / total * 100, 2)

    # Per-column contribution
    num_cols = df_flagged.select_dtypes(include=np.number).columns.tolist()
    exclude  = {"anomaly_score","is_anomaly","zscore_anomaly"}
    num_cols = [c for c in num_cols if c not in exclude]

    anomaly_rows = df_flagged[df_flagged["is_anomaly"] == 1]
    col_stats = {}
    for col in num_cols:
        col_stats[col] = {
            "normal_mean":  round(df_flagged.loc[df_flagged["is_anomaly"]==0, col].mean(), 3),
            "anomaly_mean": round(anomaly_rows[col].mean(), 3) if len(anomaly_rows) > 0 else None,
        }

    return {
        "total_rows":    total,
        "anomaly_count": int(flagged),
        "anomaly_pct":   pct,
        "col_stats":     col_stats,
    }


def plot_anomaly_overview(df_flagged: pd.DataFrame, top_n: int = 4) -> plt.Figure:
    """Plot anomaly scores + scatter plots for top numerical features."""
    num_cols = [c for c in df_flagged.select_dtypes(include=np.number).columns
                if c not in ("anomaly_score","is_anomaly","zscore_anomaly")][:top_n]

    n   = len(num_cols)
    fig = plt.figure(figsize=(14, 4 + n * 2.5))
    fig.patch.set_facecolor(PANEL)

    gs = gridspec.GridSpec(n + 1, 2, figure=fig, hspace=0.5, wspace=0.35)

    colors_map = df_flagged["is_anomaly"].map({0: TEAL, 1: CORAL})

    # Top row: score distribution
    ax_score = fig.add_subplot(gs[0, :])
    ax_score.set_facecolor(BG)
    normal  = df_flagged.loc[df_flagged["is_anomaly"]==0, "anomaly_score"]
    anomaly = df_flagged.loc[df_flagged["is_anomaly"]==1, "anomaly_score"]
    ax_score.hist(normal,  bins=40, color=TEAL,  alpha=0.7, label="Normal",  edgecolor="none")
    ax_score.hist(anomaly, bins=40, color=CORAL, alpha=0.9, label="Anomaly", edgecolor="none")
    ax_score.set_title("Anomaly Score Distribution", color="white", fontweight="bold", pad=8)
    ax_score.set_xlabel("Anomaly Score (higher = more anomalous)", color="#94a3b8")
    ax_score.set_ylabel("Count", color="#94a3b8")
    ax_score.tick_params(colors="#94a3b8")
    ax_score.spines[:].set_color("#2d3748")
    ax_score.legend(framealpha=0.2, labelcolor="white")

    # Per-column scatter vs anomaly score
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i + 1, i % 2] if n > 1 else gs[i + 1, :])
        ax.set_facecolor(BG)
        sc = ax.scatter(
            df_flagged[col], df_flagged["anomaly_score"],
            c=df_flagged["is_anomaly"].map({0: TEAL, 1: CORAL}),
            alpha=0.5, s=12, linewidths=0
        )
        ax.set_xlabel(col, color="#94a3b8", fontsize=9)
        ax.set_ylabel("Anomaly Score", color="#94a3b8", fontsize=9)
        ax.set_title(f"{col} vs Anomaly Score", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.spines[:].set_color("#2d3748")

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=TEAL, label='Normal'),
                           Patch(facecolor=CORAL, label='Anomaly')]
        ax.legend(handles=legend_elements, framealpha=0.15, labelcolor="white", fontsize=8)

    return fig
