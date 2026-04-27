"""
InsOps · Visualization Module
Matplotlib/Seaborn charts for the Streamlit UI.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Consistent palette
PURPLE  = "#6c63ff"
ACCENT  = "#ff6584"
BG      = "#f7f7fa"
GRID    = "#e8e8f0"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor":  BG,
    "figure.facecolor": "white",
    "axes.edgecolor":  "#ccccdd",
    "axes.grid":       True,
    "grid.color":      GRID,
    "grid.linestyle":  "--",
    "grid.linewidth":  0.7,
})


def plot_distributions(df: pd.DataFrame, cols: list) -> plt.Figure:
    n = len(cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=30, color=PURPLE, edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        # Overlay KDE line
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(xs, kde(xs), color=ACCENT, linewidth=2)
            ax2.set_yticks([])
        except Exception:
            pass

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_categoricals(df: pd.DataFrame, cols: list) -> plt.Figure:
    n = len(cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    cmap = plt.get_cmap("Set2")

    for ax, col in zip(axes, cols):
        vc = df[col].value_counts().head(15)
        colors = [cmap(i / len(vc)) for i in range(len(vc))]
        bars = ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=colors, edgecolor="white")
        ax.set_title(col, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Count")
        # Value labels
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=8, color="#444")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Categorical Feature Value Counts", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list) -> plt.Figure:
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 0.8), max(5, len(num_cols) * 0.7)))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", linewidths=0.5,
        square=True, cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig
