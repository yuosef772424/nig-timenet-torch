"""PURPOSE: The figures of the TF chicks report with the same titles, axes, colours, sizes and file names: calibration,
    confidence vs error, errors, lag, importance, clusters, equity and batches.
TAGS: chicks, figures, matplotlib, calibration curve, confidence vs error, error distribution, equity curve, batch
    performance, lag correlation, feature importance, clusters pca, tf parity
PITFALLS: matplotlib is an optional dependency (extra ``plots``), imported inside each function WITHOUT forcing a
    backend (a notebook shows the figure, a headless run falls back to Agg and only saves); every figure is saved to
    ``<save_dir>/figures/<name>.png`` and closed; titles and axis labels are English on purpose (matplotlib cannot
    shape Arabic without ``arabic_reshaper``, which this project does not depend on), the Arabic column names of the
    batch / importance tables are only read, never drawn; a function returns None when there is nothing to draw;
    the file name of an equity curve comes from its title (spaces become ``_``).
SOURCE: crypto-signal-prediction ``evaluation/plots.py``.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_outputs import DEFAULT_OUTPUT_DIR, save_figure, show_figure

__all__ = ["plot_calibration_curve", "plot_confidence_vs_error", "plot_error_distribution", "plot_actual_vs_predicted",
           "plot_lag_correlation", "plot_confusion_matrix_heatmap", "plot_feature_importance", "plot_clusters_2d",
           "plot_equity_curve", "plot_batch_performance"]

_WIN, _IMPORTANCE, _FEATURE = "نسبة_نجاح", "الأهمية", "الخاصية"


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _finish(plt, fig, name: str, save_dir: str) -> str:
    plt.tight_layout()
    path = save_figure(fig, name, save_dir)
    show_figure(plt, fig)
    return path


def plot_calibration_curve(reliability_df: pd.DataFrame, group_val=None, group_col: str = "target",
                           title: str | None = None, save_dir: str = DEFAULT_OUTPUT_DIR) -> str | None:
    """Reliability diagram: predicted confidence against actual accuracy, one point per quantile bin."""
    df = reliability_df if group_val is None else reliability_df[reliability_df[group_col] == group_val]
    if df.empty:
        return None
    df = df.sort_values("avg_confidence")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.plot(df["avg_confidence"], df["avg_accuracy"], "o-", color="steelblue", label="Model")
    for _, row in df.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["avg_confidence"], row["avg_accuracy"]), fontsize=7, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(title or f"Calibration Curve{f' — {group_val}' if group_val else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, f"calibration_curve_{group_val or 'overall'}", save_dir)


def plot_confidence_vs_error(df: pd.DataFrame, target: str | None = None, error_col: str = "abs_error",
                             save_dir: str = DEFAULT_OUTPUT_DIR) -> str | None:
    """Confidence against error: a calibrated model shows an inverse relation (right green, wrong red, a trend line)."""
    work = df if target is None else df[df["target"] == target]
    work = work.dropna(subset=["confidence", error_col])
    if work.empty:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    right = work["correct"].astype(bool) if "correct" in work.columns else pd.Series(True, index=work.index)
    ax.scatter(work.loc[right, "confidence"], work.loc[right, error_col], s=10, alpha=0.4, color="seagreen",
               label="Correct")
    ax.scatter(work.loc[~right, "confidence"], work.loc[~right, error_col], s=10, alpha=0.4, color="crimson",
               label="Wrong")
    if work["confidence"].std() > 1e-9:
        z = np.polyfit(work["confidence"], work[error_col], 1)
        xs = np.linspace(work["confidence"].min(), work["confidence"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.7, label="Trend")
    ax.set_xlabel("Confidence")
    ax.set_ylabel(error_col)
    ax.set_title(f"Confidence vs Error{f' — {target}' if target else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, f"confidence_vs_error_{target or 'all'}", save_dir)


def plot_error_distribution(df: pd.DataFrame, target: str | None = None, error_col: str = "pct_error",
                            save_dir: str = DEFAULT_OUTPUT_DIR) -> str | None:
    """Histogram of the errors with their mean and median (fat tails or a bias show here)."""
    work = df if target is None else df[df["target"] == target]
    vals = work[error_col].dropna()
    if vals.empty:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(vals.mean(), color="crimson", ls="--", label=f"Mean={vals.mean():.3f}")
    ax.axvline(vals.median(), color="orange", ls="--", label=f"Median={vals.median():.3f}")
    ax.set_xlabel(error_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution{f' — {target}' if target else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, f"error_distribution_{target or 'all'}", save_dir)


def plot_actual_vs_predicted(y_actual, y_pred, y_naive=None, title: str = "Actual vs Predicted",
                             save_dir: str = DEFAULT_OUTPUT_DIR, filename: str | None = None) -> str:
    """Actual against predicted values over the steps, with an optional naive t-1 baseline."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_actual, c="limegreen", lw=2.2, alpha=0.85, label="Actual", zorder=5)
    ax.plot(y_pred, c="darkorange", lw=1.8, label="Predicted", alpha=0.9)
    if y_naive is not None:
        ax.plot(y_naive, c="gray", lw=1.3, ls="--", label="Naive Baseline (t-1)", alpha=0.7)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, filename or f"actual_vs_predicted_{title}".replace(" ", "_"), save_dir)


def plot_lag_correlation(lags, corrs, best_lag: int, title: str = "Cross-Correlation vs Lag",
                         save_dir: str = DEFAULT_OUTPUT_DIR, filename: str | None = None) -> str:
    """Correlation at each lag, the best lag in red (0 is healthy, another lag suggests late copying)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, corrs, color=["crimson" if lag == best_lag else "steelblue" for lag in lags])
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, filename or f"lag_correlation_{title}".replace(" ", "_"), save_dir)


def plot_confusion_matrix_heatmap(cm, class_names, title: str = "Confusion Matrix", save_dir: str = DEFAULT_OUTPUT_DIR,
                                  filename: str | None = None) -> str:
    """Heat map of a confusion matrix (categorical targets)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.2 * len(class_names) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(plt, fig, filename or f"confusion_matrix_{title}".replace(" ", "_"), save_dir)


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, save_dir: str = DEFAULT_OUTPUT_DIR,
                            filename: str = "feature_importance") -> str:
    """The random-forest importance of each feature in telling success from failure."""
    plt = _pyplot()
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(top))))
    ax.barh(top[_FEATURE], top[_IMPORTANCE], color="teal")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    return _finish(plt, fig, filename, save_dir)


def plot_clusters_2d(clustered_df: pd.DataFrame, feature_cols, save_dir: str = DEFAULT_OUTPUT_DIR,
                     filename: str = "clusters_pca") -> str:
    """The behaviour clusters (K-Means) projected on two PCA components."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(clustered_df[feature_cols])
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clustered_df["cluster"], cmap="tab10", s=15, alpha=0.6)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Behavioral Clusters (PCA Projection)", fontweight="bold")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    ax.grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, filename, save_dir)


def plot_equity_curve(equity, drawdown_pct, title: str = "Equity Curve", save_dir: str = DEFAULT_OUTPUT_DIR,
                      filename: str | None = None) -> str:
    """The equity curve and, below it, the drawdown in percent (the standard tearsheet pair)."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(equity, color="steelblue", lw=1.8)
    axes[0].set_title(title, fontweight="bold")
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[1].fill_between(range(len(drawdown_pct)), drawdown_pct, 0, color="crimson", alpha=0.5)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Trade #")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    return _finish(plt, fig, filename or f"equity_curve_{title}".replace(" ", "_"), save_dir)


def plot_batch_performance(batch_stats: pd.DataFrame, z_threshold: float = 1.5,
                           title: str = "Performance Across Batches", save_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """Win rate per consecutive batch, the unusual batches coloured (red below, green above)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(batch_stats["z_score"] <= -z_threshold, "crimson",
                      np.where(batch_stats["z_score"] >= z_threshold, "seagreen", "steelblue"))
    ax.bar(batch_stats["batch_id"], batch_stats[_WIN], color=colors)
    ax.axhline(batch_stats[_WIN].mean(), color="black", ls="--", alpha=0.6, label="Mean")
    ax.set_xlabel("Batch #")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    return _finish(plt, fig, "batch_performance", save_dir)
