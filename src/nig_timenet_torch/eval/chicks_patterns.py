"""PURPOSE: The TF automatic pattern discovery over the flat frame, per head: a shallow decision tree, random-forest
    importances, K-Means clusters and isolation-forest outliers, against the naive baseline.
TAGS: chicks, pattern discovery, decision tree rules, feature importance, random forest, kmeans clusters, isolation
    forest, anomalies, per head, tf parity
PITFALLS: needs scikit-learn (dev / reports extra), imported inside the functions; the tree's balanced hit rate is
    measured on the rows it was trained on (in-sample), its naive baseline is the majority-class share, so a head
    whose raw hit rate sits at the baseline has no pattern; the random forest uses at most ``rf_max_samples`` rows
    per tree (50 000, ~5x faster on 340 000 rows, importances within 0.015); the features are confidence,
    uncertainty, aleatoric, epistemic and the predicted move %, the uncertainties as a share of the entry price
    (``*_ret``) when present, and ``p_up`` only when it is finite for every row; a head with fewer than 20 usable
    rows is skipped and listed in ``skipped``; the figures (feature importance, PCA of the clusters) are drawn only
    when ``make_plots`` is true (the TF code drew them whenever ``verbose`` was).
SOURCE: crypto-signal-prediction ``evaluation/patterns.py`` (detect_success_failure_patterns and its helpers).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_outputs import DEFAULT_OUTPUT_DIR, save_or_print
from nig_timenet_torch.eval.chicks_text import Say

__all__ = ["detect_success_failure_patterns", "PATTERN_RF_MAX_SAMPLES", "MIN_PATTERN_ROWS"]

PATTERN_RF_MAX_SAMPLES = 50_000      # most rows each tree of the importance forest sees (None = all, the TF 'old' way)
MIN_PATTERN_ROWS = 20                # fewest usable rows of a head to look for patterns
_IMPORTANCE, _FEATURE = "الأهمية", "الخاصية"
_COUNT, _WIN = "عدد", "نسبة_نجاح"


def _balanced_hit_rate(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean over the classes present in ``y`` of the share of that class predicted right (balanced accuracy)."""
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def _features(work: pd.DataFrame, feature_cols) -> list:
    if feature_cols is not None:
        return list(feature_cols)
    cols = [c for c in ("confidence", "uncertainty", "aleatoric", "epistemic", "predicted_change_pct")
            if c in work.columns]
    return [f"{c}_ret" if c in ("uncertainty", "aleatoric", "epistemic") and f"{c}_ret" in work.columns
            and work[f"{c}_ret"].notna().all() else c for c in cols]


def _design(work: pd.DataFrame, feature_cols: list, top_n_assets: int) -> pd.DataFrame:
    if "asset" not in work.columns:
        return work[feature_cols].reset_index(drop=True)
    top = work["asset"].value_counts().nlargest(top_n_assets).index
    work["asset_grouped"] = np.where(work["asset"].isin(top), work["asset"], "أخرى")
    dummies = pd.get_dummies(work["asset_grouped"], prefix="asset")
    return pd.concat([work[feature_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def _fit_models(X: pd.DataFrame, y: pd.Series, work: pd.DataFrame, cluster_features: list, max_depth: int,
                n_clusters: int, rf_max_samples) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(10, len(X) // 50),
                                  class_weight="balanced", random_state=42).fit(X, y)
    rf_kw = {} if (rf_max_samples is None or len(X) <= rf_max_samples) else {"max_samples": int(rf_max_samples)}
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1,
                                **rf_kw).fit(X, y)
    importance = pd.DataFrame({_FEATURE: X.columns, _IMPORTANCE: rf.feature_importances_}).sort_values(
        _IMPORTANCE, ascending=False).reset_index(drop=True)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
        StandardScaler().fit_transform(work[cluster_features]))
    flags = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1).fit_predict(work[cluster_features])
    return {"tree_rules": export_text(tree, feature_names=list(X.columns)), "tree_accuracy": float(tree.score(X, y)),
            "tree_naive_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
            "tree_balanced_accuracy": _balanced_hit_rate(y, tree.predict(X)), "feature_importance": importance,
            "cluster_labels": labels, "anomaly_flags": flags}


def _print_single(res: dict, work: pd.DataFrame, cluster_features: list, n_clusters: int, head, tag: str, out_dir: str,
                  make_plots: bool, say: Say) -> None:
    from nig_timenet_torch.eval.chicks_plots import plot_clusters_2d, plot_feature_importance

    print("=" * 100)
    say("pat_title", h=say.text("pat_head_tag", h=head) if head else "")
    print("=" * 100)
    say("pat_tree", a=res["tree_accuracy"] * 100, n=res["tree_naive_baseline_accuracy"] * 100,
        b=res["tree_balanced_accuracy"] * 100)
    print("-" * 60)
    save_or_print(res["tree_rules"], f"decision_tree_rules{tag}", out_dir=out_dir, say=say)
    say("pat_importance")
    print("-" * 60)
    imp = res["feature_importance"]
    save_or_print(imp, f"feature_importance{tag}", out_dir=out_dir, say=say)
    say("pat_top_feature", f=imp.iloc[0][_FEATURE])
    if make_plots:
        plot_feature_importance(imp, save_dir=out_dir, filename=f"feature_importance{tag}")
    say("pat_clusters", k=n_clusters)
    print("-" * 60)
    summary = res["clusters"]
    save_or_print(summary.round(4), f"cluster_summary{tag}", out_dir=out_dir, say=say)
    best, worst = summary.iloc[0], summary.iloc[-1]
    say("pat_best", c=int(best["cluster"]), w=best[_WIN])
    say("pat_worst", c=int(worst["cluster"]), w=worst[_WIN])
    if make_plots:
        try:
            plot_clusters_2d(res["clustered_df"], cluster_features, save_dir=out_dir, filename=f"clusters_pca{tag}")
        except Exception as e:  # noqa: BLE001 - the TF code reports and goes on
            say("pat_cluster_fail", e=e)
    anomalies, flagged = res["anomalies"], res["_work_anom"]
    say("pat_anom_head", a=len(anomalies), n=len(flagged), p=len(anomalies) / len(flagged) * 100)
    print("-" * 60)
    if len(anomalies) > 0:
        normal = flagged[~flagged["is_anomaly"]]["correct"].mean() * 100
        anom = anomalies["correct"].mean() * 100
        say("pat_normal", a=normal)
        say("pat_anom", a=anom)
        save_or_print(anomalies, f"anomalous_predictions{tag}", out_dir=out_dir, say=say)
        say("pat_anom_worse" if anom < normal - 10 else "pat_anom_better" if anom > normal + 10 else "pat_anom_same")


def _detect_single(df: pd.DataFrame, feature_cols=None, max_tree_depth: int = 4, n_clusters: int = 4,
                   top_n_assets_onehot: int = 8, verbose: bool = True, head=None, rf_max_samples=PATTERN_RF_MAX_SAMPLES,
                   out_dir: str = DEFAULT_OUTPUT_DIR, make_plots: bool = True, say: Say | None = None) -> dict:
    say = say or Say()
    work = df.copy()
    if "predicted_change_pct" not in work.columns and "pct_error" in work.columns:
        work["predicted_change_pct"] = 0.0
    auto = feature_cols is None
    cols = _features(work, feature_cols)
    work = work.dropna(subset=cols + ["correct"])
    if len(work) < MIN_PATTERN_ROWS:
        raise ValueError(say.text("pat_too_few"))
    if auto and "p_up" in work.columns and work["p_up"].notna().all():
        cols = cols + ["p_up"]
    tag = f"_{head}" if head else ""
    X, y = _design(work, cols, top_n_assets_onehot), work["correct"].astype(int).reset_index(drop=True)
    cluster_features = [c for c in cols if c in work.columns]
    fitted = _fit_models(X, y, work, cluster_features, max_tree_depth, n_clusters, rf_max_samples)
    clustered = work.copy()
    clustered["cluster"] = fitted.pop("cluster_labels")
    summary = clustered.groupby("cluster").agg(
        عدد=("correct", "size"), نسبة_نجاح=("correct", "mean"),
        **{f"متوسط_{c}": (c, "mean") for c in cluster_features}).reset_index()
    summary[_WIN] = (summary[_WIN] * 100).round(2)
    flagged = work.copy()
    flagged["is_anomaly"] = fitted.pop("anomaly_flags") == -1
    res = {"head": head, "n": int(len(work)), "features": list(cols), **fitted,
           "clusters": summary.sort_values(_WIN, ascending=False), "clustered_df": clustered,
           "anomalies": flagged[flagged["is_anomaly"]]}
    if verbose:
        _print_single({**res, "_work_anom": flagged}, work, cluster_features, n_clusters, head, tag, out_dir,
                      make_plots, say)
    return res


def _pattern_summary(per_head: dict) -> pd.DataFrame:
    """One row per head: the naive baseline (majority share) against the tree's raw and balanced hit rates."""
    rows = []
    for h, r in per_head.items():
        imp = r["feature_importance"]
        rows.append({"head": h, "n": r["n"], "naive_baseline_accuracy": r["tree_naive_baseline_accuracy"],
                     "tree_accuracy": r["tree_accuracy"], "tree_balanced_accuracy": r["tree_balanced_accuracy"],
                     "top_feature": imp.iloc[0][_FEATURE] if len(imp) else None, "features": ",".join(r["features"])})
    return pd.DataFrame(rows)


def detect_success_failure_patterns(df: pd.DataFrame, feature_cols=None, max_tree_depth: int = 4,
                                    n_clusters: int = 4, top_n_assets_onehot: int = 8, verbose: bool = True,
                                    per_head: bool = True, head_col: str = "target",
                                    rf_max_samples=PATTERN_RF_MAX_SAMPLES, out_dir: str = DEFAULT_OUTPUT_DIR,
                                    make_plots: bool = True, say: Say | None = None) -> dict:
    """Success / failure patterns per head (``per_head``) or over the pooled frame. Per head: ``mode``, ``per_head``
    ({head: result}), ``heads``, ``skipped`` ({head: reason}) and ``summary``; with a single head (or ``per_head``
    False) the result's own fields (``tree_rules``, ``feature_importance``, ``clusters``, ``clustered_df``,
    ``anomalies``, ``tree_*``) are also at the top level."""
    say = say or Say()
    kw = {"feature_cols": feature_cols, "max_tree_depth": max_tree_depth, "n_clusters": n_clusters,
          "top_n_assets_onehot": top_n_assets_onehot, "verbose": verbose, "rf_max_samples": rf_max_samples,
          "out_dir": out_dir, "make_plots": make_plots, "say": say}
    heads = list(pd.unique(df[head_col].dropna())) if head_col in df.columns else []
    if not per_head or len(heads) <= 1:
        sub = df if not heads or not per_head else df[df[head_col] == heads[0]]
        res = _detect_single(sub, **kw)
        name = str(heads[0]) if (per_head and heads) else "all"
        res["head"] = name
        out = {**res, "mode": "per_head" if per_head else "pooled", "summary": _pattern_summary({name: res}),
               "skipped": {}}
        if per_head:
            out.update(per_head={name: res}, heads=[name])
        return out
    per, skipped = {}, {}
    for h in heads:
        try:
            per[str(h)] = _detect_single(df[df[head_col] == h], head=str(h), **kw)
        except ValueError as e:
            skipped[str(h)] = str(e)
            if verbose:
                say("pat_head_skipped", h=h, e=e)
    if not per:
        raise ValueError(say.text("pat_none_left", s=skipped))
    summary = _pattern_summary(per)
    if verbose:
        print("=" * 100)
        say("pat_summary_title")
        print("=" * 100)
        print(summary.drop(columns=["features"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        say("pat_summary_note")
    return {"mode": "per_head", "per_head": per, "heads": list(per), "skipped": skipped, "summary": summary}
