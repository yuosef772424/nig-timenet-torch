"""Verification block d: does a class head beat what the last candle's own shape already says

PURPOSE: Block d of the integrated verification (old TF section 7-c): the high / low class heads against baselines
    built from the last candle (close position, wicks, the last input step), with a day-block bootstrap of the AUC gap.
TAGS: verification, candle baseline, close position, last step, auc gap, day bootstrap, class head
PITFALLS: baselines are fitted on TRAIN rows only and the best one is chosen on VAL; the verdict is TEST; the verdict
    needs a gain of >= ``min_auc_gain`` AUC when model and baseline are combined (a lower bound above 0 alone is
    meaningless at 100 000 rows); needs scikit-learn (the ``dev`` extra), imported inside the function; the bootstrap
    resamples whole days and calls ``stats.metrics.auc`` per resample: ``n_boot`` x 2 sorts of the test rows.
SOURCE: ported from crypto-signal-prediction workflow/candle_baseline.py (candle_baseline_report; verdict thresholds
    unchanged; the model scores come from the run's PredictionTable, the candle from last_candles).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.eval.verification_text import t as text
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics

__all__ = ["candle_baseline_report", "CANDLE_COLUMNS", "BASELINES"]

CANDLE_COLUMNS = ("cpos", "upper_wick", "lower_wick", "range_rel")
BASELINES = ("rule_cpos", "candle_gbm", "laststep_gbm")
PASS, FAIL, FLAT = "✅", "❌", "➖"     # verdict symbols (✅ / ❌ / ➖)
TS = LAST_COLUMN_INDEX["timestamp"]


def _candle_frame(dataset: dict, rows: np.ndarray, target: str, block: str = "day") -> pd.DataFrame:
    """Shape of each row's last candle (close position in the range, wicks, range, all relative) + the class label."""
    lc = np.asarray(dataset["last_candles"], dtype=np.float64)[rows]
    h, lo, c = (lc[:, LAST_COLUMN_INDEX[k]] for k in ("last_high", "last_low", "last_close"))
    label = np.asarray(dataset[head_spec("binary_classification", target).label_key])[rows]
    ts = pd.to_datetime(lc[:, TS])
    day = ts.values if block == "timestamp" else ts.values.astype("datetime64[D]")     # the bootstrap's resampled unit
    return pd.DataFrame({"cpos": (c - lo) / np.maximum(h - lo, 1e-12), "upper_wick": (h - c) / c,
                         "lower_wick": (c - lo) / c, "range_rel": (h - lo) / c, "y": (label > 0).astype(int),
                         "day": day})


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _auc_gap_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, days: np.ndarray, n_boot: int, seed: int):
    """(AUC(a) - AUC(b), lo, hi): the 95 % percentile interval from resampling whole DAYS (paired, same rows)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    gaps = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ix])) == 2:
            gaps.append(metrics.auc(y[ix], a[ix]) - metrics.auc(y[ix], b[ix]))
    point = metrics.auc(y, a) - metrics.auc(y, b)
    lo, hi = np.percentile(gaps, [2.5, 97.5]) if gaps else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)


def _fit_baselines(train: pd.DataFrame, x_train: np.ndarray, seed: int) -> dict:
    """{name: predict(frame, x_last) -> P(up)} fitted on the TRAIN rows (needs scikit-learn)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    cols, y = list(CANDLE_COLUMNS), train["y"].to_numpy()
    cpos = LogisticRegression().fit(train[["cpos"]].to_numpy(), y)
    gbm = {"depth": 3, "iters": 200, "lr": 0.1}
    candle = HistGradientBoostingClassifier(max_depth=gbm["depth"], max_iter=gbm["iters"], learning_rate=gbm["lr"],
                                            random_state=seed).fit(train[cols].to_numpy(), y)
    last = HistGradientBoostingClassifier(max_depth=gbm["depth"], max_iter=gbm["iters"], learning_rate=gbm["lr"],
                                          random_state=seed).fit(x_train, y)
    return {"rule_cpos": lambda f, x: cpos.predict_proba(f[["cpos"]].to_numpy())[:, 1],
            "candle_gbm": lambda f, x: candle.predict_proba(f[cols].to_numpy())[:, 1],
            "laststep_gbm": lambda f, x: last.predict_proba(x)[:, 1]}


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float((pred[pos].mean() + (1 - pred[neg]).mean()) / 2)


def _verdict(best: str, gap: tuple, comb: tuple, min_gain: float, lang: str) -> str:
    d, lo, hi = gap
    di, loi, hii = comb
    if loi >= min_gain:
        return f"{PASS} " + text(lang, "cv_pass", best=best, di=di, lo=loi, hi=hii)
    if d < 0 or hi < 0:
        return f"{FAIL} " + text(lang, "cv_fail", best=best, di=di, gain=min_gain)
    return f"{FLAT} " + text(lang, "cv_flat", best=best)


def _one_target(target: str, frames: dict, x_last: dict, preds: dict, scores_fn: dict, n_boot: int, seed: int,
                min_gain: float, lang: str) -> tuple[list, dict]:
    """The table rows and the TEST verdict of one target. ``frames`` / ``x_last`` / ``preds``: {"val" | "test": ...}."""
    scores = {part: {name: fn(frames[part], x_last[part]) for name, fn in scores_fn.items()} | preds[part]
              for part in ("val", "test")}
    rows = []
    for part, fr in frames.items():
        y = fr["y"].to_numpy()
        for name, s in scores[part].items():
            pred = (s >= 0.5).astype(int)
            rows.append({"target": target, "split": part, "score": name, "n": len(y), "auc": metrics.auc(y, s),
                         "accuracy": float((pred == y).mean()), "balanced_acc": _balanced_accuracy(y, pred),
                         "majority_baseline": max(y.mean(), 1 - y.mean())})
    y_te, days = frames["test"]["y"].to_numpy(), frames["test"]["day"].to_numpy()
    y_va = frames["val"]["y"].to_numpy()
    best = max(BASELINES, key=lambda k: metrics.auc(y_va, scores["val"][k]))          # chosen on val
    gap = _auc_gap_ci(y_te, scores["test"]["model_class"], scores["test"][best], days, n_boot, seed)

    def stack(part: str, with_model: bool) -> np.ndarray:
        cols = [_logit(scores[part][best])] + ([_logit(scores[part]["model_class"])] if with_model else [])
        return np.column_stack(cols)
    from sklearn.linear_model import LogisticRegression

    only_base = LogisticRegression().fit(stack("val", False), y_va)           # the combination is learned on VAL only
    combined = LogisticRegression().fit(stack("val", True), y_va)
    comb = _auc_gap_ci(y_te, combined.predict_proba(stack("test", True))[:, 1],
                       only_base.predict_proba(stack("test", False))[:, 1], days, n_boot, seed + 1)
    return rows, {"best_baseline": best, "auc_diff_vs_best": gap, "auc_gain_when_combined": comb,
                  "verdict": _verdict(best, gap, comb, min_gain, lang), "n_test_days": int(len(np.unique(days)))}


def candle_baseline_report(dataset: dict, idx: dict, tables: dict, *, targets, x_key: str, n_boot: int = 300,
                           seed: int = 0, max_train: int = 300_000, min_auc_gain: float = 0.005, lang: str = "en",
                           block: str = "day") -> dict:
    """``{"table": per target x split x score (auc, accuracy, balanced_acc, majority_baseline), "verdicts": {target:
    {best_baseline, auc_diff_vs_best, auc_gain_when_combined, verdict, n_test_days}}}`` for the class head of each
    ``targets`` entry. ``idx``: {"train" | "val" | "test": dataset rows}; ``tables``: {"val" | "test": the run's
    PredictionTable of those rows, in order}; ``x_key``: the ``X_<tf>`` of the base timeframe (its last step feeds
    the ``laststep_gbm`` baseline); ``lang``: the language of the verdict text; ``block``: what the bootstrap
    resamples, the UTC calendar "day" or every distinct bar "timestamp" (the old project's unit)."""
    x = np.asarray(dataset[x_key])
    sel = np.sort(np.random.default_rng(seed).choice(len(idx["train"]), size=min(max_train, len(idx["train"])),
                                                     replace=False))
    train_rows = np.asarray(idx["train"])[sel]
    x_last = {"train": np.asarray(x[train_rows, -1, :], dtype=np.float32),
              **{p: np.asarray(x[np.asarray(idx[p]), -1, :], dtype=np.float32) for p in ("val", "test")}}
    table_rows, verdicts = [], {}
    for t in targets:
        cls_key = head_spec("binary_classification", t).output_keys["logits"]
        mu_key = head_spec("nig_regression", t).output_keys["mu"]
        frames = {p: _candle_frame(dataset, np.asarray(idx[p]), t, block) for p in ("val", "test")}
        fitted = _fit_baselines(_candle_frame(dataset, train_rows, t, block), x_last["train"], seed)
        preds = {}
        for p in ("val", "test"):
            mu = tables[p][mu_key].to_numpy(dtype=np.float64)
            preds[p] = {"model_class": tables[p][cls_key].to_numpy(dtype=np.float64),
                        "model_reg_mu": 1.0 / (1.0 + np.exp(-mu / (np.std(mu) + 1e-12)))}   # sign(mu) at 0.5
        rows, verdicts[t] = _one_target(t, frames, x_last, preds, fitted, n_boot, seed, min_auc_gain, lang)
        table_rows += rows
    return {"table": pd.DataFrame(table_rows), "verdicts": verdicts}
