"""Integrated verification: data integrity, baselines on test, confidence and trading, candle shape, one table

PURPOSE: ``run_full_verification``: the old TF section 7-d in one call over a finished run's arrays: data integrity (a),
    naive baselines with a day-block bootstrap (b), confidence and R:R trading (c), candle shape (d), one verdict table.
TAGS: verification, verdict table, label checks, finite checks, split gap, day bootstrap, baselines, run verification
PITFALLS: pure arrays in, no Run object: ``idx`` is {"train" | "val" | "test": rows of ``dataset``} and ``tables`` the
    run's PredictionTables of val and test in row order; every threshold is chosen on val / train and judged on test;
    the label check runs only for target modes "return" / None (any other skips with an info row); the table's
    labels are English (the old TF ones were Arabic) and its verdict symbols are the old ones; the independent unit of
    every bootstrap is the UTC calendar day; block d needs scikit-learn; blocks c and d need the close head (c also the
    high / low heads for trades); the cost is the battery's per side.
SOURCE: ported from crypto-signal-prediction workflow/verification.py (run_full_verification; thresholds and verdict
    logic unchanged: label agreement >= 0.99, gap >= window + horizon, day-bootstrap interval for blocks b).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import FUTURE_OF, LAST_COLUMN_INDEX, LAST_OF, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, anchor_column, codec_name
from nig_timenet_torch.eval.battery_adapter import to_battery_table
from nig_timenet_torch.eval.day_blocks import calendar_day, day_blocks
from nig_timenet_torch.eval.verification_candle import candle_baseline_report
from nig_timenet_torch.eval.verification_latest import latest_by_target, print_latest_by_target
from nig_timenet_torch.eval.verification_trading import FAIL, PASS, rr_trading_report, selective_direction_report
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics

__all__ = ["run_full_verification", "signals_frame", "label_agreement", "summary_text", "SUMMARY_CSV", "SUMMARY_JSON",
           "LATEST_CSV", "DEFAULT_TARGET_MODES", "MIN_LABEL_AGREEMENT"]

SUMMARY_CSV, SUMMARY_JSON = "verification_summary.csv", "verification_summary.json"
LATEST_CSV = "verification_latest.csv"
FLAT, WARN, INFO = "➖", "⚠️", "ℹ️"        # ➖ ⚠️ ℹ️
#: the target modes whose labels ARE returns of the raw prices (the label check compares them with last_candles)
DEFAULT_TARGET_MODES = (None, "return")
MIN_LABEL_AGREEMENT = 0.99
CHUNK = 20_000
A, B, C_CONF, C_TRADE, C, D = ("a) data", "b) baselines", "c) confidence", "c) trading", "c) confidence and trading",
                               "d) candle shape")


def _row(section: str, check: str, value, verdict: str, note: str = "") -> dict:
    return {"section": section, "check": check, "value": value, "verdict": verdict, "note": note}


def _all_finite(array, rows: np.ndarray) -> bool:
    return all(bool(np.isfinite(np.asarray(array[rows[i:i + CHUNK]])).all()) for i in range(0, len(rows), CHUNK))


def label_agreement(dataset: dict, rows, targets) -> dict:
    """{"<t>_class": share, "<t>_reg": share} of ``rows`` whose stored label equals the label rebuilt from the raw
    prices of ``last_candles``: the class by the head's own last price, the regression by the registered codec (rows
    whose label sits at the clip are left out of the regression share: the codec state may not know the clip)."""
    rows = np.asarray(rows)
    lc = np.asarray(dataset["last_candles"], dtype=np.float64)[rows]
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    codec, out = TARGET_CODECS[state["mode"]](), {}
    for t in targets:
        own, future = lc[:, LAST_COLUMN_INDEX[LAST_OF[t]]], lc[:, LAST_COLUMN_INDEX[FUTURE_OF[t]]]
        y_cls = np.asarray(dataset[head_spec("binary_classification", t).label_key])[rows] > 0
        out[f"{t}_class"] = float(np.mean(y_cls == (future > own)))
        y = np.asarray(dataset[head_spec("nig_regression", t).label_key], dtype=np.float64)[rows]
        expected = codec.encode(anchor_and_future(lc, t, state), state)
        ok = np.abs(y / state["scale"]) < 0.999
        close = np.abs(y - expected)[ok] <= 1e-4 + 1e-3 * np.abs(expected)[ok]
        out[f"{t}_reg"] = float(np.mean(close)) if ok.any() else 1.0
    return out


def _gap_rows(dataset: dict, idx: dict) -> list:
    """train -> val and val -> test: the time gap in base bars must be at least window + horizon."""
    lc = np.asarray(dataset["last_candles"])
    ts = lc[:, TS_COL].astype(np.int64)
    bar = int(pd.Timedelta(dataset["base_tf"]).value)
    window = int(round((ts[0] - sample_times(dataset)[0][0]) / bar)) + 1        # widest window in base bars
    horizon = int(dataset["forecast_horizon"])
    rows = []
    for a, b in (("train", "val"), ("val", "test")):
        if not len(idx[a]) or not len(idx[b]):
            continue
        gap = (ts[np.asarray(idx[b])].min() - ts[np.asarray(idx[a])].max()) / bar
        verdict = PASS if gap >= window + horizon else (WARN if gap > 0 else FAIL)
        rows.append(_row(A, f"{a} before {b} in time (gap in bars)", round(float(gap), 1), verdict,
                         f"needs >= window + horizon = {window + horizon}"))
    return rows


def data_rows(dataset: dict, idx: dict, targets, x_keys, target_mode) -> list:
    """Block a: finite inputs, labels equal to the raw prices, split order, up share of train."""
    rows = []
    for stage in ("train", "val", "test"):
        ok = all(_all_finite(dataset[k], np.asarray(idx[stage])) for k in x_keys)
        rows.append(_row(A, f"X_{stage} has no NaN / Inf", ok, PASS if ok else FAIL))
    for stage in ("val", "test"):
        if target_mode not in DEFAULT_TARGET_MODES:       # their own self-check is in the target-mode switch
            rows.append(_row(A, f"labels match the raw prices ({stage})", "-", INFO,
                             f"skipped: target mode {target_mode!r} labels are not plain returns"))
            continue
        agree = label_agreement(dataset, idx[stage], targets)
        worst = min(agree.values())
        rows.append(_row(A, f"labels match the raw prices ({stage})", round(worst, 4),
                         PASS if worst >= MIN_LABEL_AGREEMENT else FAIL,
                         json.dumps({k: round(v, 4) for k, v in agree.items()})))
    rows += _gap_rows(dataset, idx)
    for t in targets:
        key = head_spec("binary_classification", t).label_key
        p = float(np.mean(np.asarray(dataset[key])[np.asarray(idx["train"])] > 0))
        rows.append(_row(A, f"up share in train ({t})", round(p, 4), INFO, "majority baseline = max(p, 1 - p)"))
    return rows


def _day_of(dataset: dict, rows) -> np.ndarray:
    return calendar_day(np.asarray(dataset["last_candles"])[np.asarray(rows), TS_COL].astype(np.int64))


def _verdict_of(lo: float, hi: float) -> str:
    return PASS if lo > 0 else (FAIL if hi < 0 else FLAT)


def baseline_rows(dataset: dict, idx: dict, tables: dict, targets, n_boot: int) -> list:
    """Block b on TEST: direction accuracy minus the majority class, and MAE(constant = train median) minus MAE(model),
    each per row with a day-block bootstrap interval (a verdict of PASS needs the whole interval above 0)."""
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    blocks = day_blocks(_day_of(dataset, idx["test"]), 1)
    rows = [_row(B, "independent test days", int(len(np.unique(blocks))), INFO,
                 "every verdict below is on these days: fewer than ~250 means statistically weak verdicts")]
    train, test = np.asarray(idx["train"]), np.asarray(idx["test"])
    for t in targets:
        cls, reg = head_spec("binary_classification", t), head_spec("nig_regression", t)
        if cls.output_keys["logits"] in tables["test"] and cls.label_key in dataset:
            y = (np.asarray(dataset[cls.label_key])[test] > 0).astype(int)
            maj = int(np.mean(np.asarray(dataset[cls.label_key])[train] > 0) >= 0.5)       # the larger train class
            pred = (tables["test"][cls.output_keys["logits"]].to_numpy() >= 0.5).astype(int)
            m, lo, hi = metrics.block_bootstrap_ci((pred == y).astype(float) - (maj == y).astype(float), blocks,
                                                   n_boot=n_boot)
            rows.append(_row(B, f"direction accuracy {t} - majority class (test)", round(m, 4), _verdict_of(lo, hi),
                             f"model {np.mean(pred == y):.4f} | majority {np.mean(maj == y):.4f} | "
                             f"interval [{lo:+.4f}, {hi:+.4f}]"))
        if reg.output_keys["mu"] in tables["test"] and reg.label_key in dataset:
            y = np.asarray(dataset[reg.label_key], dtype=np.float64)[test] / state["scale"]       # the unit of mu
            mu = tables["test"][reg.output_keys["mu"]].to_numpy(dtype=np.float64) / state["scale"]
            const = float(np.median(np.asarray(dataset[reg.label_key], dtype=np.float64)[train])) / state["scale"]
            m, lo, hi = metrics.block_bootstrap_ci(np.abs(y - const) - np.abs(y - mu), blocks, n_boot=n_boot)
            rows.append(_row(B, f"MAE constant - MAE model ({t}_reg, test)", round(m, 5), _verdict_of(lo, hi),
                             f"model MAE {metrics.mae(y, mu):.5f} | constant ({const:+.5f}) "
                             f"{metrics.mae(y, np.full_like(y, const)):.5f} | interval [{lo:+.5f}, {hi:+.5f}]"))
    return rows


def signals_frame(dataset: dict, rows, table: pd.DataFrame, state: dict, targets) -> pd.DataFrame:
    """The battery table of ``rows`` (``eval.battery_adapter.to_battery_table``) + ``up`` (the close went up), ``day``
    (UTC calendar day of the row) and ``conf`` (the close confidence head, when the table has it)."""
    rows = np.asarray(rows)
    lc = np.asarray(dataset["last_candles"])
    keys = {t: head_spec("nig_regression", t).label_key for t in targets}
    labels = {key: np.asarray(dataset[key])[rows] for key in keys.values()}
    anchors = {t: lc[rows, LAST_COLUMN_INDEX[anchor_column(state, t)]] for t in targets}
    df = to_battery_table(table, labels, anchors=anchors, last_close=lc[rows, LAST_COLUMN_INDEX["last_close"]],
                          codec_state=state)
    df["up"] = (df["y_ret"] > 0).astype(int)
    df["day"] = _day_of(dataset, rows)
    conf = head_spec("nig_regression", "close").output_keys["confidence"]
    if conf in table.columns:
        df["conf"] = table[conf].to_numpy(dtype=np.float64)
    return df


def confidence_rows(dataset: dict, idx: dict, tables: dict, targets, target_mode, *, target_acc: float, rr: float,
                    stop_mode: str, cost_per_side: float, min_n: int) -> tuple[list, dict]:
    """Block c: (rows, details). Skipped with an info row when the target mode is not plain returns or the close head
    is missing; the trades also need the high and low heads (their realised extremes are the exits)."""
    empty = {"selective": pd.DataFrame(), "trades": pd.DataFrame(), "val_df": pd.DataFrame(), "test_df": pd.DataFrame()}
    if target_mode not in DEFAULT_TARGET_MODES:
        return [_row(C, "skipped", "-", INFO, f"assumes 'return' targets: target mode {target_mode!r}")], empty
    mu_key = head_spec("nig_regression", "close").output_keys["mu"]
    if "close" not in targets or any(mu_key not in tables[s] for s in ("val", "test")):
        return [_row(C, "skipped", "-", INFO, "no close head in the run's tables")], empty
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    val, test = (signals_frame(dataset, idx[s], tables[s], state, targets) for s in ("val", "test"))
    sel = selective_direction_report(val, test, target_acc=target_acc, min_n=min_n)
    can_trade = all(c in test for c in ("fut_high", "fut_low")) and (stop_mode != "model_levels" or "pred_low" in test)
    trd = (rr_trading_report(val, test, rr=rr, stop_mode=stop_mode, cost_per_side=cost_per_side, min_trades=min_n)
           if can_trade else pd.DataFrame())
    rows = []
    for _, r in sel.iterrows():
        cons = (r["monotonic_val"] > 0.5) and (r["monotonic_test"] > 0.5)
        rows.append(_row(C_CONF, f"confidence consistency ({r['score']})",
                         f"val {r['monotonic_val']:+.2f} / test {r['monotonic_test']:+.2f}", PASS if cons else FAIL,
                         "needs > +0.5 on both periods"))
        rows.append(_row(C_CONF, f"slice >= {target_acc:.0%} ({r['score']})", r.get("test_acc", np.nan),
                         PASS if str(r["verdict"]).startswith(PASS) else FAIL, str(r["verdict"])))
    if not can_trade:
        rows.append(_row(C_TRADE, "skipped", "-", INFO, f"{stop_mode!r} trades need the high and low heads in the run"))
    for _, r in trd.iterrows():
        rows.append(_row(C_TRADE, f"trades {rr:g}:1 ({r['score']})", r.get("test_exp_R", np.nan),
                         PASS if str(r["verdict"]).startswith(PASS) else FAIL,
                         f"val_exp_R={r.get('val_exp_R', np.nan):.3f}, t_days={r.get('test_t_days', np.nan):.2f}"))
    return rows, {"selective": sel, "trades": trd, "val_df": val, "test_df": test}


def candle_rows(dataset: dict, idx: dict, tables: dict, targets, x_key: str, n_boot: int) -> tuple[list, dict]:
    """Block d: (rows, details) for the high / low class heads that the run has."""
    wanted = tuple(t for t in TARGET_COLUMNS[:2] if t in targets and all(
        head_spec(k, t).output_keys[o] in tables[s] for s in ("val", "test")
        for k, o in (("binary_classification", "logits"), ("nig_regression", "mu"))))
    if not wanted:
        return [_row(D, "skipped", "-", INFO, "no high / low class head in the run's tables")], {}
    cb = candle_baseline_report(dataset, idx, tables, targets=wanted, x_key=x_key, n_boot=n_boot)
    rows = [_row(D, f"{t}_class above {v['best_baseline']}", round(v["auc_gain_when_combined"][0], 4),
                 v["verdict"][:1], v["verdict"]) for t, v in cb["verdicts"].items()]
    return rows, {"candle": cb}


def summary_text(summary: pd.DataFrame) -> str:
    """The verdict table as text, its counts and the one-line conclusion about confidence / trades."""
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        body = summary.to_string(index=False)
    n_ok, n_bad = int((summary["verdict"] == PASS).sum()), int((summary["verdict"] == FAIL).sum())
    trade_ok = summary[summary["section"].isin([C_CONF, C_TRADE])]["verdict"].eq(PASS).any()
    end = ("at least one confidence / trade source passed test: check it on other time windows before any use"
           if trade_ok else "nothing tradeable passed test with this model and this data")
    return (f"{'=' * 100}\nINTEGRATED VERIFICATION\n{'=' * 100}\n{body}\n\n   pass {n_ok} | fail {n_bad} | "
            f"other {len(summary) - n_ok - n_bad}\n   conclusion: {end}")


def run_full_verification(dataset: dict, idx: dict, tables: dict, *, targets, x_keys, cost_per_side: float,
                          target_mode=None, target_acc: float = 0.65, rr: float = 2.0, stop_mode: str = "nig",
                          n_boot: int = 500, run_candle: bool = True, min_n: int = 200, latest_n: int = 1,
                          latest_coins: int = 3, out_dir=None, verbose: bool = False) -> dict:
    """Every check of the integrated verification -> ``{"summary": verdict table (section, check, value, verdict,
    note), "details": {selective, trades, val_df, test_df, candle}}``; with ``out_dir`` also
    ``verification_summary.csv`` / ``.json``. ``dataset``: the artifact dict; ``idx``: {"train" | "val" | "test": rows};
    ``tables``: {"val" | "test": PredictionTable of those rows in order}; ``targets``: the run's price targets;
    ``x_keys``: the ``X_<tf>`` keys the model reads (the first is the base frame of the last-step baseline);
    ``target_mode``: the stamped mode of the labels (None / "return" = plain returns); ``cost_per_side``: the battery's;
    ``min_n``: the smallest slice / trade count a verdict accepts (200 in the old project); ``latest_n`` > 0 adds the
    latest-samples table of each price target (``details["latest"]``, ``verification_latest.csv``): the last
    ``latest_n`` test samples of ``latest_coins`` coins."""
    rows = data_rows(dataset, idx, targets, x_keys, target_mode)
    rows += baseline_rows(dataset, idx, tables, targets, n_boot)
    conf_rows, details = confidence_rows(dataset, idx, tables, targets, target_mode, target_acc=target_acc, rr=rr,
                                         stop_mode=stop_mode, cost_per_side=cost_per_side, min_n=min_n)
    rows += conf_rows
    if run_candle:
        candle, more = candle_rows(dataset, idx, tables, targets, x_keys[0], n_boot)
        rows, details = rows + candle, {**details, **more}
    summary = pd.DataFrame(rows)
    if latest_n > 0:
        state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
        details["latest"] = latest_by_target(dataset, idx["test"], tables["test"], state, targets, latest_n,
                                             latest_coins)
    if verbose:
        print(summary_text(summary))
        if "latest" in details:
            print_latest_by_target(details["latest"], latest_n)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if "latest" in details:
            details["latest"].to_csv(out / LATEST_CSV, index=False)
        summary.to_csv(out / SUMMARY_CSV, index=False, encoding="utf-8-sig")
        records = summary.astype(str).to_dict("records")
        (out / SUMMARY_JSON).write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"summary": summary, "details": details}
