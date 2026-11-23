"""Integrated verification: data integrity, baselines on test, confidence and trading, candle shape, one table

PURPOSE: ``run_full_verification``: the old TF section 7-d in one call over a finished run's arrays: data integrity (a),
    naive baselines with a day-block bootstrap (b), confidence and R:R trading (c), candle shape (d), one verdict table.
TAGS: verification, verdict table, label checks, finite checks, split gap, day bootstrap, baselines, run verification
PITFALLS: pure arrays in, no Run object: ``idx`` is {"train" | "val" | "test": rows of ``dataset``} and ``tables`` the
    run's PredictionTables of val and test in row order; every threshold is chosen on val / train and judged on test;
    the label check runs only for target modes "return" / None (any other skips with an info row); the defaults are
    this project's (English text, the battery's cost per side, the NIG predictive sd as the stop width, UTC-day
    bootstrap blocks); ``match_old=True`` switches every one of them to the old TF project's (Arabic text, 0.08 %
    round-trip cost, Student-t stop width, raw-timestamp blocks, 300 candle draws, label clip 1.0, no latest table) and
    its numbers then equal the old project's (tests/parity); blocks c and d need the close head (c also high / low for
    trades) and d needs scikit-learn.
SOURCE: ported from crypto-signal-prediction workflow/verification.py (run_full_verification; thresholds and verdict
    logic unchanged: label agreement >= 0.99, gap >= window + horizon, bootstrap interval for blocks b).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import FUTURE_OF, LAST_COLUMN_INDEX, LAST_OF, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, codec_name
from nig_timenet_torch.eval.battery_adapter import to_battery_table
from nig_timenet_torch.eval.battery_lib import round_trip
from nig_timenet_torch.eval.day_blocks import calendar_day, day_blocks
from nig_timenet_torch.eval.verification_candle import candle_baseline_report
from nig_timenet_torch.eval.verification_latest import latest_by_target, print_latest_by_target
from nig_timenet_torch.eval.verification_print import (
    label_check,
    print_candle,
    print_selective,
    print_trades,
)
from nig_timenet_torch.eval.verification_print import summary_text as _summary_text
from nig_timenet_torch.eval.verification_text import DEFAULT_VARIANT, OLD_COST_PER_SIDE, OLD_VARIANT, Variant, t
from nig_timenet_torch.eval.verification_trading import FAIL, PASS, rr_trading_report, selective_direction_report
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.workflow.run_common import decode_anchors

__all__ = ["run_full_verification", "signals_frame", "label_agreement", "summary_text", "SUMMARY_CSV", "SUMMARY_JSON",
           "LATEST_CSV", "DEFAULT_TARGET_MODES", "MIN_LABEL_AGREEMENT", "Variant", "OLD_VARIANT", "OLD_COST_PER_SIDE"]

SUMMARY_CSV, SUMMARY_JSON = "verification_summary.csv", "verification_summary.json"
LATEST_CSV = "verification_latest.csv"
FLAT, WARN, INFO = "➖", "⚠️", "ℹ️"        # ➖ ⚠️ ℹ️
#: the target modes whose labels ARE returns of the raw prices (the label check compares them with last_candles)
DEFAULT_TARGET_MODES = (None, "return")
MIN_LABEL_AGREEMENT = 0.99
CHUNK = 20_000
#: the section names of the summary table (``verification_text``: "sec_<key>"), in the old table's order
A, B, C_CONF, C_TRADE, C, D = "a", "b", "conf", "trade", "c", "d"
SECTIONS = (A, B, C, C_CONF, C_TRADE, D)


def _sec(lang: str, key: str) -> str:
    return t(lang, f"sec_{key}")


def _row(lang: str, section: str, check: str, value, verdict: str, note: str = "") -> dict:
    return {"section": _sec(lang, section), "check": check, "value": value, "verdict": verdict, "note": note}


def _all_finite(array, rows: np.ndarray) -> bool:
    return all(bool(np.isfinite(np.asarray(array[rows[i:i + CHUNK]])).all()) for i in range(0, len(rows), CHUNK))


def label_agreement(dataset: dict, rows, targets, clip: float | None = None) -> dict:
    """{"<t>_class": share, "<t>_reg": share} of ``rows`` whose stored label equals the label rebuilt from the raw
    prices of ``last_candles``: the class by the head's own last price, the regression by the registered codec. With
    ``clip`` None the rows whose label sits at the clip are left out of the regression share (the codec state may not
    know the clip); with a number every row is compared with the return clipped to +-``clip`` (the old project)."""
    rows = np.asarray(rows)
    lc = np.asarray(dataset["last_candles"], dtype=np.float64)[rows]
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    codec, out = TARGET_CODECS[state["mode"]](), {}
    for tg in targets:
        own, future = lc[:, LAST_COLUMN_INDEX[LAST_OF[tg]]], lc[:, LAST_COLUMN_INDEX[FUTURE_OF[tg]]]
        y_cls = np.asarray(dataset[head_spec("binary_classification", tg).label_key])[rows] > 0
        out[f"{tg}_class"] = float(np.mean(y_cls == (future > own)))
        y = np.asarray(dataset[head_spec("nig_regression", tg).label_key], dtype=np.float64)[rows]
        expected = codec.encode(anchor_and_future(lc, tg, state), state)
        if clip is None:
            ok = np.abs(y / state["scale"]) < 0.999
        else:
            expected, ok = np.clip(expected / state["scale"], -clip, clip) * state["scale"], np.ones(len(y), bool)
        close = np.abs(y - expected)[ok] <= 1e-4 + 1e-3 * np.abs(expected)[ok]
        out[f"{tg}_reg"] = float(np.mean(close)) if ok.any() else 1.0
    return out


def _gap_rows(dataset: dict, idx: dict, lang: str) -> list:
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
        rows.append(_row(lang, A, t(lang, "gap", a=a, b=b), round(float(gap), 1), verdict,
                         t(lang, "gap_note", n=window + horizon)))
    return rows


def data_rows(dataset: dict, idx: dict, targets, x_keys, target_mode, variant: Variant = DEFAULT_VARIANT) -> list:
    """Block a: finite inputs, labels equal to the raw prices, split order, up share of train."""
    lang, rows = variant.lang, []
    for stage in ("train", "val", "test"):
        ok = all(_all_finite(dataset[k], np.asarray(idx[stage])) for k in x_keys)
        rows.append(_row(lang, A, t(lang, "finite", stage=stage), ok, PASS if ok else FAIL))
    for stage in ("val", "test"):
        check = t(lang, "labels", stage=stage)
        if target_mode not in DEFAULT_TARGET_MODES:       # their own self-check is in the target-mode switch
            rows.append(_row(lang, A, check, "-", INFO, t(lang, "labels_skip", mode=target_mode)))
            continue
        agree = label_agreement(dataset, idx[stage], targets, variant.label_clip)
        worst = min(agree.values())
        rows.append(_row(lang, A, check, round(worst, 4), PASS if worst >= MIN_LABEL_AGREEMENT else FAIL,
                         json.dumps({k: round(v, 4) for k, v in agree.items()})))
    rows += _gap_rows(dataset, idx, lang)
    for tg in targets:
        key = head_spec("binary_classification", tg).label_key
        p = float(np.mean(np.asarray(dataset[key])[np.asarray(idx["train"])] > 0))
        rows.append(_row(lang, A, t(lang, "up_share", t=tg), round(p, 4), INFO, t(lang, "up_note")))
    return rows


def _unit(dataset: dict, rows, block: str) -> np.ndarray:
    """What a bootstrap resamples, per row: its UTC calendar day, or its raw bar timestamp (``block="timestamp"``)."""
    ts = np.asarray(dataset["last_candles"])[np.asarray(rows), TS_COL].astype(np.int64)
    return calendar_day(ts) if block == "day" else pd.to_datetime(ts).values


def _verdict_of(lo: float, hi: float) -> str:
    return PASS if lo > 0 else (FAIL if hi < 0 else FLAT)


def baseline_rows(dataset: dict, idx: dict, tables: dict, targets, n_boot: int,
                  variant: Variant = DEFAULT_VARIANT) -> list:
    """Block b on TEST: direction accuracy minus the majority class, and MAE(constant = train median) minus MAE(model),
    each per row with a block-bootstrap interval (a verdict of PASS needs the whole interval above 0). The old table
    lists the direction rows of every target first, then the MAE rows."""
    lang, state = variant.lang, TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    units = _unit(dataset, idx["test"], variant.block)
    blocks = day_blocks(units, 1) if variant.block == "day" else units
    rows = [_row(lang, B, t(lang, "days"), int(len(np.unique(blocks))), INFO, t(lang, "days_note"))]
    train, test = np.asarray(idx["train"]), np.asarray(idx["test"])
    for tg in targets:
        cls = head_spec("binary_classification", tg)
        if cls.output_keys["logits"] in tables["test"] and cls.label_key in dataset:
            y = (np.asarray(dataset[cls.label_key])[test] > 0).astype(int)
            maj = int(np.mean(np.asarray(dataset[cls.label_key])[train] > 0) >= 0.5)       # the larger train class
            pred = (tables["test"][cls.output_keys["logits"]].to_numpy() >= 0.5).astype(int)
            m, lo, hi = metrics.block_bootstrap_ci((pred == y).astype(float) - (maj == y).astype(float), blocks,
                                                   n_boot=n_boot)
            rows.append(_row(lang, B, t(lang, "dir", t=tg), round(m, 4), _verdict_of(lo, hi),
                             t(lang, "dir_note", acc=np.mean(pred == y), maj=np.mean(maj == y), lo=lo, hi=hi)))
    for tg in targets:
        reg = head_spec("nig_regression", tg)
        if reg.output_keys["mu"] in tables["test"] and reg.label_key in dataset:
            y = np.asarray(dataset[reg.label_key], dtype=np.float64)[test] / state["scale"]       # the unit of mu
            mu = tables["test"][reg.output_keys["mu"]].to_numpy(dtype=np.float64) / state["scale"]
            const = float(np.median(np.asarray(dataset[reg.label_key], dtype=np.float64)[train])) / state["scale"]
            m, lo, hi = metrics.block_bootstrap_ci(np.abs(y - const) - np.abs(y - mu), blocks, n_boot=n_boot)
            rows.append(_row(lang, B, t(lang, "mae", t=tg), round(m, 5), _verdict_of(lo, hi),
                             t(lang, "mae_note", m=metrics.mae(y, mu), c=const,
                               mc=metrics.mae(y, np.full_like(y, const)), lo=lo, hi=hi)))
    return rows


def signals_frame(dataset: dict, rows, table: pd.DataFrame, state: dict, targets,
                  variant: Variant = DEFAULT_VARIANT) -> pd.DataFrame:
    """The battery table of ``rows`` (``eval.battery_adapter.to_battery_table``) + ``up`` (the close went up), ``day``
    (the bootstrap unit of the row, ``variant.block``) and ``conf`` (the close confidence head, when the table has it).
    ``variant.stop_width`` "student_t" replaces ``sigma`` (the NIG predictive sd) by the Student-t scale
    sqrt(beta (1 + nu) / (nu alpha)) decoded to a return (the old project's width)."""
    rows = np.asarray(rows)
    lc = np.asarray(dataset["last_candles"])
    keys = {tg: head_spec("nig_regression", tg).label_key for tg in targets}
    labels = {key: np.asarray(dataset[key])[rows] for key in keys.values()}
    anchors = decode_anchors(dataset, rows, targets, state)    # a price column, or the (n, 2) rows of a target mode
    last_close = lc[rows, LAST_COLUMN_INDEX["last_close"]]
    df = to_battery_table(table, labels, anchors=anchors, last_close=last_close, codec_state=state)
    df["up"] = (df["y_ret"] > 0).astype(int)
    df["day"] = _unit(dataset, rows, variant.block)
    out = head_spec("nig_regression", "close").output_keys
    if out["confidence"] in table.columns:
        df["conf"] = table[out["confidence"]].to_numpy(dtype=np.float64)
    if variant.stop_width == "student_t":
        nu, alpha, beta, mu = (table[out[k]].to_numpy(dtype=np.float64) for k in ("nu", "alpha", "beta", "mu"))
        width = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
        codec = TARGET_CODECS[state["mode"]]()
        df["sigma"] = (codec.decode(mu + width, state, anchors["close"]) - codec.decode(mu, state, anchors["close"])) \
            / last_close
    return df


def confidence_rows(dataset: dict, idx: dict, tables: dict, targets, target_mode, *, target_acc: float, rr: float,
                    stop_mode: str, cost_per_side: float, min_n: int, variant: Variant = DEFAULT_VARIANT,
                    verbose: bool = False) -> tuple[list, dict]:
    """Block c: (rows, details). Skipped with an info row when the target mode is not plain returns or the close head
    is missing; the trades also need the high and low heads (their realised extremes are the exits)."""
    lang = variant.lang
    empty = {"selective": pd.DataFrame(), "trades": pd.DataFrame(), "val_df": pd.DataFrame(), "test_df": pd.DataFrame()}
    if target_mode not in DEFAULT_TARGET_MODES:
        return [_row(lang, C, t(lang, "skipped"), "-", INFO, t(lang, "c_skip_mode", mode=target_mode))], empty
    mu_key = head_spec("nig_regression", "close").output_keys["mu"]
    if "close" not in targets or any(mu_key not in tables[s] for s in ("val", "test")):
        return [_row(lang, C, t(lang, "skipped"), "-", INFO, t(lang, "c_skip_close"))], empty
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    val, test = (signals_frame(dataset, idx[s], tables[s], state, targets, variant) for s in ("val", "test"))
    sel = selective_direction_report(val, test, target_acc=target_acc, min_n=min_n, lang=lang)
    if verbose:
        print_selective(sel, target_acc, lang)
    can_trade = all(c in test for c in ("fut_high", "fut_low")) and (stop_mode != "model_levels" or "pred_low" in test)
    trd = (rr_trading_report(val, test, rr=rr, stop_mode=stop_mode, cost_per_side=cost_per_side, min_trades=min_n,
                             lang=lang) if can_trade else pd.DataFrame())
    if verbose and can_trade:
        print_trades(trd, rr, stop_mode, round(100.0 * round_trip(cost_per_side), 6), lang)
    rows = []
    for _, r in sel.iterrows():
        cons = (r["monotonic_val"] > 0.5) and (r["monotonic_test"] > 0.5)
        rows.append(_row(lang, C_CONF, t(lang, "consistency", score=r["score"]),
                         f"val {r['monotonic_val']:+.2f} / test {r['monotonic_test']:+.2f}", PASS if cons else FAIL,
                         t(lang, "consistency_note")))
        rows.append(_row(lang, C_CONF, t(lang, "slice", acc=target_acc, score=r["score"]), r.get("test_acc", np.nan),
                         PASS if str(r["verdict"]).startswith(PASS) else FAIL, str(r["verdict"])))
    if not can_trade:
        rows.append(_row(lang, C_TRADE, t(lang, "skipped"), "-", INFO, t(lang, "c_skip_trade", stop=stop_mode)))
    for _, r in trd.iterrows():
        rows.append(_row(lang, C_TRADE, t(lang, "trades", rr=rr, score=r["score"]), r.get("test_exp_R", np.nan),
                         PASS if str(r["verdict"]).startswith(PASS) else FAIL,
                         f"val_exp_R={r.get('val_exp_R', np.nan):.3f}, t_days={r.get('test_t_days', np.nan):.2f}"))
    return rows, {"selective": sel, "trades": trd, "val_df": val, "test_df": test}


def candle_rows(dataset: dict, idx: dict, tables: dict, targets, x_key: str, n_boot: int,
                variant: Variant = DEFAULT_VARIANT, verbose: bool = False, target_mode=None) -> tuple[list, dict]:
    """Block d: (rows, details) for the high / low class heads that the run has."""
    lang = variant.lang
    wanted = tuple(tg for tg in TARGET_COLUMNS[:2] if tg in targets and all(
        head_spec(k, tg).output_keys[o] in tables[s] for s in ("val", "test")
        for k, o in (("binary_classification", "logits"), ("nig_regression", "mu"))))
    if not wanted:
        return [_row(lang, D, t(lang, "skipped"), "-", INFO, t(lang, "d_skip"))], {}
    cb = candle_baseline_report(dataset, idx, tables, targets=wanted, x_key=x_key, lang=lang, block=variant.block,
                                n_boot=variant.candle_n_boot or n_boot)
    if verbose:
        checks = ({s: label_check(dataset, idx[s], wanted) for s in ("val", "test")}
                  if target_mode in DEFAULT_TARGET_MODES else None)
        print_candle(cb, checks, lang)
    rows = [_row(lang, D, t(lang, "candle", t=tg, best=v["best_baseline"]), round(v["auc_gain_when_combined"][0], 4),
                 v["verdict"][:1], v["verdict"]) for tg, v in cb["verdicts"].items()]
    return rows, {"candle": cb}


def summary_text(summary: pd.DataFrame, lang: str = "en", out_dir=None) -> str:
    """The verdict table as text with its counts and the conclusion about confidence / trades (the old layout)."""
    return _summary_text(summary, lang, out_dir, [_sec(lang, C_CONF), _sec(lang, C_TRADE)])


def run_full_verification(dataset: dict, idx: dict, tables: dict, *, targets, x_keys, cost_per_side: float,
                          target_mode=None, target_acc: float = 0.65, rr: float = 2.0, stop_mode: str = "nig",
                          n_boot: int = 500, run_candle: bool = True, min_n: int = 200, latest_n: int = 1,
                          latest_coins: int = 3, variant: Variant = DEFAULT_VARIANT, match_old: bool = False,
                          out_dir=None, verbose: bool = False) -> dict:
    """Every check of the integrated verification -> ``{"summary": verdict table (section, check, value, verdict,
    note), "details": {selective, trades, val_df, test_df, candle}}``; with ``out_dir`` also
    ``verification_summary.csv`` / ``.json``. ``dataset``: the artifact dict; ``idx``: {"train" | "val" | "test": rows};
    ``tables``: {"val" | "test": PredictionTable of those rows in order}; ``targets``: the run's price targets;
    ``x_keys``: the ``X_<tf>`` keys the model reads (the first is the base frame of the last-step baseline);
    ``target_mode``: the stamped mode of the labels (None / "return" = plain returns); ``cost_per_side``: the battery's;
    ``min_n``: the smallest slice / trade count a verdict accepts (200 in the old project); ``latest_n`` > 0 adds the
    latest-samples table of each price target (``details["latest"]``, ``verification_latest.csv``): the last
    ``latest_n`` test samples of ``latest_coins`` coins. ``variant``: the choices the old project made differently
    (``verification_text.Variant``); ``match_old`` sets ALL of them to the old project's, charges its 0.08 % round-trip
    cost (``cost_per_side`` is ignored) and leaves out the latest table, so the outputs equal the old ones."""
    if match_old:
        variant, cost_per_side, latest_n = OLD_VARIANT, OLD_COST_PER_SIDE, 0
    lang = variant.lang
    rows = data_rows(dataset, idx, targets, x_keys, target_mode, variant)
    rows += baseline_rows(dataset, idx, tables, targets, n_boot, variant)
    conf_rows, details = confidence_rows(dataset, idx, tables, targets, target_mode, target_acc=target_acc, rr=rr,
                                         stop_mode=stop_mode, cost_per_side=cost_per_side, min_n=min_n,
                                         variant=variant, verbose=verbose)
    rows += conf_rows
    if run_candle:
        candle, more = candle_rows(dataset, idx, tables, targets, x_keys[0], n_boot, variant, verbose, target_mode)
        rows, details = rows + candle, {**details, **more}
    summary = pd.DataFrame(rows)
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    if latest_n > 0 and TARGET_CODECS[state["mode"]]().forecasts_price(state):   # a magnitude has no price to show
        details["latest"] = latest_by_target(dataset, idx["test"], tables["test"], state, targets, latest_n,
                                             latest_coins)
    if verbose:
        print(summary_text(summary, lang, out_dir))
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
