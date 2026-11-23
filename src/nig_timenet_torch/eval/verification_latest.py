"""Verification: the latest samples of each price target (signal, confidence, move, true move, errors) in prices

PURPOSE: The TF "latest N samples per target" table for the verification output: per coin and per price target the
    signal, confidence, uncertainty, expected / true move, move error, entry, predicted / true price and price error.
TAGS: verification, latest samples, per target table, true move, move error, price error, tf parity, decode
PITFALLS: prices come from the registered target codec (``decode``) and the raw ``last_candles`` (the truth is the
    real future price, never a decoded label); the entry and the move of a head are relative to ITS OWN last price
    (last_high / last_low / last_close), as the TF table did; the signal is the class head (p_up >= 0.5), ``conf`` the
    NIG confidence head (0-100, blank when the head is missing), ``unc`` the NIG predictive sd as a percent of the
    entry; ``ok`` = the signalled side matches the true move; err_move = move - true_move in percentage points,
    err_px = pred - true (price units); a target mode the codec registry cannot decode raises in ``codec_name``.
SOURCE: format of crypto-signal-prediction evaluation/unified.py (_print_v4_report, build_latest_table); columns
    true_move, err_move and err_px added at the owner's request (2026-11-23).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import FUTURE_OF, LAST_COLUMN_INDEX, LAST_OF, TS_COL
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_column
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import nig_predictive_std

__all__ = ["latest_by_target", "format_latest_by_target", "print_latest_by_target", "LATEST_TARGET_COLUMNS"]

LATEST_TARGET_COLUMNS = ("row", "coin", "target", "signal", "conf_pct", "unc_pct", "move_pct", "true_move_pct",
                         "err_move_pct", "entry", "pred", "true", "err_px", "ok")


def _positions(coins: np.ndarray, times: np.ndarray, n: int, n_coins: int) -> np.ndarray:
    """Positions (into the stage rows) of the last ``n`` rows of each of the ``n_coins`` coins with most rows,
    grouped by coin (most rows first) and oldest first inside a coin."""
    names, counts = np.unique(coins, return_counts=True)
    keep = names[np.argsort(-counts, kind="stable")[:n_coins]]
    out = []
    for c in keep:
        pos = np.flatnonzero(coins == c)
        out.append(pos[np.argsort(times[pos], kind="stable")[-n:]])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def col(table: pd.DataFrame, pos: np.ndarray, key: str) -> np.ndarray:
    return table[key].to_numpy(dtype=np.float64)[pos]


def latest_by_target(dataset: dict, rows, table: pd.DataFrame, state: dict, targets, n: int = 1,
                     n_coins: int = 3) -> pd.DataFrame:
    """One row per (sample, target) for the last ``n`` samples of ``n_coins`` coins of ``rows`` (the stage's dataset
    rows, ``table`` = its PredictionTable in the same order), targets that have a regression head in ``table`` only."""
    rows = np.asarray(rows)
    if n < 1 or n_coins < 1:
        raise ValueError("latest_by_target: n and n_coins must be >= 1")
    lc = np.asarray(dataset["last_candles"], dtype=np.float64)[rows]
    pos = _positions(table["asset"].astype(str).to_numpy(), lc[:, TS_COL], n, n_coins)
    codec, frames = TARGET_CODECS[state["mode"]](), []
    for t in targets:
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        if reg.output_keys["mu"] not in table.columns:
            continue
        mu = col(table, pos, reg.output_keys["mu"])
        std = nig_predictive_std(*(col(table, pos, reg.output_keys[k]) for k in ("nu", "alpha", "beta")))
        entry = lc[pos, LAST_COLUMN_INDEX[LAST_OF[t]]]
        anchor = lc[pos, LAST_COLUMN_INDEX[anchor_column(state, t)]]
        pred = codec.decode(mu, state, anchor)
        width = codec.decode(mu + std, state, anchor) - pred
        true = lc[pos, LAST_COLUMN_INDEX[FUTURE_OF[t]]]
        p_up = col(table, pos, cls.output_keys["logits"]) if cls.output_keys["logits"] in table.columns else None
        up = (p_up >= 0.5) if p_up is not None else pred >= entry
        has_conf = reg.output_keys["confidence"] in table.columns
        conf = col(table, pos, reg.output_keys["confidence"]) if has_conf else np.nan
        true_move = true / entry - 1.0
        frames.append(pd.DataFrame({
            "row": rows[pos], "coin": table["asset"].astype(str).to_numpy()[pos], "target": t,
            "signal": np.where(up, "up", "down"), "conf_pct": 100.0 * conf, "unc_pct": 100.0 * width / entry,
            "move_pct": 100.0 * (pred / entry - 1.0), "true_move_pct": 100.0 * true_move,
            "err_move_pct": 100.0 * ((pred - true) / entry), "entry": entry, "pred": pred, "true": true,
            "err_px": pred - true, "ok": np.where(up, true_move > 0, true_move < 0)}))
    if not frames:
        return pd.DataFrame(columns=list(LATEST_TARGET_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    order = {c: i for i, c in enumerate(dict.fromkeys(out["coin"]))}
    tgt = {t: i for i, t in enumerate(targets)}
    out = out.assign(_c=out["coin"].map(order), _t=out["target"].map(tgt)).sort_values(
        ["_c", "row", "_t"], kind="stable").drop(columns=["_c", "_t"])
    return out.reset_index(drop=True)[list(LATEST_TARGET_COLUMNS)]


def _num(v, spec: str) -> str:
    return "-" if pd.isna(v) else format(v, spec)


def format_latest_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """A text copy for printing, in the TF table's style: arrows for the signal, percentages with a sign for the
    moves, prices in 6 significant digits, check marks for ``ok``; the coin column only when more than one coin."""
    out = pd.DataFrame({
        "#": df["row"], "coin": df["coin"], "target": df["target"],
        "signal": df["signal"].map({"up": "▲", "down": "▼"}),
        "conf": df["conf_pct"].map(lambda v: _num(v, ".1f") + ("" if pd.isna(v) else "%")),
        "unc": df["unc_pct"].map(lambda v: _num(v, ".2f") + "%"),
        "move": df["move_pct"].map(lambda v: _num(v, "+.3f") + "%"),
        "true_move": df["true_move_pct"].map(lambda v: _num(v, "+.3f") + "%"),
        "err_move": df["err_move_pct"].map(lambda v: _num(v, "+.3f") + "%"),
        "entry": df["entry"].map(lambda v: _num(v, ".6g")), "pred": df["pred"].map(lambda v: _num(v, ".6g")),
        "true": df["true"].map(lambda v: _num(v, ".6g")), "err_px": df["err_px"].map(lambda v: _num(v, "+.6g")),
        "ok": df["ok"].map(lambda v: "-" if pd.isna(v) else ("✅" if v else "❌"))})
    return out if out["coin"].nunique() > 1 else out.drop(columns=["coin"])


def print_latest_by_target(df: pd.DataFrame, n: int, echo=print) -> str:
    """Print (through ``echo``) and return the table text under the TF heading (newest at the bottom)."""
    head = f"\U0001f4ca latest {n} sample(s) per target (newest at the bottom):"
    text = format_latest_by_target(df).to_string(index=False) if len(df) else "(no samples to show)"
    out = f"{head}\n{text}"
    echo(out)
    return out
