"""PURPOSE: The latest samples per coin over a battery table, in original price units: signal, confidence,
    uncertainty, expected and true move, error, entry, expected and true target; a row without a label is live.
TAGS: latest table, live table, latest samples, entry price, expected target, true target, error, confidence,
    uncertainty, signal, battery table, tf parity
PITFALLS: pure function of a battery table (``eval/battery_adapter.BATTERY_COLUMNS``); the signal is ``trade_side``
    (p_dir > 0.5 = LONG, the battery's one rule); confidence is the class head's probability of the signalled side
    (50-100 %), uncertainty the NIG predictive sd of the close return (not a probability); a row whose ``y_ret`` is
    not finite is LIVE (its true columns and ``ok`` are NaN / None, never 0), a row with a label is CLOSED;
    prices are ``last_close * (1 + return)`` so the entry is the close of the bar the prediction was made at;
    the expected target is the close ``horizon`` bars later, not an executable limit price; the signal comes from the
    class head and the sign of ``move_pct`` from mu, ``agrees`` says whether the two point the same way.
    Exploratory (rule C5).

Why (owner, 2026-11-23): the TF project printed the last N samples of a coin (``evaluation/latest_table.py``,
``print_latest_table``) with the expected move, its uncertainty and, once known, the truth; this project had no
counterpart (``eval/legacy_reports.py`` is over a whole stage). The high / low columns appear when the table has the
high / low heads (``pred_high`` / ``fut_high``): their targets are price levels, never a direction (owner, rule D).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import trade_side

__all__ = ["latest_table", "format_latest_table", "print_latest_table", "LATEST_COLUMNS", "REQUIRED_COLUMNS"]

REQUIRED_COLUMNS = ("mu", "sigma", "p_dir", "y_ret", "coin", "end_time", "last_close")
#: the columns of ``latest_table``, in order (the high / low columns follow when the table has them)
LATEST_COLUMNS = ("coin", "time", "status", "signal", "confidence_pct", "uncertainty_pct", "move_pct",
                  "true_move_pct", "error_pct", "edge_ratio", "agrees", "entry", "target", "target_lo", "target_hi",
                  "true_target", "ok")
_LEVELS = ("high", "low")


def _check(table: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"latest_table: the table lacks {missing} (a battery table has {list(REQUIRED_COLUMNS)})")


def _level_columns(out: pd.DataFrame, d: pd.DataFrame, entry: np.ndarray) -> None:
    """``target_high`` / ``true_high`` (and low): the predicted extreme and the realised one, as prices."""
    for lv in _LEVELS:
        if f"pred_{lv}" in d.columns:
            out[f"target_{lv}"] = entry * (1.0 + d[f"pred_{lv}"].to_numpy(dtype=np.float64))
            out[f"true_{lv}"] = entry * (1.0 + d[f"fut_{lv}"].to_numpy(dtype=np.float64))


def latest_table(table: pd.DataFrame, n: int = 5, coins=None, *, now=None) -> pd.DataFrame:
    """The last ``n`` rows of every coin (or of ``coins``), coin by coin, oldest first. ``now`` (a timestamp) keeps only
    rows made at or before it. Columns: ``LATEST_COLUMNS`` (percent columns are percent, prices are prices)."""
    _check(table)
    if n < 1:
        raise ValueError("latest_table: n must be >= 1")
    d = table.assign(end_time=pd.to_datetime(table["end_time"]))
    if coins is not None:
        wanted = [coins] if isinstance(coins, str) else list(coins)
        unknown = sorted(set(wanted) - set(d["coin"]))
        if unknown:
            raise ValueError(f"latest_table: unknown coin(s) {unknown}; the table holds {sorted(set(d['coin']))}")
        d = d[d["coin"].isin(wanted)]
    if now is not None:
        cut = pd.Timestamp(now)
        cut = cut.tz_convert("UTC").tz_localize(None) if cut.tzinfo else cut
        d = d[d["end_time"] <= cut]
    d = d.sort_values(["coin", "end_time"], kind="stable").groupby("coin", sort=True).tail(n)
    if d.empty:
        return pd.DataFrame(columns=list(LATEST_COLUMNS))
    mu, sg, y = (d[c].to_numpy(dtype=np.float64) for c in ("mu", "sigma", "y_ret"))
    entry = d["last_close"].to_numpy(dtype=np.float64)
    side = trade_side(d)
    p_side = np.where(side > 0, d["p_dir"].to_numpy(dtype=np.float64), 1.0 - d["p_dir"].to_numpy(dtype=np.float64))
    live = ~np.isfinite(y)
    out = pd.DataFrame({
        "coin": d["coin"].to_numpy(), "time": d["end_time"].to_numpy(),
        "status": np.where(live, "live", "closed"), "signal": np.where(side > 0, "LONG", "SHORT"),
        "confidence_pct": 100.0 * p_side, "uncertainty_pct": 100.0 * sg, "move_pct": 100.0 * mu,
        "true_move_pct": 100.0 * y, "error_pct": 100.0 * (mu - y),
        "edge_ratio": np.abs(mu) / np.where(sg > 0, sg, np.nan), "agrees": np.sign(mu) == side,
        "entry": entry, "target": entry * (1.0 + mu), "target_lo": entry * (1.0 + mu - sg),
        "target_hi": entry * (1.0 + mu + sg), "true_target": entry * (1.0 + y),
        "ok": pd.array(np.where(live, None, side * y > 0), dtype=object)})
    _level_columns(out, d, entry)
    return out.reset_index(drop=True)


def _num(v, spec: str) -> str:
    return "-" if pd.isna(v) else format(v, spec)


def format_latest_table(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """A text copy of ``latest_table`` for printing: percentages rounded (signed for move and error), prices in 6
    significant digits, an unknown value (a live row's truth) shown as ``-``."""
    out = df.copy()
    for c in out.columns:
        if c.endswith("_pct"):
            out[c] = out[c].map(lambda v, c=c: _num(v, f"+.{digits}f" if c in ("move_pct", "true_move_pct", "error_pct")
                                                    else f".{digits}f"))
        elif c == "agrees":
            out[c] = out[c].map(lambda v: "yes" if v else "NO")
        elif c == "edge_ratio":
            out[c] = out[c].map(lambda v: _num(v, ".2f"))
        elif c == "entry" or c.startswith(("target", "true_")):
            out[c] = out[c].map(lambda v: _num(v, ".6g"))
    if "ok" in out:
        out["ok"] = out["ok"].map(lambda v: "-" if v is None or pd.isna(v) else ("yes" if v else "NO"))
    out["time"] = pd.to_datetime(out["time"]).dt.strftime("%Y-%m-%d %H:%M")
    return out


def print_latest_table(df: pd.DataFrame, title: str | None = None, echo=print) -> str:
    """Print (through ``echo``) and return the table text, with a one-line legend."""
    if df is None or df.empty:
        text = "(no samples to show)"
    else:
        text = format_latest_table(df).to_string(index=False)
        text += ("\nconfidence = P(signalled side); uncertainty = predictive sd of the close return; move / error in "
                 "percent of the entry (error = expected - true); target = entry x (1 + move); 'live' rows have no "
                 "truth yet")
    out = f"{title}\n{text}" if title else text
    echo(out)
    return out
