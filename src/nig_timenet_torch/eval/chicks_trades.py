"""PURPOSE: The TF trade selection over the flat frame: keep samples whose predicted move exceeds the uncertainty times
    an edge ratio, rank them, print the best with the hit rate before and after.
TAGS: chicks, trade selection, select and rank trades, edge ratio, min edge ratio, top n, rank score, win rate after
    filter, tf parity
PITFALLS: this is the report over the FLAT frame in price units (``predicted_change``, ``uncertainty``,
    ``confidence`` columns); ``eval.trade_selection.select_trades`` is the same idea over the latest-samples table
    in return units, a different table; the filter is strict (``edge_ratio`` > ``min_edge_ratio``); the combined score
    ranks inside each head when there are several (a head with a higher raw confidence would otherwise take the top);
    a filter that keeps nothing prints a warning and returns an empty frame; the hit-rate line before / after is a
    prediction, not a result: it reads the ``correct`` column of the same samples.
SOURCE: crypto-signal-prediction ``evaluation/trade_selection.py`` (select_and_rank_trades, print_trade_selection).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_outputs import fmt_num, fmt_price
from nig_timenet_torch.eval.chicks_text import Say

__all__ = ["select_and_rank_trades", "print_trade_selection", "trade_accuracy_pct", "SORT_BY", "MIN_EDGE_RATIO",
           "TOP_N"]

SORT_BY = ("confidence", "move", "both", "edge")
MIN_EDGE_RATIO = 1.0
TOP_N = 20
_COMPATIBLE = (("predicted_change", "uncertainty", "confidence"), ("move_%", "uncertainty_%", "confidence_%"))


def _detect_columns(df: pd.DataFrame, move_col=None, uncertainty_col=None, confidence_col=None):
    """The (move, uncertainty, confidence) columns: the three given together, or a known compatible triple."""
    if any(c is not None for c in (move_col, uncertainty_col, confidence_col)):
        missing = [n for n, c in (("move_col", move_col), ("uncertainty_col", uncertainty_col),
                                  ("confidence_col", confidence_col)) if c is None]
        if missing:
            raise ValueError(f"give all three columns together: {missing} missing")
        for c in (move_col, uncertainty_col, confidence_col):
            if c not in df.columns:
                raise KeyError(f"column {c!r} not found; columns: {list(df.columns)}")
        return move_col, uncertainty_col, confidence_col
    for m, u, c in _COMPATIBLE:
        if m in df.columns and u in df.columns and c in df.columns:
            return m, u, c
    raise KeyError(f"no compatible (move, uncertainty, confidence) columns among {_COMPATIBLE}; "
                   f"columns: {list(df.columns)}")


def _rank_score(work: pd.DataFrame, conf, move, sort_by: str) -> np.ndarray:
    if sort_by == "confidence":
        return conf.values
    if sort_by == "move":
        return move.values
    if sort_by == "edge":
        return work["edge_ratio"].values
    if "target" in work.columns and work["target"].nunique() > 1:      # percentile inside each head
        groups = work["target"].values
        conf_pct = conf.groupby(groups).rank(pct=True, na_option="bottom")
        move_pct = move.groupby(groups).rank(pct=True, na_option="bottom")
    else:
        conf_pct, move_pct = conf.rank(pct=True, na_option="bottom"), move.rank(pct=True, na_option="bottom")
    return (conf_pct.values + move_pct.values) / 2.0


def select_and_rank_trades(df: pd.DataFrame, move_col=None, uncertainty_col=None, confidence_col=None,
                           min_edge_ratio: float = MIN_EDGE_RATIO, sort_by: str = "both", ascending: bool = False,
                           top_n: int | None = None, say: Say | None = None) -> pd.DataFrame:
    """The samples with |move| > ``min_edge_ratio`` x uncertainty, ranked by ``sort_by`` ("confidence", "move",
    "edge" or "both"), best first, with ``edge_ratio``, ``direction`` and ``rank_score`` added; ``top_n`` keeps the
    first rows. Empty when nothing passes."""
    say = say or Say()
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    work = df.copy()
    if "kind" in work.columns:
        work = work[work["kind"] == "continuous"].copy()
    if work.empty:
        return work
    mcol, ucol, ccol = _detect_columns(work, move_col, uncertainty_col, confidence_col)
    move, unc, conf = (pd.to_numeric(work[c], errors="coerce") for c in (mcol, ucol, ccol))
    valid = np.isfinite(move) & np.isfinite(unc) & (unc > 0)
    work, move, unc, conf = work.loc[valid].copy(), move[valid], unc[valid], conf[valid]
    edge = move.abs() / unc
    keep = edge > float(min_edge_ratio)
    work = work.loc[keep].copy()
    if work.empty:
        say("trade_none", k=min_edge_ratio, n=int(valid.sum()))
        return work
    work["edge_ratio"] = edge.loc[keep].values
    work["direction"] = np.where(move.loc[keep].values > 0, "▲", "▼")
    if sort_by not in SORT_BY:
        raise ValueError("sort_by must be one of: 'confidence' | 'move' | 'both' | 'edge'")
    work["rank_score"] = _rank_score(work, conf.loc[keep], move.loc[keep].abs(), sort_by)
    work = work.sort_values("rank_score", ascending=ascending).reset_index(drop=True)
    return work.head(int(top_n)) if top_n is not None else work


def trade_accuracy_pct(series: pd.Series) -> tuple[float, int]:
    """(hit rate %, n) of an ok / direction_correct / correct column, ignoring missing values."""
    s = series.dropna()
    if len(s) == 0:
        return float("nan"), 0
    return float(s.astype(bool).mean()) * 100.0, len(s)


def print_trade_selection(result: pd.DataFrame, original: pd.DataFrame | None = None, title: str | None = None,
                          max_rows: int | None = TOP_N, indent: str = "  ", say: Say | None = None) -> None:
    """Print the selected trades compactly (rank, coin, head, direction, confidence, uncertainty, move, edge, the
    predicted and entry price, hit or miss) and the hit rate before and after the filter when ``original`` is given."""
    say = say or Say()
    if result is None or result.empty:
        if title:
            print(title)
        print(indent + say.text("trade_empty"))
        return
    if title:
        print(title)
    show = result.head(max_rows) if max_rows else result
    mcol, ucol, ccol = _detect_columns(show)
    out = pd.DataFrame(index=show.index)
    out["#"] = np.arange(1, len(show) + 1)
    if "date" in show.columns and show["date"].notna().any():
        out["date"] = show["date"].map(lambda d: d.strftime("%Y-%m-%d %H:%M") if pd.notna(d) else "—")
    elif "idx" in show.columns:
        out["#idx"] = show["idx"]
    if "asset" in show.columns:
        out["asset"] = show["asset"]
    out["target"] = show["target"]
    out["dir"] = show["direction"]
    conf_vals = show[ccol].astype(float)
    out["conf%"] = (conf_vals * 100.0 if conf_vals.max(skipna=True) <= 1.5 else conf_vals).map(
        lambda v: fmt_num(v, ".1f"))
    out["unc"] = show[ucol].map(lambda v: fmt_num(v, ".4g"))
    out["move"] = show[mcol].map(lambda v: fmt_num(v, "+.3f"))
    out["edge_x"] = show["edge_ratio"].map(lambda v: fmt_num(v, ".2f"))
    if "pred" in show.columns:
        out["pred"] = show["pred"].map(fmt_price)
    if "entry" in show.columns:
        out["entry"] = show["entry"].map(fmt_price)
    ok_col = next((c for c in ("direction_correct", "ok", "correct") if c in show.columns), None)
    if ok_col:
        out["ok"] = show[ok_col].map(lambda v: "⏳" if (v is None or (isinstance(v, float) and np.isnan(v)))
                                     else ("✅" if bool(v) else "❌"))
    print("\n".join(indent + ln for ln in out.to_string(index=False).split("\n")))
    if max_rows and len(result) > max_rows:
        print(indent + say.text("trade_more", n=len(result) - max_rows))
    ok_col = next((c for c in ("direction_correct", "correct", "ok") if c in result.columns), None)
    if ok_col and original is not None and ok_col in original.columns:
        base_acc, base_n = trade_accuracy_pct(original[ok_col])
        sel_acc, sel_n = trade_accuracy_pct(result[ok_col])
        if base_n and sel_n:
            print("\n" + indent + say.text("trade_rates", b=base_acc, bn=base_n, s=sel_acc, sn=sel_n,
                                    share=sel_n / base_n * 100))
