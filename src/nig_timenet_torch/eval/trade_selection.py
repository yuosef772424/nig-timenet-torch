"""PURPOSE: The TF trade selection over the latest-samples table: keep the rows whose expected move exceeds k times its
    uncertainty (and clears the round-trip cost), rank them by confidence, move size, edge or a combined score.
TAGS: trade selection, select and rank trades, edge vs uncertainty, move over sd, rank by confidence, combined score,
    latest table, round trip cost, tf parity
PITFALLS: a pure function of ``eval.latest_table.latest_table`` (live rows included: selection needs no truth); the
    filter is strict (``edge_ratio`` > k: a row with edge exactly k is dropped, as in TF) so ``k`` = 0 keeps every
    row with a non-zero move and a positive uncertainty; ``combined`` is the mean of the two percentile ranks
    (confidence, |move|) inside the kept rows, so it is relative to the kept set and not comparable across calls;
    ``after_cost_pct`` is |move| minus the round trip of ``cost`` per side, a prediction not a result;
    ``require_agreement`` drops rows whose class head and mu point opposite ways. ``eval.legacy_reports.move_over_sd``
    is the SAME rule judged on a whole stage. Exploratory (rule C5).

Source: crypto-signal-prediction ``evaluation/trade_selection.py`` (``select_and_rank_trades``): a trade is worth
trusting only when the expected move is larger than the uncertainty of the prediction.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.costs import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.eval.battery_lib import round_trip

__all__ = ["select_trades", "RANK_BY"]

RANK_BY = ("confidence", "move", "edge", "combined")
_NEEDED = ("confidence_pct", "uncertainty_pct", "move_pct", "edge_ratio", "agrees")


def _pct_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, method="average")


def _score(kept: pd.DataFrame, rank_by: str) -> pd.Series:
    if rank_by == "confidence":
        return kept["confidence_pct"]
    if rank_by == "move":
        return kept["move_pct"].abs()
    if rank_by == "edge":
        return kept["edge_ratio"]
    return (_pct_rank(kept["confidence_pct"]) + _pct_rank(kept["move_pct"].abs())) / 2.0


def select_trades(latest: pd.DataFrame, k: float = 1.0, rank_by: str = "combined", *, top: int | None = None,
                  min_confidence: float = 0.0, cost: float = DEFAULT_COST_PER_SIDE,
                  require_agreement: bool = False, min_after_cost: float | None = None) -> pd.DataFrame:
    """The rows of ``latest`` (an ``eval.latest_table.latest_table`` frame) that pass the selection, best first, with
    ``rank`` (1 = best), ``score`` and ``after_cost_pct`` added. ``k``: keep ``edge_ratio`` > k; ``min_confidence``
    (percent, of the signalled side); ``top``: at most this many rows; ``min_after_cost`` (percent): keep rows whose
    |move| minus the round trip is at least this."""
    if rank_by not in RANK_BY:
        raise ValueError(f"rank_by must be one of {RANK_BY}, not {rank_by!r}")
    missing = [c for c in _NEEDED if c not in latest.columns]
    if missing:
        raise ValueError(f"select_trades: the table lacks {missing}; pass the output of latest_table")
    if k < 0 or not 0.0 <= min_confidence <= 100.0:
        raise ValueError("select_trades: k must be >= 0 and min_confidence in [0, 100]")
    after = latest["move_pct"].abs() - 100.0 * round_trip(cost)
    keep = (latest["edge_ratio"] > k) & (latest["confidence_pct"] >= min_confidence) & latest["uncertainty_pct"].gt(0)
    if require_agreement:
        keep &= latest["agrees"].astype(bool)
    if min_after_cost is not None:
        keep &= after >= min_after_cost
    kept = latest[keep.to_numpy()].assign(after_cost_pct=after[keep.to_numpy()])
    if kept.empty:
        return kept.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype="int64"))
    kept = kept.assign(score=_score(kept, rank_by)).sort_values("score", ascending=False, kind="stable")
    kept = kept.assign(rank=np.arange(1, len(kept) + 1))
    return (kept.head(top) if top else kept).reset_index(drop=True)
