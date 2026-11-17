"""PURPOSE: What a model output MEANS (OUTPUT_KINDS): each kind turns ``score`` into a direction, a strength that ranks
    rows and, only when it decodes to a price without loss (absolute), the predicted move used by 'move' exits.
TAGS: output kind, decode, absolute output, ordinal output, rank, probability, class, predicted price, return, lossless
PITFALLS: an absolute kind (return, log_return, price) gives ``pred_move`` (the predicted return over the horizon); a
    lossy kind (score, rank, probability, class) gives NaN there, so an exit in 'move' units raises for it instead of
    guessing a price; 'price' needs ``ref_price`` (the price at ``time``); 'rank' is read inside each time (direction =
    above / below that time's median), so its trades are relative to the cross-section, not to the market.
"""

import numpy as np
import pandas as pd

from nig_backtest._base import Registry

__all__ = ["OUTPUT_KINDS", "decode", "is_absolute", "DECODED"]

#: name -> fn(rows) -> (direction +1 / -1 / 0, strength >= 0, predicted move or NaN); meta absolute, needs
OUTPUT_KINDS = Registry("output_kind", required_meta=("absolute", "needs"))
DECODED = ("dir", "strength", "pred_move", "period")


def _nan(n):
    return np.full(n, np.nan)


def _score(rows):
    s = rows["score"].to_numpy()
    return np.sign(s), np.abs(s), _nan(len(s))


def _return(rows):
    s = rows["score"].to_numpy()
    return np.sign(s), np.abs(s), s


def _log_return(rows):
    m = np.expm1(rows["score"].to_numpy())
    return np.sign(m), np.abs(m), m


def _price(rows):
    m = rows["score"].to_numpy() / rows["ref_price"].to_numpy() - 1.0
    return np.sign(m), np.abs(m), m


def _rank(rows):
    s = rows["score"].to_numpy()
    med = pd.Series(s).groupby(rows["time"].to_numpy()).transform("median").to_numpy()
    d = s - med
    return np.sign(d), np.abs(d), _nan(len(s))


def _probability(rows):
    p = rows["score"].to_numpy()
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("output kind 'probability' needs scores in [0, 1]")
    return np.sign(p - 0.5), np.abs(p - 0.5), _nan(len(p))


def _class(rows):
    s = rows["score"].to_numpy()
    if not np.isin(s, (-1.0, 0.0, 1.0)).all():
        raise ValueError("output kind 'class' needs scores in {-1, 0, 1}")
    return s, np.abs(s), _nan(len(s))


OUTPUT_KINDS.register("score", _score, absolute=False, needs=())            # any directional number (default)
OUTPUT_KINDS.register("return", _return, absolute=True, needs=())           # predicted simple return over the horizon
OUTPUT_KINDS.register("log_return", _log_return, absolute=True, needs=())   # predicted log return
OUTPUT_KINDS.register("price", _price, absolute=True, needs=("ref_price",))  # predicted price at the horizon
OUTPUT_KINDS.register("rank", _rank, absolute=False, needs=())              # ordinal / cross-section rank
OUTPUT_KINDS.register("probability", _probability, absolute=False, needs=())  # p(up)
OUTPUT_KINDS.register("class", _class, absolute=False, needs=())            # -1 / 0 / +1


def is_absolute(kind: str) -> bool:
    return bool(OUTPUT_KINDS.entry(kind).meta["absolute"])


def decode(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """``rows`` with the columns ``dir``, ``strength`` and ``pred_move`` of ``kind`` (unknown kinds raise)."""
    entry = OUTPUT_KINDS.entry(kind)
    missing = [c for c in entry.meta["needs"] if c not in rows]
    if missing:
        raise ValueError(f"output kind {kind!r} needs the columns {missing}")
    taken = [c for c in DECODED if c in rows]
    if taken:
        raise ValueError(f"the signal table may not hold the decoded column names {taken}")
    d, s, m = entry.obj(rows)
    return rows.assign(dir=np.asarray(d, float), strength=np.asarray(s, float), pred_move=np.asarray(m, float))
