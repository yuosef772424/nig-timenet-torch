"""PURPOSE: A strategy as strict data (Entry, Exit, Costs, Book): which signal rows become trades and how a trade
    leaves (after N bars, at a take-profit / stop in ATR, %, sigma or predicted-move units); custom rules by name.
TAGS: strategy, entry rules, exit rules, take profit, stop loss, atr multiple, time exit, registry, custom filter,
    strict config, feature check
PITFALLS: a rule that needs an optional feature (confidence, sigma, p_up, pred_high / pred_low, stage) raises when
    the feature is absent or switched off (``check_strategy``); ``top_share`` / ``confidence_top_share`` rank inside
    each stage (a TEST threshold never sees VAL rows and the reverse); a custom filter is ONE function registered in
    ENTRY_FILTERS and named in ``Entry.filters`` (unknown names raise), so a rule written once runs in every strategy.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nig_backtest._base import Registry, from_dict, to_dict

__all__ = ["Entry", "Exit", "Costs", "Book", "Strategy", "ENTRY_FILTERS", "SIDES", "UNITS", "check_strategy",
           "select_entries", "level_fraction", "strategy_from_dict", "strategy_to_dict"]

SIDES = ("score", "long", "short", "p_up")
UNITS = ("atr", "pct", "sigma", "pred", "move")
ENTRY_PRICES = ("next_open", "signal_close")
BOTH_HIT = ("stop_first", "target_first")
#: name -> fn(rows: DataFrame, params: dict) -> bool array (True = keep); register your own rule here
ENTRY_FILTERS = Registry("entry_filter", required_meta=("needs",))


@dataclass(frozen=True)
class Entry:
    side: str = "score"                    # score: the output's direction (kinds); long / short: one side; p_up: > 0.5
    min_strength: float = 0.0              # keep strength >= this (strength: kinds, e.g. |predicted move|)
    top_share: float | None = None         # keep the strongest share of the rows of each stage
    score_over_sigma: float | None = None  # keep strength > k sigma (needs sigma)
    min_confidence: float | None = None    # keep confidence >= this (needs confidence)
    confidence_top_share: float | None = None  # keep the most confident share of each stage (needs confidence)
    per_time_top_k: int | None = None      # cross-section: per time keep the k strongest of each side
    coins: tuple = ()                      # only these coins (empty = all)
    exclude_coins: tuple = ()
    stages: tuple = ()                     # only these stages (empty = all; needs stage)
    filters: tuple = ()                    # names of ENTRY_FILTERS, applied in order
    filter_params: dict = field(default_factory=dict)  # {filter name: {param: value}}
    entry_price: str = "next_open"         # next_open: first bar opening at / after time; signal_close: close at time
    one_per_coin: bool = True              # skip a signal while the coin's previous trade is still open


@dataclass(frozen=True)
class Exit:
    max_bars: int = 2                      # time exit at the close of the N-th PRICE bar held ("after N candles")
    hold: str | None = None                # or a duration ("8h"; "horizon" = SignalSet.horizon): replaces max_bars,
                                           #   so a 4h model runs on 1h candles with the same holding time
    take_profit: float | None = None       # distance in take_profit_unit (None = no target)
    take_profit_unit: str = "atr"          # atr: x ATR at entry; pct: fraction of the entry price; sigma: x sigma;
    stop_loss: float | None = None         #   pred: x the predicted extreme (pred_high / pred_low); move: x the
                                           #   decoded predicted move (absolute output kinds only)
    stop_loss_unit: str = "atr"
    atr_bars: int = 14
    atr_timeframe: str | None = None       # ATR on candles resampled to this frame (e.g. the model's "4h"); None = the
                                           #   price candles' own frame
    both_hit: str = "stop_first"           # a bar that touches target and stop: which one counts (no intrabar data)


@dataclass(frozen=True)
class Costs:
    fee_per_side: float = 0.0008
    slippage_per_side: float = 0.0


@dataclass(frozen=True)
class Book:
    risk_per_trade: float = 0.02           # share of the capital put on one trade
    max_open: int | None = None            # at most this many trades open at once (later ones are skipped)


@dataclass(frozen=True)
class Strategy:
    name: str = "strategy"
    entry: Entry = field(default_factory=Entry)
    exit: Exit = field(default_factory=Exit)
    costs: Costs = field(default_factory=Costs)
    book: Book = field(default_factory=Book)

    def __post_init__(self):
        e, x = self.entry, self.exit
        for name, value, allowed in (("entry.side", e.side, SIDES), ("entry.entry_price", e.entry_price, ENTRY_PRICES),
                                     ("exit.take_profit_unit", x.take_profit_unit, UNITS),
                                     ("exit.stop_loss_unit", x.stop_loss_unit, UNITS),
                                     ("exit.both_hit", x.both_hit, BOTH_HIT)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
        if x.max_bars < 1 or x.atr_bars < 1:
            raise ValueError("exit.max_bars and exit.atr_bars must be >= 1")
        if x.atr_timeframe == "horizon":
            raise ValueError("exit.atr_timeframe takes a duration, not 'horizon'")
        for name, v in (("exit.hold", x.hold), ("exit.atr_timeframe", x.atr_timeframe)):
            if v is not None and v != "horizon" and pd.Timedelta(v) <= pd.Timedelta(0):
                raise ValueError(f"{name} must be a positive duration, got {v!r}")
        for name, v in (("top_share", e.top_share), ("confidence_top_share", e.confidence_top_share)):
            if v is not None and not 0 < v <= 1:
                raise ValueError(f"entry.{name} must be in (0, 1], got {v}")
        for name in e.filters:
            ENTRY_FILTERS.entry(name)          # unknown names raise with the known ones
        unknown = sorted(set(e.filter_params) - set(e.filters))
        if unknown:
            raise ValueError(f"entry.filter_params for filters not in entry.filters: {unknown}")


def strategy_from_dict(d: dict) -> Strategy:
    """A strategy from its JSON form (unknown keys raise)."""
    d = dict(d)
    entry = dict(d.get("entry", {}))
    for k in ("coins", "exclude_coins", "stages", "filters"):
        if k in entry:
            entry[k] = tuple(entry[k])
    d["entry"] = entry
    return from_dict(Strategy, d)


def strategy_to_dict(s: Strategy) -> dict:
    return to_dict(s)


def _needs(s: Strategy) -> set[str]:
    e, x = s.entry, s.exit
    need = set()
    need |= {"sigma"} if e.score_over_sigma is not None else set()
    need |= {"confidence"} if e.min_confidence is not None or e.confidence_top_share is not None else set()
    need |= {"p_up"} if e.side == "p_up" else set()
    need |= {"stage"} if e.stages else set()
    for value, unit in ((x.take_profit, x.take_profit_unit), (x.stop_loss, x.stop_loss_unit)):
        if value is not None and unit == "sigma":
            need.add("sigma")
        if value is not None and unit == "pred":
            need |= {"pred_high", "pred_low"}
    for name in e.filters:
        need |= set(ENTRY_FILTERS.entry(name).meta["needs"])
    return need


def check_strategy(s: Strategy, enabled_features, absolute: bool = False, horizon: str | None = None) -> None:
    """Raise when a rule needs an optional feature that the signal set does not have (absent or switched off), a
    'move' exit is asked of an output that does not decode to a price without loss, or ``hold: "horizon"`` has no
    horizon."""
    missing = sorted(_needs(s) - set(enabled_features))
    if missing:
        raise ValueError(f"strategy {s.name!r} needs the features {missing}, which the signals do not provide or "
                         f"SignalSet.use switched off (enabled: {sorted(enabled_features)})")
    x = s.exit
    moves = [u for v, u in ((x.take_profit, x.take_profit_unit), (x.stop_loss, x.stop_loss_unit)) if v is not None]
    if "move" in moves and not absolute:
        raise ValueError(f"strategy {s.name!r}: an exit in 'move' units needs an absolute output kind (return, "
                         f"log_return, price); this output gives a direction and a ranking only (use atr / pct / "
                         f"sigma units)")
    if x.hold == "horizon" and horizon is None:
        raise ValueError(f"strategy {s.name!r}: exit.hold 'horizon' needs SignalSet(horizon=...)")


def _rank_share(values: np.ndarray, groups: np.ndarray, share: float) -> np.ndarray:
    """True for the top ``share`` of the finite ``values`` inside each group (ties broken by row order); -inf marks a
    row already removed, so the share is of the rows still kept."""
    keep = np.zeros(len(values), dtype=bool)
    for g in pd.unique(groups):
        idx = np.flatnonzero((groups == g) & np.isfinite(values))
        if not len(idx):
            continue
        k = max(1, int(round(len(idx) * share)))
        keep[idx[np.argsort(-values[idx], kind="stable")[:k]]] = True
    return keep


def _side(rows: pd.DataFrame, side: str) -> np.ndarray:
    if side == "score":
        return rows["dir"].to_numpy()
    if side == "p_up":
        return np.where(rows["p_up"].to_numpy() > 0.5, 1.0, -1.0)
    return np.full(len(rows), 1.0 if side == "long" else -1.0)


def select_entries(rows: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    """The rows that become trades, with their ``side`` (+1 long / -1 short) and the ``rule_trace`` counts in
    ``attrs`` (rows kept after each rule, in order: what each decision removed)."""
    e = s.entry
    strength = rows["strength"].to_numpy()
    stage = rows["stage"].to_numpy() if "stage" in rows else np.zeros(len(rows))
    side = _side(rows, e.side)
    keep = side != 0
    if e.side in ("long", "short"):
        keep &= rows["dir"].to_numpy() == side
    trace = [("signals", len(rows)), ("side", int(keep.sum()))]

    def step(name, mask):
        nonlocal keep
        keep &= mask
        trace.append((name, int(keep.sum())))

    if e.coins:
        step("coins", rows["coin"].isin(e.coins).to_numpy())
    if e.exclude_coins:
        step("exclude_coins", ~rows["coin"].isin(e.exclude_coins).to_numpy())
    if e.stages:
        step("stages", rows["stage"].isin(e.stages).to_numpy())
    if e.min_strength > 0:
        step("min_strength", strength >= e.min_strength)
    if e.score_over_sigma is not None:
        step("score_over_sigma", strength > e.score_over_sigma * rows["sigma"].to_numpy())
    if e.min_confidence is not None:
        step("min_confidence", rows["confidence"].to_numpy() >= e.min_confidence)
    for name in e.filters:
        fn = ENTRY_FILTERS.get(name)
        step(f"filter:{name}", np.asarray(fn(rows, dict(e.filter_params.get(name, {}))), dtype=bool))
    if e.top_share is not None:
        step("top_share", _rank_share(np.where(keep, strength, -np.inf), stage, e.top_share) & keep)
    if e.confidence_top_share is not None:
        conf = np.where(keep, rows["confidence"].to_numpy(), -np.inf)
        step("confidence_top_share", _rank_share(conf, stage, e.confidence_top_share) & keep)
    if e.per_time_top_k is not None:
        step("per_time_top_k", _per_time(rows, side, keep, e.per_time_top_k))
    out = rows[keep].copy()
    out["side"] = side[keep]
    out.attrs["rule_trace"] = trace
    return out


def _per_time(rows: pd.DataFrame, side: np.ndarray, keep: np.ndarray, k: int) -> np.ndarray:
    """Per time and side, the k strongest rows (an equal-count long / short book)."""
    frame = pd.DataFrame({"t": rows["time"].to_numpy(), "side": side, "a": rows["strength"].to_numpy(),
                          "keep": keep})
    frame = frame[frame["keep"]]
    rank = frame.groupby(["t", "side"])["a"].rank(ascending=False, method="first")
    out = np.zeros(len(rows), dtype=bool)
    out[frame.index[rank.to_numpy() <= k]] = True
    return out


def level_fraction(rows: pd.DataFrame, value: float | None, unit: str, atr_at_entry: np.ndarray,
                   entry_price: np.ndarray, kind: str) -> np.ndarray:
    """The distance of a target / stop from the entry as a FRACTION of the entry price, per trade (NaN = none).
    ``kind`` "target" or "stop"; 'move' uses the decoded predicted move; 'pred' uses the predicted extreme of the
    trade's direction for a target and of the opposite direction for a stop."""
    n = len(rows)
    if value is None:
        return np.full(n, np.nan)
    if unit == "pct":
        return np.full(n, float(value))
    if unit == "atr":
        return value * atr_at_entry / entry_price
    if unit == "sigma":
        return value * rows["sigma"].to_numpy()
    if unit == "move":
        return value * np.abs(rows["pred_move"].to_numpy())
    side = rows["side"].to_numpy()
    up, down = rows["pred_high"].to_numpy(), -rows["pred_low"].to_numpy()
    toward = (side > 0) == (kind == "target")
    return value * np.abs(np.where(toward, up, down))


def _filter_meta(*needs):
    return {"needs": tuple(needs)}


def _quiet_window(rows: pd.DataFrame, params: dict) -> np.ndarray:
    """Example custom rule: keep rows whose free column ``x_window_vol`` (any volatility the caller supplies) is above
    ``min`` (the failure-chart audit's 'do not fade a quiet base' idea)."""
    col = params.get("column", "x_window_vol")
    if col not in rows:
        raise ValueError(f"filter min_column: the signal table has no column {col!r}")
    return rows[col].to_numpy() >= float(params["min"])


ENTRY_FILTERS.register("min_column", _quiet_window, version=1, **_filter_meta())
