"""PURPOSE: The few generic pieces the backtest needs from nobody: a name registry with required metadata, strict
    frozen-dataclass loading from dicts (unknown keys raise), OHLCV names and a per-trade tearsheet.
TAGS: registry, strict config, from dict, to dict, tearsheet, sharpe, sortino, drawdown, portable, no project import
PITFALLS: kept here on purpose instead of importing the host project: nig_backtest must move to another project as a
    folder (numpy / pandas only). Equity of the tearsheet risks ``risk`` of the capital per trade in order.
"""

import dataclasses
import difflib
import math
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["Registry", "UnknownNameError", "from_dict", "to_dict", "OHLCV", "tearsheet", "equity_curve"]

OHLCV = ("open", "high", "low", "close", "volume")


class UnknownNameError(KeyError, ValueError):
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class _Entry:
    obj: object
    version: int
    meta: dict


class Registry:
    """``register(name, obj, version=1, **meta)`` with the ``required_meta`` keys; unknown names raise with the known
    names and a did-you-mean; a duplicate name raises."""

    def __init__(self, kind: str, required_meta: tuple = ()):
        self.kind, self.required_meta, self._items = kind, tuple(required_meta), {}

    def register(self, name: str, obj, version: int = 1, **meta):
        if name in self._items:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        missing = [k for k in self.required_meta if k not in meta]
        if missing:
            raise ValueError(f"{self.kind} {name!r}: missing metadata {missing}")
        self._items[name] = _Entry(obj, int(version), dict(meta))
        return obj

    def entry(self, name: str) -> _Entry:
        if name not in self._items:
            close = difflib.get_close_matches(str(name), list(self._items), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise UnknownNameError(f"unknown {self.kind} {name!r} (known: {sorted(self._items)}){hint}")
        return self._items[name]

    def get(self, name: str):
        return self.entry(name).obj

    def names(self) -> list[str]:
        return sorted(self._items)


def _convert(tp, value):
    if dataclasses.is_dataclass(tp) and isinstance(value, Mapping):
        return from_dict(tp, value)
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        for arg in typing.get_args(tp):
            if dataclasses.is_dataclass(arg) and isinstance(value, Mapping):
                return from_dict(arg, value)
    if (tp is tuple or origin is tuple) and isinstance(value, list):
        return tuple(value)
    return value


def from_dict(cls, d: Mapping):
    """A frozen dataclass from a mapping: unknown keys raise (with a did-you-mean), nested dataclasses are built,
    lists become tuples where the field is a tuple; missing keys take the field default."""
    if not isinstance(d, Mapping):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in d if k not in fields]
    if unknown:
        close = {k: difflib.get_close_matches(k, list(fields), n=1) for k in unknown}
        hint = "; ".join(f"{k!r} (did you mean {v[0]!r}?)" if v else repr(k) for k, v in close.items())
        raise ValueError(f"{cls.__name__}: unknown key {hint}")
    return cls(**{k: _convert(hints.get(k), v) for k, v in d.items()})


def to_dict(obj) -> dict:
    """The plain nested dict of a dataclass (tuples become lists), the inverse of ``from_dict``."""
    def conv(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v) if f.init}
        if isinstance(v, Mapping):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return conv(obj)


def equity_curve(returns, risk: float, start: float = 1.0) -> dict:
    """Capital after each trade when ``risk`` of it goes on every trade (a loss below -100 % is clipped)."""
    r = np.clip(np.asarray(returns, dtype=float), -1.0, None)
    eq = start * np.concatenate([[1.0], np.cumprod(1.0 + risk * r)])
    return {"equity": eq, "drawdown": eq / np.maximum.accumulate(eq) - 1.0}


def tearsheet(returns, risk: float = 0.02) -> dict:
    """n, win rate, mean, median, Sharpe and Sortino per trade, profit factor (= Omega at 0 for equal-weight trades),
    SQN, payoff, the longest win / loss streaks, final capital and max drawdown at ``risk`` per trade."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return {"n": int(len(r))}
    sd, down = r.std(ddof=1), r[r < 0]
    dsd = math.sqrt(float(np.mean(np.square(np.minimum(r, 0.0)))))
    gain, loss = r[r > 0].sum(), -down.sum()
    streak, run, last = {1.0: 0, -1.0: 0}, 0, 0.0
    for s in np.sign(r):
        run = run + 1 if s == last else 1
        last = s
        if s in streak:
            streak[s] = max(streak[s], run)
    eq = equity_curve(r, risk)
    pf = float(gain / loss) if loss > 0 else math.inf
    return {"n": int(len(r)), "win_rate": float((r > 0).mean()), "mean": float(r.mean()),
            "median": float(np.median(r)), "sharpe": float(r.mean() / sd) if sd > 0 else 0.0,
            "sortino": float(r.mean() / dsd) if dsd > 0 else math.inf, "profit_factor": pf,
            "sqn": float(math.sqrt(min(len(r), 100)) * r.mean() / sd) if sd > 0 else 0.0,
            "payoff": float(r[r > 0].mean() / -down.mean()) if len(down) and (r > 0).any() else math.nan,
            "max_win_streak": int(streak[1.0]), "max_loss_streak": int(streak[-1.0]),
            "final_equity": float(eq["equity"][-1]), "max_drawdown": float(eq["drawdown"].min())}
