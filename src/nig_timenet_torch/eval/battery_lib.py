"""The battery private helpers: trade stats, row masks, section shaping. Split out of battery.py
so the classic sections (battery_classics.py) and the core sections share ONE implementation
(rule C2: one implementation per metric).
"""

from __future__ import annotations

import numpy as np

__all__ = ["_stats", "_scored", "_top_abs", "_dir_pnl", "_section", "_fmt", "_HEADER"]


def _stats(pnl: np.ndarray, min_n: int = 50) -> dict | None:
    """n, win rate, average, Sharpe (per-trade, unannualized) and profit factor - None below min_n."""
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < min_n:
        return None
    s = pnl.std(ddof=1)
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    return {"n": int(len(pnl)), "win": float((pnl > 0).mean()), "avg": float(pnl.mean()),
            "sharpe": float(pnl.mean() / s) if s > 0 else 0.0,
            "pf": float(gross_win / gross_loss) if gross_loss > 1e-12 else float("inf")}


def _scored(df) -> "np.ndarray":
    """The non-flat rows: a flat truth teaches nothing about direction."""
    return (df["dir_true"] != 0).values


def _top_abs(df, field: str, pct: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the strongest `pct` fraction by |field| (optionally inside a mask)."""
    base = np.where(mask if mask is not None else np.ones(len(df), dtype=bool))[0]
    strength = np.abs(df[field].values[base])
    k = max(1, int(len(base) * pct))
    return base[np.argsort(-strength)[:k]]


def _dir_pnl(df, idx: np.ndarray, cost: float) -> np.ndarray:
    """Trade the model's direction (p_dir) on the true return, minus cost."""
    sign = np.where(df["p_dir"].values[idx] > 0.5, 1.0, -1.0)
    return sign * df["y_ret"].values[idx] - cost


def _section(title: str, rows: list[str], note: str = "") -> dict:
    return {"title": title, "rows": rows, "note": note}


def _fmt(label: str, s: dict | None) -> str:
    if s is None:
        return f"{label:<44}(n<50)"
    return (f"{label:<44}{s['n']:>8,}{s['win'] * 100:>7.2f}%{s['avg'] * 100:>+9.4f}%"
            f"{s['sharpe']:>+9.3f}{s['pf']:>7.2f}")


_HEADER = f"{'strategy':<44}{'n':>8}{'win%':>8}{'avg%':>10}{'Sharpe':>9}{'PF':>7}"


