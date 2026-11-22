"""Capital account: a book of the battery table run as money (fixed slots, cash, fee on turnover, compounding)

PURPOSE: Runs a book of the battery table as a capital account (sub-books per horizon phase, fixed slot per coin,
    cash, fee on turnover, compounding): CAGR / Sharpe / max drawdown / years beside always-long, BTC, a timing placebo.
TAGS: account, capital, cagr, sharpe, max drawdown, turnover, fixed slots, timing placebo, buy and hold, battery
PITFALLS: the long book is mu >= the VAL quantile of mu, long whatever p_dir says (the audit's book); every TEST row
    with a finite y_ret and mu is kept, flat rows included (E5); one sub-book per row step inside the label horizon,
    the horizon read from last_close (or given); fee is PER SIDE on the weight change; a period with no pick is cash.

Audit 2026-11-21 (`docs/results/independent-review-001.md`, E2 / E3): the battery's "mean day" is the equal-weight mean
of a day's trades, not the return of money, and every row pays a full round trip even when the coin is kept. This
module is the account view (ROADMAP N8): the capital is split into ``n_books`` sub-books, one per phase of the row
step inside the label horizon (an 8 h label on 4 h rows = 2 sub-books, each rebalanced every 8 h on its own grid),
each sub-book gives every picked coin a fixed slot of ``1 / slots`` (the rest stays in cash), pays ``cost`` per side
on the weight it changes (a coin kept from one period to the next pays nothing) and compounds; the account is the mean
of the sub-book equity curves. Benchmarks under the same account: always long on the same slots, BTC buy-and-hold, a
placebo that picks RANDOM coins at the same timestamps in the same numbers (timing without coin choice), and the
equal-weight basket held with the book's breadth as its exposure (timing alone). ``account_section`` reproduces the
audit scripts a3 (``sizing="equal"``), a4 (fixed slots), a5 (placebo, fee grid) and a6 (breadth timing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE, section

__all__ = ["account_section", "account_frame", "account_curve", "account_stats", "horizon_steps", "row_step",
           "book_weights", "placebo_weights", "ACCOUNT_QUANTILE", "BENCHMARK_COIN", "N_PLACEBO", "PLACEBO_SEED",
           "FEE_GRID", "FEES", "SIZINGS", "MAX_HORIZON_STEPS"]

#: the long book: TEST rows with mu at or above this quantile of the fold's VAL mu (the book of dw1_read / the audit)
ACCOUNT_QUANTILE = 0.90
#: the buy-and-hold benchmark coin (the market reference of the data, ``data.settings.MarketSettings``)
BENCHMARK_COIN = "BTCUSDT"
#: random-coin timing placebo draws (same timestamps, same number of picks) and their seed
N_PLACEBO = 20
PLACEBO_SEED = 0
#: other per-side costs the model book is also run at (audit a5)
FEE_GRID = (0.0002, 0.0005, 0.0010, 0.0015)
#: the longest label horizon, in row steps, looked for in ``horizon_steps``
MAX_HORIZON_STEPS = 48
#: a label horizon is accepted when at least this share of rows has y_ret == last_close(t + h) / last_close(t) - 1
_HORIZON_MATCH = 0.9


def _turnover(w: pd.DataFrame) -> pd.Series:
    """Weight bought plus weight sold per period (the first period buys its whole book)."""
    return w.diff().fillna(w).abs().sum(axis=1)


def _round_trip(w: pd.DataFrame) -> pd.Series:
    """The battery's charge (``battery_lib.round_trip``): every held weight enters and exits each period."""
    return 2.0 * w.abs().sum(axis=1)


#: fee bases: the weight a period trades (``turnover``, a real account) or a round trip on all it holds (the battery)
FEES = {"turnover": _turnover, "round_trip": _round_trip}


def _slots(picks: pd.DataFrame, slots: int) -> pd.DataFrame:
    w = picks.astype(np.float64) / slots
    total = w.sum(axis=1)
    return w.div(total.where(total > 1.0, 1.0), axis=0)          # more picks than slots: share the capital


def _equal(picks: pd.DataFrame, slots: int) -> pd.DataFrame:
    n = picks.sum(axis=1)
    return picks.astype(np.float64).div(n.where(n > 0, 1), axis=0)


#: position sizing: a fixed slot of 1 / slots per pick, the rest in cash (``slots``), or all the capital spread over
#: the picks of the period (``equal``, the audit's a3: one pick holds the whole sub-book)
SIZINGS = {"slots": _slots, "equal": _equal}


def row_step(times) -> pd.Timedelta:
    """The most common gap between consecutive distinct timestamps (the row step of the table)."""
    t = np.unique(pd.to_datetime(np.asarray(times)).values)
    if len(t) < 2:
        raise ValueError("the account needs at least two distinct timestamps")
    gaps, counts = np.unique(np.diff(t), return_counts=True)
    return pd.Timedelta(gaps[np.argmax(counts)])


def horizon_steps(frame: pd.DataFrame, step: pd.Timedelta | None = None) -> int:
    """The label horizon in row steps, read from the table itself: the lag ``h`` at which ``y_ret`` equals
    ``last_close(t + h steps) / last_close(t) - 1`` on the coin with the most rows. Raises when the table has no
    ``last_close`` or no lag up to ``MAX_HORIZON_STEPS`` matches ``_HORIZON_MATCH`` of the rows."""
    if "last_close" not in frame.columns:
        raise ValueError("horizon_steps: the table has no last_close (pass the number of sub-books)")
    step = step or row_step(frame["end_time"])
    coin = frame["coin"].value_counts().index[0]
    one = frame[frame["coin"] == coin].set_index("end_time").sort_index()
    grid = pd.date_range(one.index[0], one.index[-1], freq=step)
    close, y = one["last_close"].reindex(grid), one["y_ret"].reindex(grid)
    best, share = 0, 0.0
    for h in range(1, MAX_HORIZON_STEPS + 1):
        pred = close.shift(-h) / close - 1.0
        ok = pred.notna() & y.notna()
        if ok.sum() == 0:
            break
        s = float((np.abs(pred[ok] - y[ok]) <= 1e-5 * (1.0 + np.abs(y[ok]))).mean())
        if s > share:
            best, share = h, s
    if share < _HORIZON_MATCH:
        raise ValueError(f"horizon_steps: no lag <= {MAX_HORIZON_STEPS} steps explains y_ret (best {best}: "
                         f"{share:.0%} of rows); pass the number of sub-books")
    return best


def account_frame(folds, quantile: float = ACCOUNT_QUANTILE) -> pd.DataFrame:
    """The TEST rows of one or more folds ``[(val, test), ...]`` with the column ``long`` (mu >= the quantile of that
    fold's VAL mu) and ``fold``; rows without a finite y_ret or mu are dropped, flat rows are kept (E5)."""
    parts = []
    for k, (val, test) in enumerate(folds):
        thr = float(np.quantile(val["mu"].to_numpy(dtype=np.float64), quantile))
        t = test.loc[np.isfinite(test["y_ret"]) & np.isfinite(test["mu"])].copy()
        t["long"] = t["mu"].to_numpy() >= thr
        t["fold"] = k
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def account_curve(w: pd.DataFrame, r: pd.DataFrame, cost: float, n_books: int, step: pd.Timedelta,
                  fee: str = "turnover") -> pd.Series:
    """The account's equity curve: ``w`` (time x instrument weights, cash = 1 - row sum) earns ``r`` (the same shape,
    simple returns over the label horizon) in ``n_books`` sub-books, one per phase of ``step`` inside the horizon; each
    sub-book pays ``cost`` per side on ``FEES[fee]`` and compounds; the account is the mean of the sub-book curves."""
    pos = np.rint((w.index - w.index[0]) / step).astype(np.int64)
    curves = []
    for phase in range(n_books):
        m = pos % n_books == phase
        if not m.any():
            continue
        wp, rp = w.loc[m], r.loc[m].reindex(columns=w.columns).fillna(0.0)
        ret = (wp * rp).sum(axis=1) - cost * FEES[fee](wp)
        curves.append((1.0 + ret).cumprod())
    return pd.concat(curves, axis=1, sort=True).ffill().fillna(1.0).mean(axis=1)


def account_stats(eq: pd.Series, step: pd.Timedelta, exposure: float = float("nan")) -> dict:
    """Final multiple, CAGR, annualised Sharpe of the period returns, max drawdown and the return of each calendar
    year (the first year from 1.0, the last one partial) of an equity curve."""
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    sd = float(r.std())
    yearly = eq.resample("YE").last()
    yearly = pd.concat([pd.Series([1.0]), pd.Series(yearly.to_numpy())]).pct_change().dropna().to_numpy()
    final = float(eq.iloc[-1])
    return {"final": final, "cagr": final ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
            "sharpe": float(r.mean()) / sd * float(np.sqrt(pd.Timedelta(days=365) / step)) if sd > 0 else 0.0,
            "max_dd": float((eq / eq.cummax() - 1.0).min()), "exposure": float(exposure),
            "years": {str(y): float(v) for y, v in zip(eq.resample("YE").last().index.year, yearly, strict=True)}}


def book_weights(picks: pd.DataFrame, slots: int, sizing: str = "slots") -> pd.DataFrame:
    """Weights of a book from its picks (time x coin booleans) with ``SIZINGS[sizing]``."""
    return SIZINGS[sizing](picks, slots)


def placebo_weights(picks: pd.DataFrame, present: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The same number of picks per timestamp as ``picks``, drawn at random among the coins ``present`` then."""
    u = np.where(present.to_numpy(), rng.random(present.shape), np.inf)
    rank = np.argsort(np.argsort(u, axis=1, kind="stable"), axis=1)
    return pd.DataFrame(rank < picks.sum(axis=1).to_numpy()[:, None], index=picks.index, columns=picks.columns)


def _line(name: str, s: dict) -> str:
    years = " ".join(f"{v * 100:+.0f}%" for v in s["years"].values())
    expo = f"  expo {s['exposure']:.2f}" if s["exposure"] == s["exposure"] else ""
    return (f"  {name:<50} x{s['final']:6.2f}  CAGR {s['cagr'] * 100:+7.1f}%  Sharpe {s['sharpe']:5.2f}"
            f"  maxDD {s['max_dd'] * 100:6.1f}%{expo}  years {years}")


def _books(frame: pd.DataFrame, slots: int | None, benchmark: str):
    r = frame.pivot(index="end_time", columns="coin", values="y_ret").sort_index()
    present = r.notna()
    picks = frame.pivot(index="end_time", columns="coin", values="long").reindex_like(r).fillna(False).astype(bool)
    slots = int(slots or r.shape[1])
    books = {"model": picks, "always long": present}
    if benchmark in r.columns:
        hold = present.copy()
        hold.loc[:, hold.columns != benchmark] = False
        books["buy and hold"] = hold
    return r, present, picks, slots, books


def _expo(w: pd.DataFrame) -> float:
    return float(w.sum(axis=1).mean())


def account_section(folds, cost: float = DEFAULT_COST_PER_SIDE, *, n_books: int | None = None,
                    quantile: float = ACCOUNT_QUANTILE, slots: int | None = None, n_placebo: int = N_PLACEBO,
                    seed: int = PLACEBO_SEED, fee_grid: tuple = FEE_GRID, benchmark: str = BENCHMARK_COIN) -> dict:
    """The ACCOUNT section over the folds ``[(val, test), ...]`` (one run = one fold): the long book in a capital
    account (fixed slots of ``1 / slots``, ``slots`` None = the coins of TEST; fee on turnover; ``n_books`` None = read
    from the table) next to the same book paying the battery's round trip (E3) and all its capital on the picks
    (a3), the fee grid (a5), always long, ``benchmark`` buy-and-hold, the random-coin timing placebo (mean, 95th
    percentile, share of draws the book beats) and the basket held with the book's breadth as exposure (a6).
    SKIPPED with the reason when the table cannot make an account (no end_time / coin, no horizon)."""
    title = "ACCOUNT (TEST) - the long book as capital: fixed slots, cash, fee on turnover, compounded"
    try:
        frame = account_frame(folds, quantile)
        if not {"end_time", "coin"} <= set(frame.columns):
            raise ValueError("the table has no end_time / coin")
        step = row_step(frame["end_time"])
        books = int(n_books or horizon_steps(frame, step))
    except ValueError as e:
        return section(f"{title} - SKIPPED", [f"  ({e})"], "the account needs end_time, coin and the horizon") | {
            "stats": {}}
    r, present, picks, slots, named = _books(frame, slots, benchmark)
    rows, stats = [], {}

    def run(key, label, w, rr=r, c=cost, fee="turnover"):
        stats[key] = account_stats(account_curve(w, rr, c, books, step, fee), step, _expo(w))
        rows.append(_line(label, stats[key]))

    model_w = book_weights(picks, slots)
    run("model", f"MODEL long book, {slots} slots, fee on turnover", model_w)
    run("model round trip", "MODEL long book, a round trip every period (battery)", model_w, fee="round_trip")
    run("model equal", "MODEL long book, all capital on the picks (equal share)", book_weights(picks, slots, "equal"))
    for c in fee_grid:
        run(f"model fee {c:g}", f"MODEL long book, fee {c * 100:.2f}% per side", model_w, c=c)
    run("always long", f"always long, {slots} slots", book_weights(named["always long"], slots))
    if "buy and hold" in named:
        run("buy and hold", f"{benchmark} buy and hold", book_weights(named["buy and hold"], 1))
    breadth = (picks.sum(axis=1) / present.sum(axis=1)).clip(0.0, 1.0).to_frame("basket")
    run("breadth timing", "basket (equal weight) x the book's breadth (timing only)", breadth,
        rr=r.mean(axis=1).to_frame("basket"))
    rng = np.random.default_rng(seed)
    draws = [account_stats(account_curve(book_weights(placebo_weights(picks, present, rng), slots), r, cost, books,
                                         step), step) for _ in range(n_placebo)]
    if draws:
        cagr = np.array([d["cagr"] for d in draws])
        sharpe = np.array([d["sharpe"] for d in draws])
        stats["placebo"] = {"n": len(draws), "cagr_mean": float(cagr.mean()), "cagr_p95": float(np.quantile(cagr, .95)),
                            "sharpe_mean": float(sharpe.mean()), "sharpe_p95": float(np.quantile(sharpe, .95)),
                            "model_beats": float((stats["model"]["cagr"] > cagr).mean())}
        p = stats["placebo"]
        rows.append(f"  {'placebo: random coins, same timestamps and counts':<50} {p['n']} draws  CAGR mean "
                    f"{p['cagr_mean'] * 100:+.1f}% p95 {p['cagr_p95'] * 100:+.1f}%  Sharpe mean {p['sharpe_mean']:.2f}"
                    f" p95 {p['sharpe_p95']:.2f}  the book beats {p['model_beats'] * 100:.0f}% of draws")
    m, bh = stats["model"], stats.get("buy and hold")
    stats["setup"] = {"n_books": books, "step_hours": step / pd.Timedelta(hours=1), "slots": slots,
                      "quantile": quantile, "folds": len(folds), "rows": int(len(frame))}
    beats = {"buy and hold": bool(bh and m["cagr"] > bh["cagr"] and m["sharpe"] > bh["sharpe"]),
             "placebo p95": bool(draws and m["cagr"] > stats["placebo"]["cagr_p95"]),
             "always long": bool(m["cagr"] > stats["always long"]["cagr"])}
    stats["beats"] = beats
    note = (f"{books} sub-books of a {step / pd.Timedelta(hours=1):g} h step ({books * step / pd.Timedelta(hours=1):g} "
            f"h horizon), the long book = mu >= VAL q{quantile * 100:g} of mu, {slots} slots, {cost:.4%} per side on "
            f"the turnover; {len(folds)} fold(s), {len(frame):,} rows (flat rows kept). The book beats: "
            + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in beats.items())
            + ". A book that does not beat the placebo times the market, it does not pick coins.")
    return section(title, rows, note) | {"stats": stats}
