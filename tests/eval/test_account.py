"""The capital account (eval/account.py): known answers, the horizon read from the table, and the audit scripts a3,
a4 and a6 (`docs/audit/2026-11-21-independent-review/`, ported here line for line) reproduced through the section."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval.account import (
    account_curve,
    account_frame,
    account_section,
    account_stats,
    book_weights,
    horizon_steps,
    placebo_weights,
    row_step,
)

SIDE = 0.0008
STEP = pd.Timedelta(hours=4)


def _panel(n_times: int = 900, coins: int = 8, horizon: int = 2, seed: int = 0, skill: float = 0.3):
    """A 4 h panel whose label is the close ``horizon`` rows later: y_ret = last_close(t + h) / last_close(t) - 1;
    mu leans toward y_ret by ``skill``. Coin 0 is BTCUSDT; the last coin is missing from every 7th timestamp."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2023-01-01") + STEP * np.arange(n_times + horizon)
    names = ["BTCUSDT"] + [f"C{i}" for i in range(1, coins)]
    rows = []
    for c in names:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(t))))
        y = close[horizon:] / close[:-horizon] - 1.0
        mu = skill * y + rng.normal(0.0, 0.02, n_times)
        rows.append(pd.DataFrame({"coin": c, "end_time": t[:n_times], "last_close": close[:n_times], "y_ret": y,
                                  "mu": mu, "sigma": 0.02, "p_dir": 0.5 + 0.0 * mu,
                                  "dir_true": np.sign(y).astype(int)}))
    d = pd.concat(rows, ignore_index=True)
    d = d[~((d.coin == names[-1]) & (np.arange(len(d)) % 7 == 0))]
    return d.reset_index(drop=True)


def _val_test(seed: int = 0):
    d = _panel(seed=seed)
    cut = d.end_time.sort_values().unique()[300]
    return d[d.end_time < cut].reset_index(drop=True), d[d.end_time >= cut].reset_index(drop=True)


# --- the audit scripts, as written there (a4.run / a6.run / a3.subbook), on the loader's frame ------------------
def _a4_run(d, sel, slots, side=SIDE):
    res = []
    for phase in (0, 4):
        x = d[(d.end_time.dt.hour % 8 == phase)]
        prev, eq, curve = {}, 1.0, []
        for t, g in x.groupby("end_time", sort=True):
            s = g[sel.loc[g.index]]
            w = {c: 1.0 / slots for c in s.coin}
            if sum(w.values()) > 1:
                w = {c: 1.0 / len(s) for c in s.coin}
            turn = sum(abs(w.get(c, 0) - prev.get(c, 0)) for c in set(w) | set(prev))
            r = float(sum(w[c] * y for c, y in zip(s.coin, s.y_ret, strict=True))) - side * turn
            eq *= 1 + r
            curve.append((t, eq))
            prev = w
        res.append(pd.Series(dict(curve)))
    eq = pd.concat(res, axis=1, sort=True).ffill().fillna(1.0).mean(axis=1)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(6 * 365)
    return eq.iloc[-1], eq.iloc[-1] ** (1 / yrs) - 1, sharpe, (eq / eq.cummax() - 1).min()


def _a6_run(ts, expo, ret):
    curves = []
    for phase in (0, 4):
        idx = ts[ts.hour % 8 == phase]
        e, r = expo.loc[idx].clip(0, 1), ret.loc[idx]
        turn = e.diff().abs().fillna(e.iloc[0])
        curves.append((1 + e * r - SIDE * turn).cumprod())
    eq = pd.concat(curves, axis=1, sort=True).ffill().fillna(1).mean(axis=1)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rr = eq.pct_change().dropna()
    return eq.iloc[-1] ** (1 / yrs) - 1, rr.mean() / rr.std() * np.sqrt(6 * 365), (eq / eq.cummax() - 1).min()


def _a3_final(d, sel_col, fee_mode):
    out = {}
    for phase in (0, 4):
        x = d[d.end_time.dt.hour % 8 == phase]
        prev, rets = {}, []
        for t, g in x.groupby("end_time", sort=True):
            s = g[g[sel_col]]
            w = {c: 1.0 / len(s) for c in s.coin} if len(s) else {}
            if fee_mode == "turnover":
                turn = sum(abs(w.get(c, 0) - prev.get(c, 0)) for c in set(w) | set(prev))
            else:
                turn = 2.0 * sum(w.values())
            r = float((s.y_ret * (1.0 / len(s))).sum()) if len(s) else 0.0
            rets.append((t, r - SIDE * turn))
            prev = w
        out[phase] = pd.Series(dict(rets))
    return 0.5 * (1 + out[0]).cumprod().iloc[-1] + 0.5 * (1 + out[4]).cumprod().iloc[-1]


@pytest.fixture(scope="module")
def setup():
    val, test = _val_test()
    sec = account_section([(val, test)], SIDE, n_placebo=3)
    frame = account_frame([(val, test)])
    return val, test, sec, frame


def test_the_horizon_and_the_step_are_read_from_the_table():
    d = _panel(horizon=3)
    assert row_step(d.end_time) == STEP
    assert horizon_steps(d) == 3
    with pytest.raises(ValueError, match="no last_close"):
        horizon_steps(d.drop(columns="last_close"))
    d["y_ret"] = np.random.default_rng(1).normal(0, 0.02, len(d))     # a label no lag explains
    with pytest.raises(ValueError, match="explains y_ret"):
        horizon_steps(d)


def test_the_book_is_mu_at_or_above_the_val_quantile_and_flat_rows_are_kept():
    val, test = _val_test()
    test.loc[:9, "y_ret"] = 0.0
    test.loc[:9, "dir_true"] = 0
    f = account_frame([(val, test)])
    assert len(f) == len(test)                                       # E5: the flat rows stay
    assert (f["long"] == (f["mu"] >= np.quantile(val["mu"], 0.9))).all()


def test_a4_fixed_slots_reproduces_the_audit(setup):
    _, _, sec, frame = setup
    s = sec["stats"]
    slots = frame.coin.nunique()
    for key, sel, n_slots in (("model", frame["long"], slots),
                              ("always long", pd.Series(True, index=frame.index), slots),
                              ("buy and hold", frame.coin == "BTCUSDT", 1)):
        final, cagr, sharpe, dd = _a4_run(frame, sel, n_slots)
        assert s[key]["final"] == pytest.approx(final, rel=1e-9), key
        assert s[key]["cagr"] == pytest.approx(cagr, rel=1e-9)
        assert s[key]["sharpe"] == pytest.approx(sharpe, rel=1e-9)
        assert s[key]["max_dd"] == pytest.approx(dd, rel=1e-9)
    assert s["setup"]["n_books"] == 2 and s["setup"]["slots"] == slots


def test_a5_fee_grid_reproduces_the_audit(setup):
    _, _, sec, frame = setup
    for c in (0.0002, 0.0015):
        assert sec["stats"][f"model fee {c:g}"]["cagr"] == pytest.approx(
            _a4_run(frame, frame["long"], frame.coin.nunique(), side=c)[1], rel=1e-9)


def test_a6_breadth_timing_reproduces_the_audit(setup):
    _, _, sec, frame = setup
    g = frame.groupby("end_time")
    cagr, sharpe, dd = _a6_run(g["long"].mean().index, g["long"].mean(), g["y_ret"].mean())
    s = sec["stats"]["breadth timing"]
    assert (s["cagr"], s["sharpe"], s["max_dd"]) == pytest.approx((cagr, sharpe, dd), rel=1e-9)


def test_a3_equal_share_and_the_round_trip_fee_reproduce_the_audit(setup):
    _, _, sec, frame = setup
    assert sec["stats"]["model equal"]["final"] == pytest.approx(_a3_final(frame, "long", "turnover"), rel=1e-9)
    slots_rt = account_stats(account_curve(book_weights(_wide(frame, "long"), 1, "equal"), _wide(frame, "y_ret"),
                                           SIDE, 2, STEP, fee="round_trip"), STEP)
    assert slots_rt["final"] == pytest.approx(_a3_final(frame, "long", "round_trip"), rel=1e-9)
    # E3: a book that keeps coins pays less on turnover than on a round trip every period
    assert sec["stats"]["model"]["final"] > sec["stats"]["model round trip"]["final"]


def _wide(frame, col):
    w = frame.pivot(index="end_time", columns="coin", values=col).sort_index()
    return w.fillna(False).astype(bool) if col == "long" else w


def test_known_answers_cash_stays_flat_and_a_held_coin_pays_one_entry_per_sub_book():
    t = pd.Timestamp("2024-01-01") + STEP * np.arange(12)
    r = pd.DataFrame({"A": 0.01}, index=t)
    cash = account_curve(pd.DataFrame({"A": 0.0}, index=t), r, SIDE, 2, STEP)
    assert (cash == 1.0).all()
    held = account_curve(pd.DataFrame({"A": 1.0}, index=t), r, SIDE, 2, STEP)
    assert held.iloc[-1] == pytest.approx((1.01 - SIDE) * 1.01 ** 5)          # 6 periods per sub-book, 1 entry
    rt = account_curve(pd.DataFrame({"A": 1.0}, index=t), r, SIDE, 2, STEP, fee="round_trip")
    assert rt.iloc[-1] == pytest.approx((1.01 - 2 * SIDE) ** 6)


def test_the_placebo_keeps_the_count_per_timestamp_and_picks_only_present_coins():
    frame = account_frame([_val_test()])
    picks, present = _wide(frame, "long"), _wide(frame, "y_ret").notna()
    p = placebo_weights(picks, present, np.random.default_rng(0))
    assert (p.sum(axis=1) == picks.sum(axis=1)).all()
    assert not (p & ~present).to_numpy().any()


def test_the_section_reads_and_skips(setup):
    val, test, sec, _ = setup
    assert sec["title"].startswith("ACCOUNT (TEST)")
    assert "placebo" in sec["stats"] and 0.0 <= sec["stats"]["placebo"]["model_beats"] <= 1.0
    assert set(sec["stats"]["beats"]) == {"buy and hold", "placebo p95", "always long"}
    skipped = account_section([(val, test.drop(columns="last_close"))], SIDE, n_placebo=0)
    assert "SKIPPED" in skipped["title"] and skipped["stats"] == {}
    given = account_section([(val, test.drop(columns="last_close"))], SIDE, n_books=2, n_placebo=0)
    assert given["stats"]["model"]["final"] == pytest.approx(sec["stats"]["model"]["final"])


def test_account_from_folders_joins_the_folds_in_one_account_and_the_cli_prints_it(tmp_path, capsys):
    from nig_timenet_torch.workflow.battery_run import account_from_folders, main, table_path

    d = _panel(n_times=1200)
    times = d.end_time.sort_values().unique()
    folders, folds = [], []
    for k, (a, b, c) in enumerate(((0, 200, 600), (200, 600, 1200))):        # VAL [a, b), TEST [b, c)
        val = d[(d.end_time >= times[a]) & (d.end_time < times[b])].reset_index(drop=True)
        test = d[(d.end_time >= times[b]) & (d.end_time < times[min(c, len(times) - 1)])].reset_index(drop=True)
        f = tmp_path / f"fold_{k:03d}"
        f.mkdir()
        val.to_parquet(table_path(f, "val"), index=False)
        test.to_parquet(table_path(f, "test"), index=False)
        folders.append(f)
        folds.append((val, test))
    joined = account_from_folders(folders, n_placebo=2)
    assert joined["stats"]["setup"]["folds"] == 2
    assert joined["stats"]["model"]["final"] == pytest.approx(account_section(folds, n_placebo=2)["stats"]["model"]
                                                              ["final"])
    assert main([str(f) for f in folders] + ["--placebo", "0", "--books", "2"]) == 0
    assert "ACCOUNT (TEST)" in capsys.readouterr().out
