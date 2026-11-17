"""eval/legacy_reports.py, eval/legacy_plots.py, workflow/legacy_report.py: the TF reports ported over a battery
table. Known answers: a hand-checked tearsheet / equity, a model that copies the PREVIOUS return peaks at lag -1 in the
lag scan, a perfectly calibrated p_dir has ECE ~ 0, the worst trades are the ones with the largest losses."""

import math

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import legacy_reports as lr
from nig_timenet_torch.eval import sample_charts
from nig_timenet_torch.workflow import legacy_report
from nig_timenet_torch.workflow.battery_run import table_path


def _table(n_coins=6, n_bars=400, signal=0.3, copy_last=False, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2026-01-01")
    for c in range(n_coins):
        y = 0.02 * rng.standard_normal(n_bars)
        mu = np.r_[0.0, y[:-1]] if copy_last else signal * y + 0.02 * rng.standard_normal(n_bars) * math.sqrt(
            1 - signal ** 2)
        p = 1.0 / (1.0 + np.exp(-mu / 0.01))
        rows.append(pd.DataFrame({"step": 1, "mu": mu, "sigma": 0.02 + 0.01 * rng.random(n_bars), "p_dir": p,
                                  "y_ret": y, "dir_true": np.sign(y).astype(int), "last_ret": np.r_[0.0, y[:-1]],
                                  "coin": f"C{c}USDT",
                                  "end_time": t0 + pd.to_timedelta(4 * np.arange(n_bars), unit="h")}))
    return pd.concat(rows, ignore_index=True)


def test_tearsheet_and_equity_on_a_hand_checked_series():
    r = np.array([0.10, -0.05, 0.10, -0.05])
    ts = lr.tearsheet(r, risk=1.0)
    assert ts["n"] == 4 and ts["win_rate"] == 0.5 and ts["mean"] == pytest.approx(0.025)
    assert ts["profit_factor"] == pytest.approx(2.0) and ts["omega"] == ts["profit_factor"]
    assert ts["payoff"] == pytest.approx(2.0) and ts["max_win_streak"] == 1 and ts["max_loss_streak"] == 1
    assert ts["final_equity"] == pytest.approx(1.1 * 0.95 * 1.1 * 0.95)
    assert ts["max_drawdown"] == pytest.approx(-0.05)
    eq = lr.equity_curve(np.array([-2.0]), risk=0.5)          # a loss below -100 % is clipped at -100 %
    assert eq["equity"][-1] == pytest.approx(0.5)
    assert lr.tearsheet(np.array([0.1]))["n"] == 1


def test_trades_take_the_top_mu_rows_with_the_p_dir_side_and_pay_the_round_trip():
    t = _table()
    tr = lr.trades(t, cost=0.001, top_share=0.1)
    assert len(tr) == int(len(t) * 0.1) and tr["end_time"].is_monotonic_increasing
    assert np.abs(tr["mu"]).min() >= np.quantile(np.abs(t["mu"]), 0.89)
    np.testing.assert_allclose(tr["net"], tr["side"] * tr["y_ret"] - 0.002)
    daily = lr.daily_portfolio(tr)
    assert daily["trades"].sum() == len(tr) and daily["drawdown"].max() <= 0
    with pytest.raises(ValueError):
        lr.trades(t, 0.001, top_share=0.0)


def test_lag_scan_catches_late_copying():
    integ = lr.integrity(_table(copy_last=True))
    assert integ["lag_of_peak"] == -1                     # mu(t) = y(t-1): the peak sits at k = -1
    assert integ["mae_persistence"] == pytest.approx(integ["mae_model"])
    skill = lr.integrity(_table(signal=0.5))
    assert skill["lag_of_peak"] == 0 and skill["mae_model"] < skill["mae_zero"] * 1.5


def test_calibration_of_a_calibrated_probability_and_of_a_coin_flip():
    rng = np.random.default_rng(1)
    t = _table()
    p = rng.random(len(t))
    t["p_dir"], t["y_ret"] = p, np.where(rng.random(len(t)) < p, 0.01, -0.01)
    cal = lr.calibration(t)
    assert cal["ece"] < 0.05 and cal["brier"] < cal["brier_base_rate"]
    t["p_dir"] = p ** 3                                    # the same ranking, badly calibrated
    assert lr.calibration(t)["ece"] > 3 * cal["ece"]
    t["p_dir"] = 0.5
    assert lr.calibration(t)["brier"] == pytest.approx(0.25)


def test_conditioned_tables_and_coin_ranking():
    t = _table(signal=0.4)
    cond = lr.conditioned(t)
    assert set(cond) == {"by_move", "by_side", "by_sd", "by_batch"}
    assert cond["by_batch"]["rows"].sum() == int((t["y_ret"] != 0).sum()) and len(cond["by_batch"]) == 10
    assert (cond["by_move"]["ci_lo"] <= cond["by_move"]["hit"]).all()
    rank = lr.coin_ranking(t)
    assert len(rank) == 6 and rank["r"].is_monotonic_decreasing


def test_extreme_samples_are_the_largest_losses_and_gains():
    tr = lr.trades(_table(), 0.0008)
    ext = lr.extreme_samples(tr, 5)
    assert list(ext["worst"]["net"]) == sorted(tr["net"])[:5]
    assert list(ext["best"]["net"]) == sorted(tr["net"])[::-1][:5]


def test_workflow_writes_report_figures_and_a_samples_csv_sample_charts_reads(tmp_path):
    pytest.importorskip("matplotlib")
    for stage, seed in (("val", 1), ("test", 2)):
        _table(seed=seed).to_parquet(table_path(tmp_path, stage), index=False)
    reps = legacy_report.legacy_from_folder(tmp_path)
    assert set(reps) == {"val", "test"}
    md = (tmp_path / legacy_report.REPORT_MD).read_text()
    assert "## VAL" in md and "## TEST" in md and "Lag scan" in md and "Worst trades" in md
    assert (tmp_path / legacy_report.REPORT_JSON).is_file()
    figs = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert "test_equity.png" in figs and "val_best_worst_coins.png" in figs and len(figs) == 12
    samples = sample_charts.read_samples(tmp_path / legacy_report.SAMPLES_CSV)
    assert len(samples) == 40 and all(s.time for s in samples) and {s.side for s in samples} <= {"long", "short"}
    with pytest.raises(FileNotFoundError):
        legacy_report.legacy_from_folder(tmp_path / "empty")
