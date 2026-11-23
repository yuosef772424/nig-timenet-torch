"""eval/latest_table.py, eval/trade_selection.py, eval/latest_plots.py: the latest samples in price units, the
TF trade selection over them and their chart, on a battery table with a known answer."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import latest_plots, latest_table, trade_selection
from nig_timenet_torch.eval.latest_table import LATEST_COLUMNS

HOURS4 = pd.Timedelta("4h")


def _battery(live_last: bool = True) -> pd.DataFrame:
    """Two coins x 5 rows; mu / y_ret / p_dir chosen by hand. The last row of each coin is live (y_ret NaN)."""
    t = pd.date_range("2026-03-01", periods=5, freq="4h")
    rows = []
    for coin, entry, mu, y, p in (
            ("AAA", 100.0, [0.02, -0.01, 0.005, 0.03, 0.01], [0.01, 0.02, 0.004, -0.02, np.nan],
             [.8, .3, .6, .9, .7]),
            ("BBB", 50.0, [-0.02, 0.0, 0.01, -0.004, 0.02], [-0.03, 0.001, -0.01, -0.002, np.nan],
             [.1, .6, .55, .4, .9])):
        rows.append(pd.DataFrame({"coin": coin, "end_time": t, "mu": mu, "sigma": 0.01, "p_dir": p, "y_ret": y,
                                  "last_close": entry, "pred_high": 0.03, "fut_high": 0.02}))
    d = pd.concat(rows, ignore_index=True)
    if not live_last:
        d["y_ret"] = d["y_ret"].fillna(0.0)
    return d.sample(frac=1.0, random_state=0)          # unsorted on purpose


def test_columns_and_price_units():
    out = latest_table.latest_table(_battery(), n=3)
    assert list(out.columns[:len(LATEST_COLUMNS)]) == list(LATEST_COLUMNS)
    assert list(out["coin"]) == ["AAA"] * 3 + ["BBB"] * 3                       # coin order, 3 rows each
    a = out[out["coin"] == "AAA"].iloc[0]                                       # AAA row 2: mu 0.005, y 0.004
    assert a["entry"] == 100.0 and a["target"] == pytest.approx(100.5) and a["true_target"] == pytest.approx(100.4)
    assert a["move_pct"] == pytest.approx(0.5) and a["true_move_pct"] == pytest.approx(0.4)
    assert a["error_pct"] == pytest.approx(0.1) and a["uncertainty_pct"] == pytest.approx(1.0)
    assert a["target_lo"] == pytest.approx(99.5) and a["target_hi"] == pytest.approx(101.5)
    assert a["target_high"] == pytest.approx(103.0) and a["true_high"] == pytest.approx(102.0)
    assert a["edge_ratio"] == pytest.approx(0.5) and a["signal"] == "LONG" and a["confidence_pct"] == pytest.approx(60)


def test_a_row_without_a_label_is_live_never_zero():
    out = latest_table.latest_table(_battery(), n=2)
    live = out[out["status"] == "live"]
    assert len(live) == 2 and live["true_move_pct"].isna().all() and live["true_target"].isna().all()
    assert live["ok"].isna().all()                                              # None, not False
    assert live["target"].notna().all()                                         # the prediction is there
    closed = out[out["status"] == "closed"]
    assert closed["ok"].map(lambda v: isinstance(v, (bool, np.bool_))).all()


def test_signal_confidence_and_ok_follow_the_class_head():
    out = latest_table.latest_table(_battery(), n=5)
    b = out[out["coin"] == "BBB"].reset_index(drop=True)
    assert list(b["signal"][:2]) == ["SHORT", "LONG"]
    assert b["confidence_pct"][0] == pytest.approx(90.0)                        # P(short) = 1 - 0.1
    assert list(b["ok"][:4]) == [True, True, False, True]                       # side * y_ret > 0
    # BBB row 3: class head says LONG (p .55) and mu is +1 %: they agree; row 2: p .6 LONG, mu 0 -> sign 0 disagrees
    assert bool(b["agrees"][2]) and not bool(b["agrees"][1])


def test_filters_now_and_errors():
    t = _battery()
    cut = pd.Timestamp("2026-03-01 08:00")
    assert latest_table.latest_table(t, n=9, now=cut)["time"].max() == cut
    assert set(latest_table.latest_table(t, coins="BBB")["coin"]) == {"BBB"}
    with pytest.raises(ValueError, match="unknown coin"):
        latest_table.latest_table(t, coins=["ZZZ"])
    with pytest.raises(ValueError, match="lacks"):
        latest_table.latest_table(t.drop(columns="sigma"))
    with pytest.raises(ValueError, match=">= 1"):
        latest_table.latest_table(t, n=0)
    assert latest_table.latest_table(t, now=pd.Timestamp("2020-01-01")).empty


def test_print_shows_dashes_for_live_and_a_legend():
    lines = []
    text = latest_table.print_latest_table(latest_table.latest_table(_battery(), n=1), title="T", echo=lines.append)
    assert lines == [text] and text.startswith("T\n") and "live" in text and " - " in text and "confidence =" in text
    assert latest_table.print_latest_table(pd.DataFrame(), echo=lambda s: None) == "(no samples to show)"


def test_trade_selection_is_strict_edge_over_uncertainty_and_ranks():
    lt = latest_table.latest_table(_battery(), n=5)
    sel = trade_selection.select_trades(lt, k=1.0, rank_by="move")
    assert (sel["edge_ratio"] > 1.0).all()                                      # edge exactly 1 (|mu| = sd) is dropped
    assert list(sel["rank"]) == list(range(1, len(sel) + 1))
    assert sel["move_pct"].abs().is_monotonic_decreasing
    everything = trade_selection.select_trades(lt, k=0.0)                       # k = 0: every row with a non-zero move
    assert len(everything) == len(lt) - 1 and (everything["move_pct"] != 0).all()
    conf = trade_selection.select_trades(lt, k=0.0, rank_by="confidence", top=3)
    assert len(conf) == 3 and conf["confidence_pct"].is_monotonic_decreasing and conf["rank"].iloc[0] == 1
    comb = trade_selection.select_trades(lt, k=0.0, rank_by="combined")
    assert comb["score"].between(0, 1).all() and comb["score"].is_monotonic_decreasing


def test_trade_selection_cost_confidence_agreement_and_errors():
    lt = latest_table.latest_table(_battery(), n=5)
    everything = trade_selection.select_trades(lt, k=0.0)
    after = trade_selection.select_trades(lt, k=0.0, min_after_cost=1.0)        # |move| - 0.16 % must reach 1 %
    assert (after["move_pct"].abs() >= 1.16 - 1e-9).all() and len(after) == 4 < len(everything)
    assert (after["after_cost_pct"] >= 1.0).all()
    assert (trade_selection.select_trades(lt, k=0.0, min_confidence=85)["confidence_pct"] >= 85).all()
    assert trade_selection.select_trades(lt, k=0.0, require_agreement=True)["agrees"].all()
    assert trade_selection.select_trades(lt, k=99.0).empty
    with pytest.raises(ValueError, match="rank_by"):
        trade_selection.select_trades(lt, rank_by="luck")
    with pytest.raises(ValueError, match="lacks"):
        trade_selection.select_trades(lt.drop(columns="edge_ratio"))


def test_chart_one_png_per_coin_with_and_without_candles(tmp_path):
    lt = latest_table.latest_table(_battery(), n=5)
    idx = pd.date_range("2026-02-27", periods=40, freq="4h")
    px = 100 + np.arange(40) * 0.1
    candles = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px + 0.05}, index=idx)
    paths = latest_plots.latest_charts(lt, tmp_path, {"AAA": candles})
    assert [p.name.split("_")[1] for p in paths] == ["AAA", "BBB"] and all(p.stat().st_size > 5000 for p in paths)
    with pytest.raises(ValueError, match="no rows"):
        latest_plots.latest_chart(lt, "ZZZ", tmp_path)
    one = lt[lt["coin"] == "AAA"].tail(1)
    with pytest.raises(ValueError, match="horizon"):
        latest_plots.latest_chart(one, "AAA", tmp_path)
    assert latest_plots.latest_chart(one, "AAA", tmp_path, horizon=HOURS4).is_file()


def test_latest_report_writes_csv_markdown_and_charts_and_reads_a_run_folder(tmp_path):
    from nig_timenet_torch.workflow import latest_report
    from nig_timenet_torch.workflow.battery_run import table_path

    lines = []
    rep = latest_report.latest_report(_battery(), tmp_path / "o", n=3, k=0.5, top=2, echo=lines.append, title="VAL")
    assert len(rep["latest"]) == 6 and len(rep["selection"]) == 2 and len(rep["charts"]) == 2
    out = tmp_path / "o"
    assert pd.read_csv(out / latest_report.LATEST_CSV).shape[0] == 6
    assert (out / latest_report.SELECTION_CSV).is_file()
    assert "## Selected" in (out / latest_report.REPORT_MD).read_text() and len(lines) == 2
    folder = tmp_path / "run"
    folder.mkdir()
    _battery().to_parquet(table_path(folder, "val"))
    got = latest_report.latest_from_folder(folder, "val", charts=False, n=2, echo=lambda s: None)
    assert len(got["latest"]) == 4 and (folder / latest_report.REPORT_MD).is_file() and not got["charts"]
    with pytest.raises(FileNotFoundError, match="not found"):
        latest_report.latest_from_folder(folder, "test")
