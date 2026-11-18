"""eval/legacy_plots.py::pred_vs_true / top_coins: per-coin predicted-vs-realised PNGs from a battery table."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import legacy_plots


def _table(with_hl=True):
    rng = np.random.default_rng(0)
    rows = []
    for c, n in (("AAA", 60), ("BBB", 40), ("CCC", 20)):
        y = 0.02 * rng.standard_normal(n)
        t = pd.Timestamp("2026-01-01") + pd.to_timedelta(4 * np.arange(n), unit="h")
        d = pd.DataFrame({"coin": c, "end_time": t,
                          "mu": 0.3 * y + 0.01 * rng.standard_normal(n), "sigma": 0.02, "y_ret": y})
        if with_hl:
            d["pred_high"], d["fut_high"] = d["mu"] + 0.01, y + 0.01
            d["pred_low"], d["fut_low"] = d["mu"] - 0.01, y - 0.01
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_top_coins_orders_by_row_count():
    assert legacy_plots.top_coins(_table(), 2) == ["AAA", "BBB"]


@pytest.mark.parametrize("with_hl", [True, False])
def test_pred_vs_true_writes_one_png_per_coin(tmp_path, with_hl):
    paths = legacy_plots.pred_vs_true(_table(with_hl), ["AAA", "CCC"], tmp_path / "figs", "TEST")
    assert [p.name for p in paths] == ["TEST_pred_vs_true_AAA.png", "TEST_pred_vs_true_CCC.png"]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)


def test_a_missing_coin_and_a_missing_column_raise_clearly(tmp_path):
    with pytest.raises(ValueError, match="ZZZ.*not in the table"):
        legacy_plots.pred_vs_true(_table(), ["AAA", "ZZZ"], tmp_path, "VAL")
    with pytest.raises(ValueError, match="lacks columns"):
        legacy_plots.pred_vs_true(_table().drop(columns="sigma"), ["AAA"], tmp_path, "VAL")
    with pytest.raises(ValueError, match="no coin"):
        legacy_plots.pred_vs_true(_table(), [], tmp_path, "VAL")
