"""The latest-samples table per target of the verification (eval/verification_latest.py): values worked by hand on a
two-row table, the printed text, and the wiring into run_full_verification."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.schema import LAST_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval import verification as ver
from nig_timenet_torch.eval.verification_latest import (
    format_latest_by_target,
    latest_by_target,
    print_latest_by_target,
)
from nig_timenet_torch.model.heads import head_spec
from tests.eval.verification_world import TARGETS, make_world


def tiny():
    """One coin, two rows; every number asserted below is worked by hand (scale 100, return codec)."""
    lc = np.zeros((2, len(LAST_COLUMNS)))
    for name, vals in {"last_high": (10.0, 20.0), "last_low": (8.0, 16.0), "last_close": (9.0, 18.0),
                       "timestamp": (1.7e18, 1.7e18 + 86_400e9), "future_close": (9.9, 18.0),
                       "future_low_min": (7.2, 15.0), "future_high_max": (11.0, 21.0)}.items():
        lc[:, LAST_COLUMNS.index(name)] = vals
    ds = {"last_candles": lc, "reg_target_mode": "return", "reg_target_scale": 100.0}
    nu, alpha = 1.0, 5.0
    beta = nu * (alpha - 1) / (1 + nu)                    # predictive std = 1 label unit = 1 % of the anchor
    cols = {"asset": ["C0", "C0"]}
    for t, mu, p in (("high", [-10.0, 5.0], [0.3, 0.9]), ("low", [10.0, 0.0], [0.6, 0.4]),
                     ("close", [20.0, -5.0], [0.7, 0.2])):
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        cols |= {reg.output_keys["mu"]: mu, reg.output_keys["nu"]: [nu] * 2, reg.output_keys["alpha"]: [alpha] * 2,
                 reg.output_keys["beta"]: [beta] * 2, reg.output_keys["confidence"]: [0.8, 0.5],
                 cls.output_keys["logits"]: p}
    return ds, pd.DataFrame(cols), TARGET_CODECS["return"]().fit(ds)


def test_values_are_hand_computed():
    ds, table, state = tiny()
    newest = latest_by_target(ds, [0, 1], table, state, TARGETS, n=1, n_coins=1)
    assert list(newest["target"]) == ["high", "low", "close"] and list(newest["row"]) == [1, 1, 1]
    out = latest_by_target(ds, [0, 1], table, state, TARGETS, n=2, n_coins=1).set_index(["row", "target"])
    r = out.loc[(0, "high")]          # entry 10, pred 10 * (1 - 0.1) = 9, true 11, std 1 % of 10 = 0.1 -> unc 1 %
    assert (r["entry"], r["pred"], r["true"]) == (10.0, 9.0, 11.0) and r["signal"] == "down"
    assert r["move_pct"] == pytest.approx(-10.0) and r["true_move_pct"] == pytest.approx(10.0)
    assert r["err_move_pct"] == pytest.approx(-20.0) and r["err_px"] == pytest.approx(-2.0)
    assert r["conf_pct"] == pytest.approx(80.0) and r["unc_pct"] == pytest.approx(1.0) and not r["ok"]
    r = out.loc[(0, "close")]         # entry 9, pred 10.8, true 9.9: up and right
    assert (r["entry"], r["true"]) == (9.0, 9.9) and r["pred"] == pytest.approx(10.8) and r["ok"]
    assert r["move_pct"] == pytest.approx(20.0) and r["true_move_pct"] == pytest.approx(10.0)
    assert r["err_move_pct"] == pytest.approx(10.0) and r["err_px"] == pytest.approx(0.9)
    r = out.loc[(0, "low")]           # entry 8, pred 8.8, true 7.2: an up signal, the price fell
    assert r["pred"] == pytest.approx(8.8) and r["true_move_pct"] == pytest.approx(-10.0) and not r["ok"]
    assert r["err_move_pct"] == pytest.approx(20.0) and r["err_px"] == pytest.approx(1.6)
    r = out.loc[(1, "close")]         # entry 18, pred 17.1, true 18 (flat): a down signal is not right on a flat truth
    assert r["pred"] == pytest.approx(17.1) and r["true_move_pct"] == 0.0 and not r["ok"]
    assert r["err_move_pct"] == pytest.approx(-5.0) and r["err_px"] == pytest.approx(-0.9)


def test_the_text_has_the_tf_columns_and_arrows(capsys):
    ds, table, state = tiny()
    out = latest_by_target(ds, [0, 1], table, state, TARGETS, n=1, n_coins=1)
    text = format_latest_by_target(out)
    assert list(text.columns) == ["#", "target", "signal", "conf", "unc", "move", "true_move", "err_move", "entry",
                                  "pred", "true", "err_px", "ok"]
    assert list(text["signal"]) == ["▲", "▼", "▼"]
    close = text[text["target"] == "close"].iloc[0]
    assert (close["move"], close["true_move"], close["err_move"]) == ("-5.000%", "+0.000%", "-5.000%")
    assert close["conf"] == "50.0%" and close["unc"] == "1.00%" and close["entry"] == "18" and close["ok"] == "❌"
    print_latest_by_target(out, 1)
    assert "latest 1 sample(s) per target" in capsys.readouterr().out


def test_a_missing_head_is_skipped_and_bad_arguments_raise():
    ds, table, state = tiny()
    low = head_spec("nig_regression", "low").output_keys["mu"]
    out = latest_by_target(ds, [0, 1], table.drop(columns=[low]), state, TARGETS, n=1, n_coins=1)
    assert set(out["target"]) == {"high", "close"}
    assert latest_by_target(ds, [0, 1], table[[]].assign(asset=["C0", "C0"]), state, TARGETS).empty
    with pytest.raises(ValueError, match="latest_by_target"):
        latest_by_target(ds, [0, 1], table, state, TARGETS, n=0)


def test_the_verification_prints_and_writes_the_latest_table(tmp_path, capsys):
    ds, idx, tables = make_world(kind="skilled")
    args = {"targets": TARGETS, "x_keys": ["X_1D"], "cost_per_side": 0.0008, "n_boot": 30, "min_n": 50,
            "run_candle": False}
    out = ver.run_full_verification(ds, idx, tables, **args, out_dir=tmp_path, verbose=True, latest_n=2,
                                    latest_coins=2)
    latest = out["details"]["latest"]
    assert len(latest) == 2 * 2 * 3 and set(latest["target"]) == set(TARGETS) and latest["coin"].nunique() == 2
    csv = pd.read_csv(tmp_path / ver.LATEST_CSV)
    assert list(csv.columns)[:3] == ["row", "coin", "target"] and len(csv) == len(latest)
    assert "latest 2 sample(s) per target" in capsys.readouterr().out
    assert "latest" not in ver.run_full_verification(ds, idx, tables, **args, latest_n=0)["details"]
