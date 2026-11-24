"""The chicks report's pieces on the known-answer world: decoding, calibration on VAL only, the tearsheet's own
self-tests (ported from the TF module), trade selection, output manager, the English text. The line-by-line parity with
the old project's output is ``tests/parity/test_chicks_parity.py``."""

import contextlib
import dataclasses
import io

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, ReturnCodec
from nig_timenet_torch.eval import chicks_trust as ct
from nig_timenet_torch.eval.chicks_assets import test_all_assets as judge_assets
from nig_timenet_torch.eval.chicks_decode import StageData, decode_stage, evaluate_decoded, verify_decoding
from nig_timenet_torch.eval.chicks_flat import build_flat_frame
from nig_timenet_torch.eval.chicks_full import run_full_analysis
from nig_timenet_torch.eval.chicks_outputs import save_or_print
from nig_timenet_torch.eval.chicks_tearsheet import (
    compute_trading_performance_metrics,
    signal_significance_report,
    simulate_equity_curve,
)
from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.eval.chicks_trades import select_and_rank_trades
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.workflow.chicks_run import stage_from_dataset
from tests.eval.verification_world import SCALE, TARGETS
from tests.parity.chicks_world import small_world


@pytest.fixture(scope="module")
def world():
    ds, idx, tables = small_world("1D", 160, 4)
    return {"ds": ds, "test": stage_from_dataset(ds, idx["test"], tables["test"], TARGETS),
            "val": stage_from_dataset(ds, idx["val"], tables["val"], TARGETS)}


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def test_decoding_is_the_registered_codec_and_the_price_slope_is_exact_for_the_return_codec(world):
    st = world["test"]
    dec = decode_stage(st)
    codec, state = TARGET_CODECS["return"](), st.codec_state
    for t in TARGETS:
        d, mu = dec[t], st.table[head_spec("nig_regression", t).output_keys["mu"]].to_numpy()
        np.testing.assert_allclose(d["pred_real"], codec.decode(mu, state, st.anchors[t]), rtol=1e-15)
        np.testing.assert_array_equal(d["entry_price"], st.anchors[t])
        np.testing.assert_allclose(d["iqr"], st.anchors[t] / SCALE, rtol=1e-8)          # d price / d output
        np.testing.assert_allclose(d["uncertainty_real"], d["uncertainty_scaled"] * np.abs(d["iqr"]), rtol=1e-15)
        assert np.isnan(d["p_up"]).all() and not d["unc_clipped"].any()


def test_the_class_head_is_read_as_p_up_only_when_asked(world):
    key = head_spec("binary_classification", "close").output_keys["logits"]
    on = decode_stage(world["test"], class_head=True)["close"]["p_up"]
    np.testing.assert_array_equal(on, world["test"].table[key].to_numpy())


def test_a_target_mode_that_is_not_a_price_forecast_is_refused(world, monkeypatch):
    monkeypatch.setattr(ReturnCodec, "forecasts_price", lambda self, state: False)
    with pytest.raises(ValueError, match="price forecast"):
        decode_stage(world["test"])


def test_stage_data_checks_that_everything_is_aligned(world):
    st = world["test"]
    with pytest.raises(ValueError, match="last_candles"):
        dataclasses.replace(st, last_candles=st.last_candles[:-1])
    with pytest.raises(ValueError, match="anchor"):
        dataclasses.replace(st, anchors={**st.anchors, "close": st.anchors["close"][:-1]})
    with pytest.raises(ValueError, match="label"):
        dataclasses.replace(st, labels={})
    assert isinstance(st, StageData)


def test_the_verification_passes_on_a_sound_decode_and_fails_on_a_wrong_one(world):
    dec = evaluate_decoded(decode_stage(world["test"]))
    assert {r["status"] for r in verify_decoding(dec).values()} == {"ok"}
    dec["close"]["target_real"] = dec["close"]["target_real"] * 1.05          # a wrong "true price": inconsistent
    assert verify_decoding(dec)["close"]["status"] == "FAILED"


def test_calibrators_are_fitted_on_val_only_and_carry_plain_state(world):
    res = quiet(judge_assets, world["val"], out_dir=".", verbose=False, run_verification=False)
    val_flat = build_flat_frame(res["per_asset_results"], TARGETS)
    cals = quiet(ct.fit_confidence_calibrators, val_flat, verbose=False)
    assert set(cals) == set(TARGETS)
    for t, c in cals.items():
        assert c["n_val"] == int((val_flat["target"] == t).sum())
        if c["signal"] == "base_rate":                      # nothing informative on val: the constant, no learned state
            assert c["state"] is None
        else:
            assert c["state"]["fit_role"] == Role.CALIBRATION.value and c["state"]["fit_rows"] == c["n_val"]
    assert any(c["state"] is not None for c in cals.values())
    with pytest.raises(ValueError, match="no-such-method"):
        ct.fit_confidence_calibrators(val_flat, method="no-such-method")


def test_diagnose_confidence_p_value_matches_scipy_spearmanr():
    rng = np.random.default_rng(0)
    conf = rng.random(300)
    correct = (rng.random(300) < 0.3 + 0.4 * conf).astype(float)
    d = ct.diagnose_confidence(conf, correct)
    rho, p = sps.spearmanr(conf, correct)
    assert d["spearman_rho"] == pytest.approx(rho, abs=1e-12) and d["spearman_p"] == pytest.approx(p, rel=1e-8)
    assert d["verdict"] == "informative"
    assert ct.diagnose_confidence(np.full(300, 0.5), correct)["verdict"] == "constant"


def test_trade_selection_filter_is_strict_and_the_combined_rank_is_inside_each_head():
    rng = np.random.default_rng(0)
    rows = []
    for target, level in (("high", 0.85), ("close", 0.30), ("low", 0.25)):          # the TF fairness self-test
        for m, u, c in zip(rng.normal(0, 1, 200), np.abs(rng.normal(1, 0.2, 200)) + 0.1,
                           np.clip(level + rng.normal(0, 0.05, 200), 0.01, 0.99), strict=True):
            rows.append({"target": target, "predicted_change": m, "uncertainty": u, "confidence": c,
                         "kind": "continuous"})
    df = pd.DataFrame(rows)
    sel = select_and_rank_trades(df, min_edge_ratio=1.0, sort_by="both", top_n=30)
    assert len(sel["target"].value_counts()) == 3 and sel["target"].value_counts().max() <= 15
    assert (sel["edge_ratio"] > 1.0).all()
    edge = pd.DataFrame({"target": ["close"] * 2, "kind": ["continuous"] * 2, "predicted_change": [1.0, 2.0],
                         "uncertainty": [1.0, 1.0], "confidence": [0.5, 0.5]})
    assert list(select_and_rank_trades(edge)["predicted_change"]) == [2.0]            # edge exactly 1 is dropped
    with pytest.raises(ValueError, match="sort_by"):
        select_and_rank_trades(df, sort_by="nope")


def test_the_tearsheet_never_goes_below_minus_100_percent_and_matches_the_plotted_curve():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 5, 200)
    r[50] = -150.0
    m = compute_trading_performance_metrics(r)
    eq = simulate_equity_curve(r)
    assert m["max_drawdown_pct"] >= -100.0 and eq["drawdown_pct"].min() >= -100.0 and np.all(eq["equity"] > 0)
    rng = np.random.default_rng(2)
    r = rng.normal(1.5, 3, 500)
    r[250] = -150.0
    m = compute_trading_performance_metrics(r, risk_per_trade=0.1)
    direct = simulate_equity_curve(r, initial_capital=100.0, risk_per_trade=0.1)
    assert m["total_return_pct"] > -50.0
    assert m["max_drawdown_pct"] == pytest.approx(direct["drawdown_pct"].min(), abs=1e-9)
    assert m["total_return_pct"] == pytest.approx(direct["equity"][-1] - 100.0, abs=1e-9)
    assert compute_trading_performance_metrics(np.array([])) == {}


def test_the_significance_report_finds_a_real_signal_not_a_random_one_and_the_repeated_reversal():
    rng = np.random.default_rng(3)
    T, A = 200, 40
    price, rows = np.full(A, 100.0), []
    for t in range(T):
        prev, move = price.copy(), rng.normal(0, 0.02, A)
        price = price * (1 + move)
        rows += [{"asset": f"A{a}", "timestamp": float(t), "entry": prev[a], "true": price[a], "truth_move": move[a]}
                 for a in range(A)]
    df = pd.DataFrame(rows)
    df["predicted_change"] = np.where(rng.random(len(df)) < 0.60, np.sign(df["truth_move"]), -np.sign(df["truth_move"]))
    good = signal_significance_report(df, trade_cost_pct=0.0, n_perm=200, verbose=False)
    assert good["model"]["t_stat"] > 2 and good["permutation"]["p_value"] < 0.05
    df["predicted_change"] = rng.choice([-1.0, 1.0], len(df))
    rand = signal_significance_report(df, trade_cost_pct=0.0, n_perm=200, verbose=False)
    assert rand["permutation"]["p_value"] > 0.05
    df = df.sort_values(["asset", "timestamp"])
    df["predicted_change"] = -np.sign(df.groupby("asset")["entry"].diff().fillna(1.0))
    rev = signal_significance_report(df, trade_cost_pct=0.0, n_perm=50, verbose=False)
    assert rev["agreement_with_reversal_pct"] > 99


def test_save_or_print_saves_a_big_table_and_prints_a_small_one(tmp_path, capsys):
    small, big = pd.DataFrame({"a": range(3)}), pd.DataFrame({"a": range(41)})
    assert save_or_print(small, "small", out_dir=str(tmp_path)) is None and "a" in capsys.readouterr().out
    path = save_or_print(big, "big", out_dir=str(tmp_path))
    assert path == str(tmp_path / "big.csv") and len(pd.read_csv(path, encoding="utf-8-sig")) == 41
    assert save_or_print("x" * 4001, "text", out_dir=str(tmp_path)).endswith("text.txt")
    with pytest.raises(TypeError):
        save_or_print([1], "bad", out_dir=str(tmp_path))


def test_the_english_report_has_the_same_numbers_and_files_as_the_arabic_one(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kw = {"make_plots": False, "run_integrity_check": False}
    ar, en = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(ar):
        res_ar = run_full_analysis(world["test"], out_dir="ar", val_stage=world["val"], say=Say("ar"), **kw)
    with contextlib.redirect_stdout(en):
        res_en = run_full_analysis(world["test"], out_dir="en", val_stage=world["val"], say=Say("en"), **kw)
    assert res_ar["trading_metrics"]["sharpe_ratio"] == res_en["trading_metrics"]["sharpe_ratio"]
    pd.testing.assert_frame_equal(res_ar["flat_df"], res_en["flat_df"])
    assert "Trust & Calibration Report" in en.getvalue() and "تقرير الثقة" in ar.getvalue()
    assert "تقرير الثقة" not in en.getvalue()
