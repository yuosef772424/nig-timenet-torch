"""The integrated verification (eval/verification.py): every check has a case that passes and one that fails, on a small
synthetic world with a known answer (tests/eval/verification_world.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval import verification as ver
from nig_timenet_torch.eval import verification_trading as trading
from nig_timenet_torch.eval.verification_candle import BASELINES
from nig_timenet_torch.model.heads import head_spec
from tests.eval.verification_world import TARGETS, make_dataset, make_idx, make_table, make_world

PASS, FAIL, FLAT, WARN, INFO = ver.PASS, ver.FAIL, ver.FLAT, ver.WARN, ver.INFO
COST = 0.0008


def run(world, **kw):
    ds, idx, tables = world
    args = {"targets": TARGETS, "x_keys": ["X_1D"], "cost_per_side": COST, "n_boot": 100, "min_n": 50,
            "run_candle": False} | kw
    return ver.run_full_verification(ds, idx, tables, **args)


def row(out, section, check_start):
    s = out["summary"]
    hit = s[(s["section"] == section) & s["check"].str.startswith(check_start)]
    assert len(hit) == 1, (section, check_start, list(s["check"]))
    return hit.iloc[0]


@pytest.fixture(scope="module")
def skilled():
    return make_world(kind="skilled")


@pytest.fixture(scope="module")
def unskilled():
    return make_world(kind="unskilled")


def test_summary_has_the_old_columns_and_sections(skilled):
    out = run(skilled)
    s = out["summary"]
    assert list(s.columns) == ["section", "check", "value", "verdict", "note"]
    assert set(s["section"]) == {ver.A, ver.B, ver.C_CONF, ver.C_TRADE}
    assert set(out["details"]) == {"selective", "trades", "val_df", "test_df", "latest"}
    assert set(s["verdict"]) <= {PASS, FAIL, FLAT, WARN, INFO}


# ---------------------------------------------------------------------------------------------- a) data integrity
def test_a_inputs_finite_passes_and_a_nan_fails_only_its_stage(skilled):
    ds, idx, tables = skilled
    out = run(skilled)
    assert {row(out, ver.A, f"X_{s}")["verdict"] for s in ("train", "val", "test")} == {PASS}
    bad = {**ds, "X_1D": ds["X_1D"].copy()}
    bad["X_1D"][idx["test"][7], 3, 1] = np.nan
    bad["X_1D"][idx["val"][2], 0, 0] = np.inf
    out = run((bad, idx, tables))
    assert row(out, ver.A, "X_train")["verdict"] == PASS
    assert row(out, ver.A, "X_val")["verdict"] == FAIL and row(out, ver.A, "X_test")["verdict"] == FAIL


def test_a_labels_match_the_raw_prices_and_shuffled_labels_fail(skilled):
    ds, idx, tables = skilled
    out = run(skilled)
    for stage in ("val", "test"):
        r = row(out, ver.A, f"labels match the raw prices ({stage})")
        assert r["verdict"] == PASS and r["value"] == 1.0
    shuffled = dict(ds)
    rng = np.random.default_rng(0)
    for t in TARGETS:
        for kind in ("nig_regression", "binary_classification"):
            key = head_spec(kind, t).label_key
            shuffled[key] = rng.permutation(ds[key])
    out = run((shuffled, idx, tables))
    for stage in ("val", "test"):
        r = row(out, ver.A, f"labels match the raw prices ({stage})")
        assert r["verdict"] == FAIL and r["value"] < 0.99
        assert set(json.loads(r["note"])) == {f"{t}_{k}" for t in TARGETS for k in ("class", "reg")}


def test_a_one_wrong_head_is_enough_to_fail_the_label_check(skilled):
    ds, idx, tables = skilled
    key = head_spec("binary_classification", "low").label_key
    bad = {**ds, key: 1.0 - ds[key]}
    r = row(run((bad, idx, tables)), ver.A, "labels match the raw prices (test)")
    shares = json.loads(r["note"])
    assert r["verdict"] == FAIL and shares["low_class"] == 0.0 and shares["high_class"] == 1.0


def test_a_label_check_is_skipped_for_other_target_modes(skilled):
    out = run(skilled, target_mode="scaled")
    r = row(out, ver.A, "labels match the raw prices (test)")
    assert r["verdict"] == INFO and "scaled" in r["note"]
    # c) too: it assumes return targets
    assert row(out, ver.C, "skipped")["verdict"] == INFO
    assert run(skilled, target_mode="return")["summary"].eval("verdict == @INFO").sum() > 0


def test_a_label_agreement_follows_the_registered_codec_for_log_labels(skilled):
    ds, idx, _ = skilled
    state = TARGET_CODECS["return"]().fit(ds)
    assert state["scale"] == 100.0
    log = {**ds, "reg_target_mode": "return", "price_norm_mode": "logret"}    # TF logret artifacts: log labels
    agree = ver.label_agreement(log, idx["test"], TARGETS)
    assert agree["close_reg"] < 0.99            # simple-return labels are not log labels: the codec says so


def test_a_split_gap_enough_adjacent_and_overlapping(skilled):
    ds, idx, tables = skilled
    ok = run(skilled)
    for a, b in (("train", "val"), ("val", "test")):
        r = row(ok, ver.A, f"{a} before {b}")
        assert r["verdict"] == PASS and r["value"] == 13.0 and "9" in r["note"]      # window 8 + horizon 1
    near = make_world(val_start=205)            # val starts 6 days after the last train day: positive but < 9
    r = row(run(near), ver.A, "train before val")
    assert r["verdict"] == WARN and r["value"] == 6.0
    overlap = make_world(val_start=150)         # val rows lie inside the train period
    r = row(run(overlap), ver.A, "train before val")
    assert r["verdict"] == FAIL and r["value"] < 0
    assert row(run(overlap), ver.A, "val before test")["verdict"] == PASS


def test_a_up_share_of_train_is_reported_per_target(skilled):
    ds, idx, _ = skilled
    out = run(skilled)
    for t in TARGETS:
        key = head_spec("binary_classification", t).label_key
        r = row(out, ver.A, f"up share in train ({t})")
        assert r["verdict"] == INFO and r["value"] == round(float(np.mean(ds[key][idx["train"]] > 0)), 4)


# --------------------------------------------------------------------------------------------- b) baselines on test
def test_b_a_skilled_model_beats_the_majority_class_and_the_constant(skilled):
    out = run(skilled)
    for t in TARGETS:
        assert row(out, ver.B, f"direction accuracy {t}")["verdict"] == PASS
        assert row(out, ver.B, f"MAE constant - MAE model ({t}_reg")["verdict"] == PASS
    assert row(out, ver.B, "independent test days")["value"] == 88


def test_b_a_model_that_predicts_the_majority_class_is_not_a_pass():
    world = make_world(kind="majority")
    out = run(world)
    for t in TARGETS:
        r = row(out, ver.B, f"direction accuracy {t}")
        assert r["value"] == 0.0 and r["verdict"] == FLAT      # as accurate as the majority class: no gain, no loss
        r = row(out, ver.B, f"MAE constant - MAE model ({t}_reg")
        assert r["verdict"] != PASS


def test_b_an_unskilled_model_loses_to_the_baselines(unskilled):
    out = run(unskilled)
    for t in TARGETS:
        assert row(out, ver.B, f"direction accuracy {t}")["verdict"] == FAIL
        assert row(out, ver.B, f"MAE constant - MAE model ({t}_reg")["verdict"] == FAIL


def test_b_the_interval_is_by_day_blocks_not_rows(skilled):
    """Same rows, same gain, fewer days -> a wider interval: the day is the unit that is resampled."""
    ds, idx, tables = skilled
    wide = run(skilled)
    few = {s: idx[s][: len(idx[s]) // 4] for s in idx}
    few_tables = {s: tables[s].iloc[: len(few[s])].reset_index(drop=True) for s in tables}
    narrow = run((ds, few, few_tables))

    def width(o):
        lo, hi = row(o, ver.B, "MAE constant - MAE model (close_reg")["note"].split("[")[1][:-1].split(", ")
        return float(hi) - float(lo)
    assert width(narrow) > width(wide)


# -------------------------------------------------------------------------------------- c) confidence and trading
def test_c_confidence_slice_and_trades_pass_for_a_skilled_model(skilled):
    out = run(skilled)
    s = out["summary"]
    conf, trade = s[s["section"] == ver.C_CONF], s[s["section"] == ver.C_TRADE]
    assert len(conf) == 8 and set(conf["verdict"]) == {PASS}
    assert len(trade) == 4 and set(trade["verdict"]) == {PASS}
    sel = out["details"]["selective"]
    assert set(sel["score"]) == {"class_margin", "nig_edge", "conf_head", "agree_margin"}
    assert (sel["monotonic_val"] > 0.5).all() and (sel["monotonic_test"] > 0.5).all()
    assert (sel["test_acc"] >= 0.65).all() and (sel["test_wilson_lo"] > 0.5).all()
    assert (out["details"]["trades"]["test_t_days"] >= 2).all()


def test_c_an_unskilled_model_passes_nothing(unskilled):
    out = run(unskilled)
    s = out["summary"]
    assert set(s[s["section"].isin([ver.C_CONF, ver.C_TRADE])]["verdict"]) == {FAIL}
    assert "nothing tradeable" in ver.summary_text(out["summary"])


def test_c_a_slice_nobody_can_fill_fails_with_a_reason(skilled):
    out = run(skilled, target_acc=0.99, min_n=1000)           # val has 440 rows: no slice holds 1000
    r = row(out, ver.C_CONF, "slice >= 99% (class_margin)")
    assert r["verdict"] == FAIL and "no val slice reached 99%" in r["note"]
    assert row(out, ver.C_CONF, "confidence consistency (class_margin)")["verdict"] == PASS   # a separate check


def test_c_shuffled_test_labels_break_the_slice_even_with_a_good_val_threshold(skilled):
    ds, idx, tables = skilled
    perm = np.random.default_rng(3).permutation(len(idx["test"]))
    shuffled = {**ds}
    for key in [head_spec(k, "close").label_key for k in ("nig_regression", "binary_classification")]:
        col = np.array(ds[key])
        col[idx["test"]] = col[idx["test"]][perm]
        shuffled[key] = col
    out = run((shuffled, idx, tables))
    r = row(out, ver.C_CONF, "slice >= 65% (class_margin)")
    assert r["verdict"] == FAIL and "did not hold on test" in r["note"]
    assert row(out, ver.C_CONF, "confidence consistency (class_margin)")["verdict"] == FAIL


def test_c_model_levels_stop_needs_predicted_extremes_and_runs(skilled):
    out = run(skilled, stop_mode="model_levels")
    assert len(out["summary"][out["summary"]["section"] == ver.C_TRADE]) == 4
    with pytest.raises(ValueError, match="stop_mode"):
        run(skilled, stop_mode="atr")


def test_c_trades_are_skipped_without_the_high_low_heads(skilled):
    ds, idx, tables = skilled
    drop = {s: tables[s][[c for c in tables[s].columns if "high" not in c and "low" not in c]] for s in tables}
    out = run((ds, idx, drop), targets=("close",))
    assert row(out, ver.C_TRADE, "skipped")["verdict"] == INFO
    assert (out["summary"]["section"] == ver.C_CONF).sum() == 8          # direction does not need them


def test_c_skipped_without_the_close_head(skilled):
    ds, idx, tables = skilled
    drop = {s: tables[s][[c for c in tables[s].columns if "close" not in c]] for s in tables}
    out = run((ds, idx, drop), targets=("high", "low"))
    assert row(out, ver.C, "skipped")["verdict"] == INFO


def test_signals_frame_columns_and_units(skilled):
    ds, idx, tables = skilled
    state = TARGET_CODECS["return"]().fit(ds)
    df = ver.signals_frame(ds, idx["val"], tables["val"], state, TARGETS)
    assert {"mu", "sigma", "p_dir", "y_ret", "up", "day", "conf", "fut_high", "fut_low", "pred_high",
            "pred_low"} <= set(df)
    assert abs(float(df["sigma"].median()) - 0.01) < 1e-6           # the NIG predictive std of the world, return units
    assert (df["up"] == (df["y_ret"] > 0)).all() and df["day"].dtype.kind == "M"
    assert (df["day"] == df["day"].dt.normalize()).all()
    assert (df["fut_high"] >= df["y_ret"] - 1e-9).all() and (df["fut_low"] <= df["y_ret"] + 1e-9).all()


# ------------------------------------------------------------------------------- c) the pure functions, known answers
def trade_frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"sigma": [0.01] * n, "day": np.array(["2025-01-01"] * n, dtype="datetime64[D]"), "pred_low": [-0.01] * n,
            "pred_high": [0.01] * n}
    return pd.DataFrame({**base, **cols})


def test_simulate_rr_trades_known_exits_long_and_short():
    # long, d = 0.01, rr 2: tp at +0.02, sl at -0.01; cost 0.0008 per side = 0.16 R
    df = trade_frame(fut_high=[0.025, 0.005, 0.030, 0.008], fut_low=[-0.005, -0.012, -0.020, -0.004],
                     y_ret=[0.02, -0.011, 0.0, 0.004])
    tr = trading.simulate_rr_trades(df, np.ones(4), 2.0, "nig", COST)
    assert list(tr["exit"]) == ["tp", "sl", "both", "close"]
    np.testing.assert_allclose(tr["r"], [2 - 0.16, -1 - 0.16, -1 - 0.16, 0.4 - 0.16], atol=1e-9)
    # short: tp at -0.02, sl at +0.01
    df = trade_frame(fut_high=[0.004, 0.012], fut_low=[-0.025, -0.003], y_ret=[-0.02, 0.009])
    tr = trading.simulate_rr_trades(df, -np.ones(2), 2.0, "nig", COST)
    assert list(tr["exit"]) == ["tp", "sl"]
    np.testing.assert_allclose(tr["r"], [2 - 0.16, -1 - 0.16], atol=1e-9)


def test_simulate_rr_trades_invalid_distance_and_model_levels():
    df = trade_frame(fut_high=[0.0, 0.0], fut_low=[0.0, 0.0], y_ret=[0.0, 0.0], sigma=[0.0, np.nan])
    tr = trading.simulate_rr_trades(df, np.ones(2), 2.0, "nig", COST)
    assert list(tr["exit"]) == ["invalid", "invalid"] and tr["r"].isna().all()
    df = trade_frame(fut_high=[0.05], fut_low=[-0.001], y_ret=[0.03], pred_low=[-0.02], pred_high=[0.03])
    tr = trading.simulate_rr_trades(df, np.ones(1), 2.0, "model_levels", 0.0)       # d = 0.02: tp at 0.04
    assert tr["exit"][0] == "tp" and tr["r"][0] == 2.0
    tr = trading.simulate_rr_trades(df, -np.ones(1), 2.0, "model_levels", 0.0)      # short: d = 0.03, sl at +0.03
    assert tr["exit"][0] == "sl"


def test_selective_report_on_a_perfectly_monotone_and_on_a_flat_confidence():
    rng = np.random.default_rng(0)
    n = 2000
    conf = rng.random(n)
    up = (rng.random(n) < 0.4 + 0.55 * conf).astype(int)            # the more confident, the more often right (long)
    days = np.arange(n) % 50 + np.datetime64("2025-01-01")

    def frame():
        return pd.DataFrame({"p_dir": 0.5 + 0.5 * conf, "mu": conf - 0.5, "sigma": np.full(n, 0.01), "up": up,
                             "day": days})
    good = frame()
    good.attrs["p_dir_source"] = "close class head"
    rep = trading.selective_direction_report(good, good, target_acc=0.7, min_n=100).set_index("score")
    assert rep.loc["class_margin", "monotonic_val"] > 0.9 and rep.loc["class_margin", "verdict"].startswith(PASS)
    flat = frame()
    flat["up"] = (rng.random(n) < 0.5).astype(int)
    flat.attrs["p_dir_source"] = "close class head"
    rep = trading.selective_direction_report(flat, flat, target_acc=0.7, min_n=100).set_index("score")
    assert rep.loc["class_margin", "verdict"].startswith(FAIL)


# ------------------------------------------------------------------------------------------------- d) candle shape
@pytest.mark.slow   # sklearn boosting: seconds alone, minutes when the CPU is shared (rule E3)
def test_d_a_model_with_information_beyond_the_candle_shape_passes(skilled):
    out = run(skilled, run_candle=True)
    d = out["summary"][out["summary"]["section"] == ver.D]
    assert len(d) == 2 and set(d["verdict"]) == {PASS}
    cb = out["details"]["candle"]
    assert set(cb["verdicts"]) == {"high", "low"}
    assert set(cb["table"]["score"]) >= {"rule_cpos", "candle_gbm", "model_class"}
    assert cb["verdicts"]["high"]["best_baseline"] in BASELINES
    assert (cb["table"]["majority_baseline"] >= 0.5).all()


@pytest.mark.slow
def test_d_a_model_the_candle_shape_beats_fails(unskilled):
    d = run(unskilled, run_candle=True)["summary"]
    d = d[d["section"] == ver.D]
    assert len(d) == 2 and set(d["verdict"]) == {FAIL} and d["note"].str.contains("adds nothing").all()


def test_d_skipped_without_high_low_class_heads_and_off_by_the_flag(skilled):
    ds, idx, tables = skilled
    drop = {s: tables[s][[c for c in tables[s].columns if "close" in c or "timestamp" in c or c in
                          ("asset", "horizon_ns", "prediction_made_at", "label_available_at", "identity_hash", "stage",
                           "fold", "train_end", "schema_version")]] for s in tables}
    out = run((ds, idx, drop), targets=("close",), run_candle=True)
    assert row(out, ver.D, "skipped")["verdict"] == INFO
    assert ver.D not in set(run(skilled, run_candle=False)["summary"]["section"])


# ---------------------------------------------------------------------------------------------------- the writer
def test_the_summary_is_written_as_csv_and_json_and_printed(skilled, tmp_path, capsys):
    out = run(skilled, out_dir=tmp_path / "nested", verbose=True)
    csv = pd.read_csv(tmp_path / "nested" / ver.SUMMARY_CSV, encoding="utf-8-sig")
    assert list(csv.columns) == ["section", "check", "value", "verdict", "note"] and len(csv) == len(out["summary"])
    records = json.loads((tmp_path / "nested" / ver.SUMMARY_JSON).read_text(encoding="utf-8"))
    assert len(records) == len(out["summary"]) and all(isinstance(r["value"], str) for r in records)
    printed = capsys.readouterr().out
    assert "INTEGRATED VERIFICATION" in printed and "conclusion:" in printed
    assert not (tmp_path / "x").exists()


def test_the_world_is_deterministic_and_the_helpers_agree():
    ds, truth = make_dataset(0)
    ds2, _ = make_dataset(0)
    assert np.array_equal(ds["last_candles"], ds2["last_candles"])
    idx = make_idx(ds)
    assert make_table(ds, idx["val"], kind="skilled", seed=4).equals(make_table(ds, idx["val"], kind="skilled", seed=4))
    with pytest.raises(ValueError):
        make_table(ds, idx["val"], kind="oracle")
