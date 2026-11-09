"""eval/deep_side.py: the derived target and its decode, the evidence rule (a real signal passes, the close head
renamed fails), VAL-only thresholds, the clip count; workflow/deep_side_run.py writes its reports."""

import json

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import deep_side as ds


def frame(signal: float, seed: int, n_coins=25, n_ts=120, close_is_model=False) -> pd.DataFrame:
    """Synthetic battery rows: true deep z, model score = signal * z + noise, baselines independent of z."""
    rng = np.random.default_rng(seed)
    n = n_coins * n_ts
    z = rng.normal(0, 0.01, n)
    m = signal * z + rng.normal(0, 0.01, n)
    vol = rng.lognormal(-4, 0.3, n)
    mu = np.expm1(m) if close_is_model else rng.normal(0, 0.002, n)
    y = z / 2 + rng.normal(0, 0.002, n)
    return pd.DataFrame({
        "mu": mu, "p_dir": rng.random(n), "y_ret": y, "vol": vol, "last_ret": rng.normal(0, 0.01, n),
        "coin": np.tile([f"C{i}" for i in range(n_coins)], n_ts),
        "end_time": pd.Timestamp("2026-03-01") + pd.to_timedelta(np.repeat(np.arange(n_ts), n_coins) * 8, "h"),
        "fut_high": np.expm1(0.02 + z / 2), "fut_low": np.expm1(-0.02 + z / 2),
        "pred_high": np.expm1(0.015 + m / 2), "pred_low": np.expm1(-0.015 + m / 2),
        "last_bar_deep": rng.normal(0, 0.01, n)})


def frames(**kw):
    v, t = (frame(seed=s, **kw) for s in (1, 2))
    return ds.deep_frame(v, v["last_bar_deep"].values), ds.deep_frame(t, t["last_bar_deep"].values)


def test_deep_is_the_sum_of_the_log_extremes_and_the_prediction_is_mu_high_plus_mu_low():
    d = frames(signal=1.0)[0]
    raw = frame(1.0, 1)
    z_pred = np.log1p(raw["pred_high"]) + np.log1p(raw["pred_low"])
    assert np.allclose(d["deep"], np.log1p(raw["fut_high"]) + np.log1p(raw["fut_low"]))
    assert np.allclose(d["deep_pred"], z_pred)  # logret_close: decoded log return = mu, so the sum is mu_h + mu_l
    assert set(d["day"].astype(str)) and d["ts"].nunique() == 120


def test_a_real_signal_beyond_the_baselines_is_evidence_on_both_splits():
    res = ds.deep_side(*frames(signal=1.0))
    v = res["verdict"]
    assert v["skill_above_every_baseline_both_splits"] and v["profit_section"] == "run"
    assert res["skill"]["TEST"]["model"]["ic"]["ic"] > 0.4
    assert abs(res["skill"]["TEST"]["random"]["ic"]["ic"]) < 0.05
    assert abs(res["skill"]["TEST"]["shuffled model"]["ic"]["ic"]) < 0.05


def test_the_close_head_renamed_is_not_evidence_and_no_signal_is_not_evidence():
    res = ds.deep_side(*frames(signal=1.0, close_is_model=True))
    diff = res["skill"]["VAL"]["close head mu"]["model_minus_it"]
    assert diff["ic"] == pytest.approx(0.0, abs=1e-12) and not diff["lo"] > 0, "same ranks: zero paired difference"
    assert not res["verdict"]["skill_above_every_baseline_both_splits"]
    assert res["close_link"]["VAL"]["ic_pred_vs_close_mu"] == pytest.approx(1.0)
    none = ds.deep_side(*frames(signal=0.0))
    assert not none["verdict"]["checks"]["TEST"]["ic_ci_above_0"]
    assert none["verdict"]["profit_section"].startswith("not reached")


def test_thresholds_come_from_val_and_clipped_rows_are_counted():
    val, test = frames(signal=1.0)
    signs = ds.orient(val)
    nt = ds.near_ties(val, test, signs, share=0.2)
    assert nt["margin"] == pytest.approx(np.quantile(np.abs(val["deep"]), 0.2))
    assert nt["VAL"]["dropped"] == int((np.abs(val["deep"]) < nt["margin"]).sum())
    assert nt["TEST"]["dropped"] == int((np.abs(test["deep"]) < nt["margin"]).sum())
    vb = ds.vol_buckets(val, test, signs, n=3)
    assert vb["edges"] == pytest.approx(list(np.quantile(val["vol"], [1 / 3, 2 / 3])))
    assert sum(r["n"] for r in vb["TEST"].values()) == len(test)
    clipped = test.copy()
    clipped.loc[:4, "fut_high"] = np.expm1(1.0)
    clipped.loc[3:7, "fut_low"] = np.expm1(-1.0)
    c = ds.clip_counts(clipped)
    assert (c["high"], c["low"], c["high_or_low"], c["high_and_low"]) == (5, 5, 8, 2)


def test_the_step_writes_both_reports(tmp_path, monkeypatch):
    from nig_timenet_torch.workflow import deep_side_run as run_mod
    val, test = frames(signal=1.0)
    monkeypatch.setattr(run_mod, "deep_frames", lambda run: {"VAL": val, "TEST": test})
    res = run_mod.deep_side_step(None, tmp_path)
    assert "VERDICT" in (tmp_path / run_mod.REPORT).read_text()
    assert json.loads((tmp_path / run_mod.JSON).read_text())["verdict"] == json.loads(json.dumps(
        res["verdict"], default=run_mod._json))
