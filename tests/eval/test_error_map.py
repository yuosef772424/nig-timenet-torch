"""eval/error_map.py on a synthetic frame with a known bias, a known bad bucket and an uninformative std."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import error_map as em


def frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.lognormal(np.log(0.01), 0.4, n)
    mu = vol * 2
    y = 1.5 * mu + vol * rng.normal(0, 1, n)  # the model understates the move by a third
    std = np.full(n, vol.mean())
    t = pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(n) % 500 * 4, "h")
    f = pd.DataFrame({"coin": np.array([f"C{i}" for i in range(12)])[np.arange(n) % 12], "end_time": t,
                      "vol": vol, "age_bars": np.arange(n), "stale_share": (np.arange(n) % 12) / 12.0,
                      "mu_high": mu, "real_high": y, "std_high": std, "half_high": 1.28 * std,
                      "nll_high": np.abs(y - mu) / std, "har_high": 2 * vol})
    f["day"] = f["end_time"].values.astype("datetime64[D]")
    f["month"], f["hour"] = f["end_time"].dt.strftime("%Y-%m"), f["end_time"].dt.hour
    f["market_day"] = rng.normal(0, 0.01, n)
    return f


def test_bias_uncertainty_buckets_and_worst():
    f = frame()
    b = em.bias(f, "high")
    assert b["slope"] == pytest.approx(1.5, abs=0.1)
    u = em.uncertainty(f, "high")
    assert np.isnan(u["std_err_spearman"]), "a constant std carries no information about the error"
    rows = em.bucket_rows(f, "high", "vol", em.edges_from(f, "vol", 5))
    assert len(rows) == 5 and rows[-1]["mae"] > rows[0]["mae"]  # errors concentrate in the violent bucket
    assert all(abs(r["net_of_har_ic"]) < 0.1 or np.isnan(r["net_of_har_ic"]) for r in rows)
    m = em.error_map({"VAL": f, "TEST": frame(seed=1)}, heads=("high",))
    assert set(m) == {"high | VAL", "high | TEST"} and "by vol" in em.render(m)
    info = pd.DataFrame({"stale_share": np.arange(12) / 12.0}, index=[f"C{i}" for i in range(12)])
    w = em.worst_cases(f, "high", k=3, coin_info=info)
    assert len(w["coins"]) == 3 and "stale_share" in w["worst_vs_all"]
