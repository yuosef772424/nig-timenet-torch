"""E1 generator (ROADMAP G1): the oracle columns are the labels' exact conditional moments, the planted IC is known,
the oracle's T-sharp share is 1 / 1 / ~0, and the artifact goes through the normal loader and run steps."""

import numpy as np
import pytest

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
from nig_timenet_torch.data_io import synthetic
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, bar_kappa, build_synthetic, oracle_keys
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.workflow import run, run_common
from nig_timenet_torch.workflow.settings import RunSettings
from tests.data_io import envs

HEADS = ("high", "low", "close")
SHA = "b" * 40


@pytest.fixture(scope="module")
def e1():
    return envs.small()


@pytest.fixture(scope="module")
def e1_return():
    return envs.small(label_codec="return", reg_target_scale=100.0, seed=1)


def _z(ds, t):
    y = np.asarray(ds[head_spec("nig_regression", t).label_key], np.float64)
    mean, sd = (np.asarray(ds[k]) for k in oracle_keys(t))
    return (y - mean) / sd


@pytest.mark.parametrize("which", ["e1", "e1_return"])
def test_oracle_columns_are_the_conditional_moments_within_mc_error(which, request):
    ds = request.getfixturevalue(which)
    sd = np.asarray(ds[f"{synthetic.ORACLE_PREFIX}sigma"])
    signal = np.asarray(ds[f"{synthetic.ORACLE_PREFIX}signal"])
    for t in HEADS:
        z = _z(ds, t)
        # conditional: in every tercile of the planted volatility and of the planted signal, separately
        for key in (sd, signal):
            for part in np.array_split(np.argsort(key), 3):
                zz = z[part]
                se_m, se_v = zz.std() / np.sqrt(len(zz)), (zz ** 2).std() / np.sqrt(len(zz))
                assert abs(zz.mean()) < 4 * se_m, (which, t, zz.mean(), se_m)
                assert abs((zz ** 2).mean() - 1.0) < 4 * se_v, (which, t, (zz ** 2).mean(), se_v)


@pytest.mark.parametrize("ic", [0.05, 0.10])
def test_planted_close_ic_is_known(ic):
    ds = envs.small(close_ic=ic, n_coins=48)
    got = ds["synthetic"]["summary"]["close"]["planted_ic_pearson"]
    n = len(ds["last_candles"])
    assert abs(got - ic) < 4 * (1 - ic ** 2) / np.sqrt(n), (got, ic)
    assert ds["synthetic"]["summary"]["close"]["rank_ic_by_timestamp"] > 0.5 * ic


def test_oracle_t_sharp_share_is_one_one_and_about_zero(e1):
    s = e1["synthetic"]["summary"]
    assert s["high"]["share_sd_below_move"] == 1.0 and s["low"]["share_sd_below_move"] == 1.0
    assert s["close"]["share_sd_below_move"] < 0.02


@pytest.mark.parametrize("which", ["e1", "e1_return"])
def test_target_report_on_the_oracle(which, request):
    """The REAL report code (eval.target_report) on the oracle as a prediction: T-sharp 1 / 1 / ~0 for both codecs,
    confidence = -sd ranks the error. Coverage of the symmetric 80 % interval: 0.80 for close (Gaussian label), but
    about 0.88 for high / low even with the TRUE mean and sd (skewed labels, bounded at 0): a symmetric NIG head
    cannot reach [0.75, 0.85] there by calibrating its sd; the summary records the oracle's value for the gates."""
    ds = request.getfixturevalue(which)
    table, labels, anchors, last_close, state = envs.oracle_table(ds)
    r = target_report(table, labels, anchors=anchors, last_close=last_close, codec_state=state)["targets"]
    assert r["high"]["sharp"]["share_std_below_move"] == 1.0 and r["low"]["sharp"]["share_std_below_move"] == 1.0
    assert r["close"]["sharp"]["share_std_below_move"] < 0.02 and r["close"]["sharp"]["blocking"] is False
    summary = ds["synthetic"]["summary"]
    assert 0.78 <= r["close"]["sharp"]["coverage"] <= 0.82
    for t in ("high", "low"):
        assert 0.86 <= r[t]["sharp"]["coverage"] <= 0.90
    for t in HEADS:
        assert r[t]["sharp"]["coverage"] == pytest.approx(summary[t]["coverage_symmetric_80"], abs=2e-3)
        assert r[t]["conf"]["decile_error_spearman"] <= -0.8, (t, r[t]["conf"])
    assert r["high"]["skill"]["rank_ic_by_timestamp"] > 0.2 and r["low"]["skill"]["rank_ic_by_timestamp"] > 0.2


def test_bars_are_consistent_intrabar_paths(e1):
    x = np.asarray(e1["X_4h"], np.float64)
    hi, lo, cl = x[..., 0], x[..., 1], x[..., 2]
    assert (hi >= np.maximum(cl, 0.0) - 1e-6).all() and (lo <= np.minimum(cl, 0.0) + 1e-6).all()
    lc = np.asarray(e1["last_candles"])
    col = {c: lc[:, LAST_COLUMNS.index(c)] for c in LAST_COLUMNS}
    per = e1["asset_bounds"][0]["end"]
    h = e1["forecast_horizon"]
    # the next row (stride = horizon) holds this row's future bars: its last close is this row's future close, its
    # window's last h bars give this row's future high / low
    nxt = np.arange(len(lc)) + 1
    same = (nxt % per) != 0
    i, j = np.flatnonzero(same), nxt[same]
    np.testing.assert_allclose(col["future_close"][i], col["last_close"][j], rtol=1e-9)
    c0 = col["last_close"][i]
    log_prev = np.log(c0)[:, None] + np.concatenate([np.zeros((len(i), 1)), np.cumsum(x[j, -h:, 2], axis=1)[:, :-1]],
                                                    axis=1)
    np.testing.assert_allclose(col["future_high_max"][i], np.exp(log_prev + x[j, -h:, 0]).max(axis=1), rtol=1e-5)
    np.testing.assert_allclose(col["future_low_min"][i], np.exp(log_prev + x[j, -h:, 1]).min(axis=1), rtol=1e-5)
    ts = lc[:, TS_COL].astype(np.int64)
    assert (np.diff(ts[:per]) == h * 4 * 3600 * 10**9).all()


def test_defaults_mirror_the_real_artifact_scale_and_its_beta_floor_pathology():
    """Default label scale 1.0 and magnitudes of the real artifact c572b1b36960 (bar range median 0.029, close label
    sd 0.039, high label mean 0.025): y^2 is far below the NIG beta floor 0.01 (B-052), on purpose."""
    s = SyntheticSettings()
    assert (s.reg_target_scale, s.window, s.horizon, s.base_tf, s.label_codec) == (1.0, 32, 2, "4h", "logret_close")
    ds = envs.small()
    x = np.asarray(ds["X_4h"])
    assert 0.015 < float(np.median(x[:, -1, 0] - x[:, -1, 1])) < 0.045
    for t in HEADS:
        y = np.asarray(ds[head_spec("nig_regression", t).label_key], np.float64)
        assert 0.01 < y.std() < 0.06 and (y ** 2).mean() < 0.01 / 5
    assert ds["synthetic"]["settings"]["reg_target_scale"] == 1.0 and ds["reg_target_scale"] == 1.0


def test_label_scale_is_a_parameter():
    a, b = envs.small(n_coins=4), envs.small(n_coins=4, reg_target_scale=100.0)
    for t in HEADS:
        key = head_spec("nig_regression", t).label_key
        np.testing.assert_allclose(b[key], 100.0 * a[key], rtol=1e-5)
        np.testing.assert_allclose(b[oracle_keys(t)[1]], 100.0 * a[oracle_keys(t)[1]], rtol=1e-12)
    np.testing.assert_array_equal(a["X_4h"], b["X_4h"])


def test_deterministic_for_a_seed():
    a, b, c = envs.small(n_coins=4), envs.small(n_coins=4), envs.small(n_coins=4, seed=7)
    np.testing.assert_array_equal(a["X_4h"], b["X_4h"])
    assert not np.array_equal(a["X_4h"], c["X_4h"])


def test_kappa_is_the_mean_range_of_a_bar():
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.standard_normal((100_000, 48)) / np.sqrt(48), axis=1)
    rng_ = np.maximum(path.max(axis=1), 0) - np.minimum(path.min(axis=1), 0)
    assert bar_kappa(48) == pytest.approx(rng_.mean(), rel=0.01)


def test_settings_are_strict():
    with pytest.raises(ValueError, match="did you mean|unknown"):
        from_dict(SyntheticSettings, {"n_coin": 3})
    for bad in ({"label_codec": "magnitude"}, {"close_ic": 1.0}, {"n_bars": 2001}, {"window": 4},
                {"har_weight": 1.0}, {"burn_in_bars": 16}):
        with pytest.raises(ValueError):
            SyntheticSettings(**bad)


@pytest.mark.parametrize("codec", ["logret_close", "return"])
def test_contract_artifact_through_the_loader_and_the_run_steps(tmp_path, codec):
    """Rule D1 and the real path: written by the artifact writer, read by load_dataset, decoded by the codec the
    real artifact uses (check_decode), prepared by workflow.run.prepare."""
    s = SyntheticSettings(n_coins=4, n_bars=400, label_codec=codec)
    man = synthetic.write_synthetic(tmp_path / "e1.dataset", s, source_commit=SHA)
    assert man["export"]["producer"] == "nig_timenet_torch.data_io.synthetic"
    loaded = load_dataset(LoaderSettings(directory=str(tmp_path), name="e1.dataset", local_dir=""))
    ds = loaded.dataset
    assert loaded.manifest is not None and loaded.manifest.source_commit == SHA
    assert run_common.codec_state(ds)["mode"] == codec
    run_common.check_decode(ds, run_common.price_targets(ds))
    assert ds["synthetic"]["settings"]["label_codec"] == codec
    np.testing.assert_array_equal(ds[oracle_keys("high")[0]], build_synthetic(s)[oracle_keys("high")[0]])
    settings = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(directory=str(tmp_path),
                                                                              name="e1.dataset", local_dir=""))
    r = run.prepare(settings)
    assert r.x["train"].shape[1:] == (32, 3) and len(r.idx["test"]) > 0
