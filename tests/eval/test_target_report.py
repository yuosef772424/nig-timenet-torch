"""T4 target_report on synthetic tables with a known answer (no model, no fixture)."""

import numpy as np
import pytest
import scipy.stats

from nig_timenet_torch.eval.target_report import COVERAGE_RANGE, target_report
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import nig_interval_halfwidth, nig_predictive_std, student_t_quantile
from tests.eval import synth


def report(table, labels, anchors):
    return target_report(table, labels, anchors=anchors, codec_state=synth.codec_state())


def test_t_quantile_matches_scipy_where_the_nig_head_lives():
    for df in (4.0, 6.0, 10.0, 40.0):
        for p in (0.6, 0.9, 0.975):
            assert student_t_quantile(p, np.array([df]))[0] == pytest.approx(scipy.stats.t.ppf(p, df), abs=3e-3)
    with pytest.raises(ValueError):
        student_t_quantile(0.9, np.array([2.0]))


def test_predictive_std_is_the_student_t_std():
    nu, a, b = np.array([2.0]), np.array([5.0]), np.array([0.3])
    scale = np.sqrt(b * (1 + nu) / (nu * a))
    assert nig_predictive_std(nu, a, b)[0] == pytest.approx(float(scipy.stats.t.std(2 * a[0], scale=scale[0])))
    assert nig_interval_halfwidth(nu, a, b, 0.8)[0] == pytest.approx(
        float(scipy.stats.t.ppf(0.9, 2 * a[0], scale=scale[0])), rel=3e-3)


def test_skilled_table_known_answers():
    table, labels, anchors = synth.make(conf_informative=False, skilled=True)
    r = report(table, labels, anchors)
    assert set(r["targets"]) == {"high", "low", "close"} and r["n_rows"] == len(table)
    for t, rep in r["targets"].items():
        sk = rep["skill"]
        # truth = mu + noise with signal_sd == noise scale: Spearman ~ (6/pi) asin(0.707 / 2) = 0.69 per coin
        assert sk["spearman_per_coin"]["mean"] == pytest.approx(0.67, abs=0.05), t
        assert sk["spearman_per_coin"]["share_positive"] == 1.0 and sk["spearman_per_coin"]["n_coins"] == synth.N_COINS
        assert sk["rank_ic_by_timestamp"] > 0.5 and abs(sk["rank_ic_by_timestamp_shuffled_baseline"]) < 0.06
        assert sk["class"]["auc"] > 0.8 and sk["class"]["auc_constant_baseline"] == 0.5
        assert sk["class"]["auc_shuffled_baseline"] == pytest.approx(0.5, abs=0.03)
        # the 80 % interval from nu / alpha / beta covers 80 % when the noise IS that Student-t
        assert rep["sharp"]["coverage"] == pytest.approx(0.8, abs=0.02) and rep["sharp"]["coverage_within_target"]
        assert COVERAGE_RANGE[0] < min(rep["sharp"]["coverage_by_coin_min_max"]) < 0.8 + 0.05
        # homoscedastic noise and a constant confidence: no confidence signal at all
        assert rep["conf"]["decile_error_spearman"] == 0.0 and rep["conf"]["meets_target"] is False
        assert rep["conf"]["top_minus_bottom_error_gap"] == 0.0


def test_unskilled_table_has_no_skill():
    table, labels, anchors = synth.make(conf_informative=False, skilled=False)
    for rep in report(table, labels, anchors)["targets"].values():
        assert abs(rep["skill"]["spearman_per_coin"]["mean"]) < 0.05
        assert abs(rep["skill"]["rank_ic_by_timestamp"]) < 0.06
        assert rep["skill"]["class"]["auc"] == pytest.approx(0.5, abs=0.04)


def test_perfect_prediction():
    table, labels, anchors = synth.make(conf_informative=False)
    reg = head_spec("nig_regression", "close")
    table[reg.output_keys["mu"]] = labels[reg.label_key]
    sk = report(table, labels, anchors)["targets"]["close"]["skill"]
    assert sk["rank_ic_by_timestamp"] == pytest.approx(1.0) and sk["spearman_per_coin"]["mean"] == pytest.approx(1.0)


def test_informative_confidence_meets_t_conf_and_constant_does_not():
    table, labels, anchors = synth.make(conf_informative=True)
    for rep in report(table, labels, anchors)["targets"].values():
        c = rep["conf"]
        assert c["decile_error_spearman"] <= -0.8 and c["gap_ci95"][0] > 0.0 and c["meets_target"]
    flat = table.copy()
    for t in ("high", "low", "close"):
        flat[head_spec("nig_regression", t).output_keys["confidence"]] = 0.5
    assert all(rep["conf"]["decile_error_spearman"] == 0.0 and not rep["conf"]["meets_target"]
               for rep in report(flat, labels, anchors)["targets"].values())


def test_t_sharp_share_follows_width_versus_range():
    # centres +-0.02 on anchor 100 -> range ~ 4 price units; NIG std 100 * 0.0056 = 0.56 -> narrow; 0.05 -> wide
    narrow = report(*synth.make(conf_informative=False, scale=0.005))["targets"]
    wide = report(*synth.make(conf_informative=False, scale=0.05, signal_sd=0.005))["targets"]
    for t in ("high", "low", "close"):
        assert narrow[t]["sharp"]["share_width_below_range"] > 0.95 and narrow[t]["sharp"]["meets_target"]
        assert narrow[t]["sharp"]["zero_width_baseline_share"] == 1.0
        assert wide[t]["sharp"]["share_width_below_range"] < 0.05 and not wide[t]["sharp"]["meets_target"]


def test_raises_on_bad_input():
    table, labels, anchors = synth.make(conf_informative=False)
    with pytest.raises(KeyError):
        report(table, {k: v for k, v in labels.items() if k != "y_close_reg"}, anchors)
    with pytest.raises(ValueError):
        report(table.iloc[0:0], labels, anchors)
    with pytest.raises(ValueError):
        report(table[[c for c in table.columns if not c.startswith("y_")]], labels, anchors)
