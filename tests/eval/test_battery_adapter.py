"""The battery adapter on the fixture NIG table: column mapping, close-only direction, errors, and the whole
battery running on its output."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval.battery import full_battery
from nig_timenet_torch.eval.battery_adapter import BATTERY_COLUMNS, LIMIT_COLUMNS, to_battery_table
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import nig_predictive_std
from tests.eval.synth import ANCHOR, codec_state, make


def _adapt(table, labels, anchors, **kw):
    return to_battery_table(table, labels, anchors=anchors, last_close=np.full(len(table), ANCHOR),
                            codec_state=codec_state(), **kw)


def test_column_mapping_matches_the_hand_computation():
    table, labels, anchors = make()
    out = _adapt(table, labels, anchors)
    spec = head_spec("nig_regression", "close")
    assert list(out.columns[:6]) == list(BATTERY_COLUMNS[:6]) and "last_ret" not in out
    assert set(LIMIT_COLUMNS) <= set(out.columns)
    np.testing.assert_allclose(out["mu"], table[spec.output_keys["mu"]])          # scale 1, anchor == last_close
    np.testing.assert_allclose(out["y_ret"], labels[spec.label_key])
    std = nig_predictive_std(*(table[spec.output_keys[k]] for k in ("nu", "alpha", "beta")))
    np.testing.assert_allclose(out["sigma"], std)
    np.testing.assert_allclose(out["p_dir"], table[head_spec("binary_classification", "close").output_keys["logits"]])
    assert (out["dir_true"] == np.sign(out["y_ret"])).all() and (out["step"] == 1).all()
    assert out["coin"].tolist() == table["asset"].tolist()
    assert out["end_time"].tolist() == table["prediction_made_at"].dt.tz_localize(None).tolist()
    hi = head_spec("nig_regression", "high")
    np.testing.assert_allclose(out["pred_high"], table[hi.output_keys["mu"]])
    np.testing.assert_allclose(out["fut_high"], labels[hi.label_key])
    assert out.attrs["p_dir_source"] == "close class head"


def test_decode_uses_scale_and_own_anchor():
    table, labels, anchors = make()
    state = codec_state() | {"scale": 10.0}
    last = np.full(len(table), ANCHOR)
    out = to_battery_table(table, labels, anchors=anchors, last_close=last, codec_state=state)
    spec = head_spec("nig_regression", "close")
    np.testing.assert_allclose(out["mu"], table[spec.output_keys["mu"]] / 10.0)
    anchors_hi = dict(anchors, high=np.full(len(table), 110.0))                 # high anchored 10 % above the close
    out = to_battery_table(table, labels, anchors=anchors_hi, last_close=last, codec_state=codec_state())
    hi_mu = table[head_spec("nig_regression", "high").output_keys["mu"]]
    np.testing.assert_allclose(out["pred_high"], 1.1 * (1 + hi_mu) - 1)


def test_direction_is_the_close_head_only():
    """High / low class outputs are poisoned: nothing in the battery table may move, no high / low direction exists."""
    table, labels, anchors = make()
    base = _adapt(table, labels, anchors)
    poisoned = table.copy()
    for t in ("high", "low"):
        poisoned[head_spec("binary_classification", t).output_keys["logits"]] = 1.0
    pd.testing.assert_frame_equal(_adapt(poisoned, labels, anchors), base)
    assert not [c for c in base.columns if "class" in c or c in ("p_high", "p_low", "dir_high", "dir_low")]


def test_missing_class_head_falls_back_to_gaussian_and_prev_close_adds_last_ret():
    table, labels, anchors = make()
    cls = head_spec("binary_classification", "close").output_keys["logits"]
    out = _adapt(table.drop(columns=[cls]), labels, anchors, prev_close=np.full(len(table), 99.0))
    assert out.attrs["p_dir_source"].startswith("gaussian")
    assert ((out["p_dir"] > 0.5) == (out["mu"] > 0)).all() and out["p_dir"].between(0, 1).all()
    np.testing.assert_allclose(out["last_ret"], 100.0 / 99.0 - 1)


def test_close_only_table_and_bad_inputs():
    table, labels, anchors = make()
    drop = [c for c in table.columns if c.startswith("y_high") or c.startswith("y_low")]
    out = _adapt(table.drop(columns=drop), labels, anchors)
    assert not set(LIMIT_COLUMNS) & set(out.columns)
    with pytest.raises(ValueError, match="close NIG head"):
        _adapt(table.drop(columns=["y_close"]), labels, anchors)
    with pytest.raises(ValueError, match="last_close"):
        to_battery_table(table, labels, anchors=anchors, last_close=np.zeros(len(table)), codec_state=codec_state())
    with pytest.raises(KeyError):
        _adapt(table, {}, anchors)
    with pytest.raises(ValueError, match="empty"):
        _adapt(table.iloc[:0], labels, anchors)


def test_the_whole_battery_runs_on_the_adapted_nig_table():
    table, labels, anchors = make(skilled=True)
    val = _adapt(table, labels, anchors, prev_close=np.full(len(table), 99.0))
    out = full_battery(val, val.copy())
    assert {"profit_close", "profit_limit", "profit_sigma", "verdict", "neutral", "baseline"} <= set(out)
    for name, section in out.items():
        assert section["rows"], name
    assert "SKIPPED" not in out["neutral"]["title"]      # 6 coins x 1000 hourly periods: a cross-section exists
    no_prev = full_battery(_adapt(table, labels, anchors), _adapt(table, labels, anchors))
    assert "SKIPPED" in no_prev["baseline"]["title"]
