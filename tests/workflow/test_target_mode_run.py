"""split.target_mode: the run retargets its rows after the split (data.targets.retarget_splits), the labels decode to
the real prices (check_decode), the consumers go through the target-mode codec and skip where a price forecast does
not exist. The default (target_mode None) is the run of every earlier PR."""

import numpy as np
import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data.targets import center_key, target_arrays
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.battery_adapter import to_battery_table
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings
from tests.workflow.conftest import art  # noqa: F401  (fixture)

TARGETS = ("high", "low", "close")
MODES = ["return_close", "scaled", "window_scale", "volnorm", "magnitude", "entry_range", "relative", "scaled+relative",
         "magnitude+relative"]


def settings(art_dir, tmp_path, **over):
    base = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art_dir)))
    return updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", **over})


def test_the_default_run_is_untouched(art, tmp_path):  # noqa: F811
    base = steps.prepare(settings(art, tmp_path))
    assert SplitSettings().target_mode is None
    assert "pipeline_targets" not in base.dataset and base.dataset["reg_target_mode"] == "return"
    assert run_common.retarget_art(base.art, SplitSettings(), base.targets, tuple(base.idx.values())) is base.art


def test_settings_are_strict_about_the_target_mode():
    with pytest.raises(ValueError, match="unknown target mode"):
        SplitSettings(target_mode="retrun")
    with pytest.raises(ValueError, match="not supported"):
        SplitSettings(target_mode="entry_range+relative")
    with pytest.raises(ValueError, match="entry_close_reg"):
        SplitSettings(entry_close_reg="bogus")
    with pytest.raises(ValueError, match="target_clip"):
        SplitSettings(target_clip=0.0)
    assert SplitSettings(target_mode="relative").target_mode == "relative"


@pytest.mark.parametrize("mode", MODES)
def test_prepare_retargets_the_rows_of_the_split_and_the_labels_decode(art, tmp_path, mode):  # noqa: F811
    run = steps.prepare(settings(art, tmp_path, **{"split.target_mode": mode}))
    ds = run.dataset
    assert ds["target_mode"] == mode and ds["reg_target_mode"] == mode and "pipeline_targets" in ds
    state = run_common.codec_state(ds)
    assert state["mode"] == "target_raw" and state["target_mode"] == mode
    run_common.check_decode(ds, run.targets)                                    # the round trip, on the new labels
    idx = run.idx["train"]
    last, bases = np.asarray(ds["last_candles"]), np.asarray(ds["base_params"])
    y, _ = target_arrays(last[idx], bases[idx], mode, TARGETS, scale=100.0, clip=ds["reg_target_clip"])
    np.testing.assert_allclose(run.y["train"]["y_close_reg"].numpy(), y["close_reg"], rtol=1e-6)
    np.testing.assert_array_equal(run.y["train"]["y_close_class"].numpy(), y["close_class"])
    assert ("+relative" in mode or mode == "relative") == (center_key("close") in ds)
    assert run.art.manifest is not None                                          # the run identity still reads it


def test_the_magnitude_threshold_is_the_train_median_in_a_run(art, tmp_path):  # noqa: F811
    run = steps.prepare(settings(art, tmp_path, **{"split.target_mode": "magnitude"}))
    ds = run.dataset
    last = np.asarray(ds["last_candles"])
    r = np.abs(last[:, LAST_COLUMN_INDEX["future_close"]] / last[:, LAST_COLUMN_INDEX["last_close"]] - 1.0)
    thr = float(np.median(r[run.idx["train"]]))
    for stage in ("train", "val"):
        got = run.y[stage]["y_close_class"].numpy()
        np.testing.assert_array_equal(got, (r[run.idx[stage]] > thr).astype("float32"))


def test_evaluation_anchors_decode_without_truth_and_need_the_dataset(art, tmp_path):  # noqa: F811
    run = steps.prepare(settings(art, tmp_path, **{"split.target_mode": "scaled"}))
    ds, idx = run.dataset, run.idx["val"]
    state = run_common.codec_state(ds)
    lc = np.asarray(ds["last_candles"])
    a = flat_run.anchors_by_name(lc, idx, TARGETS, state, dataset=ds)
    assert all(a[t].shape == (len(idx), 2) for t in TARGETS)
    codec = TARGET_CODECS[state["mode"]]()
    y = run.y["val"]["y_close_reg"].numpy().astype(np.float64)
    price = codec.decode(y, state, a["close"])
    np.testing.assert_allclose(price, lc[idx, LAST_COLUMN_INDEX["future_close"]], rtol=1e-5)
    with pytest.raises(ValueError, match="dataset"):
        flat_run.anchors_by_name(lc, idx, TARGETS, state)


def test_the_battery_adapter_refuses_a_target_without_a_price_forecast(art, tmp_path):  # noqa: F811
    run = steps.prepare(settings(art, tmp_path, **{"split.target_mode": "entry_range"}))
    state = run_common.codec_state(run.dataset)
    assert not TARGET_CODECS[state["mode"]]().forecasts_price(state)
    with pytest.raises(ValueError, match="price forecast"):
        to_battery_table(None, {}, anchors={}, last_close=np.ones(1), codec_state=state)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["scaled", "entry_range"])
def test_a_whole_run_in_another_target_mode(art, tmp_path, mode):  # noqa: F811
    """scaled: every step runs and the battery is built; entry_range: every step runs and the battery is skipped
    (a magnitude has no sign to trade)."""
    out = tmp_path / "out"
    r = steps.run_all(settings(art, tmp_path, **{"split.target_mode": mode, "split.max_rows": 60,
                                                 "evaluation.health_max_rows": 40}), out)
    assert (out / "test_report.json").is_file() and (out / "health_report.json").is_file()
    if mode == "scaled":
        assert (out / "battery_report.txt").is_file() and "battery_skipped" not in r.extra
    else:
        assert not (out / "battery_report.txt").exists() and "entry_range" in r.extra["battery_skipped"]
