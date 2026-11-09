"""workflow/deep_side_run.py on a tiny E1 synthetic artifact (channels high / low / close, logret_close labels): a
trained run gives the VAL / TEST deep frames and both reports."""

import json

import numpy as np

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX as COL
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.workflow import deep_side_run as dsr
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings, TrainSettings

SHA = "d" * 40


def test_deep_side_step_on_a_trained_synthetic_run(tmp_path):
    art = tmp_path / "E1"
    write_synthetic(art, SyntheticSettings(n_coins=12, n_bars=400), source_commit=SHA)
    s = RunSettings(run_dir=str(tmp_path / "run"), code_sha=SHA, data=LoaderSettings(name=str(art), local_dir=""),
                    model=ModelSettings(architecture="tiny_mlp"),
                    train=TrainSettings(epochs=1, batch_size=128, device="cpu"))
    r = steps.train(steps.build(steps.prepare(s)))
    frames = dsr.deep_frames(r)
    for split, stage in (("VAL", "val"), ("TEST", "test")):
        f = frames[split]
        assert len(f) == len(r.idx[stage])
        assert np.isfinite(f[["deep", "deep_pred", "last_bar_deep"]].to_numpy()).all()
        lc = np.asarray(r.dataset["last_candles"])[r.idx[stage]]
        last = lc[:, COL["last_close"]]
        real = np.log(lc[:, COL["future_high_max"]] / last) + np.log(lc[:, COL["future_low_min"]] / last)
        assert np.allclose(f["deep"], real, atol=1e-5)
    res = dsr.deep_side_step(r, tmp_path / "deep")
    assert set(res["skill"]) == {"VAL", "TEST"} and (tmp_path / "deep" / dsr.REPORT).is_file()
    assert "verdict" in json.loads((tmp_path / "deep" / dsr.JSON).read_text())
