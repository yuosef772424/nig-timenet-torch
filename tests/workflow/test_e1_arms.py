"""workflow/e1_arms.py: the E1 arm runner (gate rows + mu calibration vs the E1 oracle + head readout gap)."""

import json

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.data_io.synthetic import SyntheticSettings, oracle_keys, write_synthetic
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.workflow import e1_arms

SHA = "e" * 40
ARM = {"architecture": "tiny_mlp", "train": {"batch_size": 128}, "model": {"nig_scale": 0.01}}


@pytest.fixture(scope="module")
def e1(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1arms")
    write_synthetic(d / "E1", SyntheticSettings(n_coins=16, n_bars=600), source_commit=SHA)
    return d


def test_mu_calibration_of_the_oracle_itself_is_perfect_and_of_a_shrunk_mu_is_not():
    rng = np.random.default_rng(0)
    ds = {oracle_keys(t)[0]: rng.normal(0.02, 0.01, 500) for t in ("high", "low")}
    idx = np.arange(500)
    key = {t: head_spec("nig_regression", t).output_keys["mu"] for t in ("high", "low")}
    perfect = e1_arms.mu_calibration(ds, idx, pd.DataFrame({key[t]: ds[oracle_keys(t)[0]] for t in key}), key)
    shrunk = e1_arms.mu_calibration(ds, idx, pd.DataFrame({key[t]: 0.5 * ds[oracle_keys(t)[0]] for t in key}), key)
    assert perfect["high"]["slope_oracle_on_mu"] == pytest.approx(1.0) and perfect["high"]["mean_ratio"] == 1.0
    assert shrunk["low"]["slope_oracle_on_mu"] == pytest.approx(2.0) and shrunk["low"]["mean_ratio"] == 0.5


def test_unknown_arm_key_raises(tmp_path):
    with pytest.raises(KeyError, match="unknown arm key"):
        e1_arms.arm_settings({"modle": {}}, tmp_path, tmp_path, 0, 1)


def test_arm_settings_merge_the_preset_unless_another_architecture(tmp_path):
    s = e1_arms.arm_settings({"model": {"dropout": 0.05}}, tmp_path, tmp_path, 3, 2)
    assert s.model.overrides["dropout"] == 0.05 and s.model.overrides["encoder"] == "gru"
    assert s.train.seed == 3 and s.train.epochs == 2 and s.train.batch_size == 512
    tiny = e1_arms.arm_settings(ARM, tmp_path, tmp_path, 0, 1)
    assert tiny.model.overrides == {"nig_scale": 0.01} and tiny.train.batch_size == 128


def test_cli_runs_an_arm_and_writes_rows_calibration_and_readout(e1, capsys):
    res = e1_arms.main([str(e1 / "out"), json.dumps({"tiny": ARM}), "0", str(e1 / "E1"), "--epochs", "1"])
    (r,) = res["tiny"]
    assert {"high", "low", "close"} == set(r["mu"]) == set(r["readout"])
    assert set(r["readout"]["high"]) == {"model", "readout"} and np.isfinite(r["readout"]["high"]["readout"]["corr"])
    assert any(x["gate"] == "A" for x in r["rows"]) and any(x["gate"] == "C" for x in r["rows"])
    assert (e1 / "out" / "res_tiny.json").is_file()
    assert capsys.readouterr().out.splitlines()[-1].startswith("tiny")
