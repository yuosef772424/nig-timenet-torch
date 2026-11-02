"""P35: the probe report is a TOOL; these tests check wiring only, never where information is lost (the fixture is
6 synthetic coins, far too small to mean anything)."""

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.probe import layer_probe_report, ridge_fit_predict
from tests.contract_doc import DATASET_DIR
from tests.oracle import oracle_loader as ol

POINTS = ["raw_input", "instance_norm", "patch_embed", "block_1", "block_2", "readout_fc", "trunk_norm"]


def fixture_model_and_data():
    d = npy_dir.load(DATASET_DIR)
    kw = dict(ol.load_config()["constructor_kwargs"])
    kw.pop("seq_len"), kw.pop("n_features")
    torch.manual_seed(0)
    x = torch.as_tensor(d["X_1h"], dtype=torch.float32)
    return build_nig_timenet_v2(x.shape[1], x.shape[2], **kw), x, d


def test_probe_report_includes_raw_input_row():
    model, x, d = fixture_model_and_data()
    df = layer_probe_report(model, x[:200], d["y_close_class"][:200], probe_points=["trunk_norm"])
    assert list(df["probe_point"]) == ["raw_input", "trunk_norm"]
    assert df.loc[0, "n_features"] == x.shape[1] * x.shape[2]


@pytest.mark.parametrize("label", ["y_close_class", "y_close_reg"])
def test_probe_report_runs_on_fixture(label):
    model, x, d = fixture_model_and_data()
    df = layer_probe_report(model, x, d[label], probe_points=POINTS)
    assert isinstance(df, pd.DataFrame) and list(df["probe_point"]) == POINTS
    assert set(df["metric"]) == {"auc" if label.endswith("class") else "r2"}
    assert np.isfinite(df[["mean", "std"]].to_numpy()).all()


def test_unknown_probe_point_raises():
    model, x, d = fixture_model_and_data()
    with pytest.raises(KeyError, match="block_9"):
        layer_probe_report(model, x[:50], d["y_close_class"][:50], probe_points=["block_9"])


def test_ridge_closed_form_primal_equals_dual_and_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 5))
    y = a @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 0.3
    pred = ridge_fit_predict(a, y, a, ridge=1e-8)
    np.testing.assert_allclose(pred, y, rtol=1e-5, atol=1e-6)
    wide = rng.normal(size=(20, 50))
    yw = rng.normal(size=20)
    from numpy.linalg import solve
    mu, sd = wide.mean(0), wide.std(0)
    z = (wide - mu) / sd
    primal = z @ solve(z.T @ z + 10.0 * np.eye(50), z.T @ (yw - yw.mean())) + yw.mean()
    np.testing.assert_allclose(ridge_fit_predict(wide, yw, wide, 10.0), primal, rtol=1e-5, atol=1e-6)
