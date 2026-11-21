"""P27: PredictionTable schema, O(1)-in-assets batched predict, leakage guard, write/read round trip."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.core.prediction_schema import PREDICTION_TABLE_COLUMNS, PredictionTableError, validate_table
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.prediction_table import predict_table, read_table, table_meta, write_table
from tests.contract_doc import DATASET_DIR
from tests.data_io.helpers import fixture_plan
from tests.trainer.fixture_setup import build_model

IDENTITY = RunIdentity(schema_version=1, settings_hash="s" * 64, seed=0, data_fingerprint="d" * 64, code_sha="c" * 40)


@pytest.fixture(scope="module")
def data():
    return npy_dir.load(DATASET_DIR)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(16, 18)


def table_for(model, data, idx, stage="report", batch_size=64):
    x = torch.as_tensor(data["X_1h"][idx], dtype=torch.float32)
    meta = table_meta(data, idx, stage=stage, fold=0, train_end=data["split_dates"]["train_end"])
    return predict_table(model, x, meta, IDENTITY, batch_size=batch_size)


def test_predict_table_schema_valid(model, data):
    df = table_for(model, data, np.arange(100))
    validate_table(df)
    assert tuple(df.columns[:len(PREDICTION_TABLE_COLUMNS)]) == PREDICTION_TABLE_COLUMNS
    assert {"y_close", "y_close_class_logits", "y_high_confidence"} <= set(df.columns)
    with pytest.raises(PredictionTableError, match=r"missing columns: \['fold'\].*extra columns: \['bogus'\]"):
        validate_table(df.drop(columns=["fold"]).assign(bogus=1))
    with pytest.raises(PredictionTableError, match="horizon_ns"):
        validate_table(df.assign(horizon_ns=df["horizon_ns"].astype(float)))


class CallCounter(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.calls = inner, 0

    def forward(self, x):
        self.calls += 1
        return self.inner(x)


@pytest.mark.parametrize("batch_size", [32, 512])
def test_predict_table_is_batched_not_per_asset(model, data, batch_size):
    """Same 200 rows labelled as 5 assets and as 50 assets: identical, O(1)-in-assets call counts."""
    idx = np.arange(200)
    calls = []
    for n_assets in (5, 50):
        counter = CallCounter(model)
        meta = table_meta(data, idx, stage="report", fold=0, train_end=data["split_dates"]["train_end"])
        meta["asset"] = np.array([f"A{i % n_assets}" for i in range(len(idx))], dtype=object)
        df = predict_table(counter, torch.as_tensor(data["X_1h"][idx]), meta, IDENTITY, batch_size=batch_size)
        assert df["asset"].nunique() == n_assets
        calls.append(counter.calls)
    assert calls[0] == calls[1] <= math.ceil(len(idx) / batch_size) + 1


def test_no_row_with_label_available_at_gt_train_end_in_train_fold(model, data):
    plan = fixture_plan()
    df = table_for(model, data, plan.train_idx, stage="fit")
    assert len(df) == len(plan.train_idx)
    assert (df["label_available_at"] <= df["train_end"]).all()
    assert (df["label_available_at"] >= df["timestamp"]).all()
    for _, g in df.groupby("asset"):
        assert g["label_available_at"].is_monotonic_increasing


def test_write_read_round_trip(model, data, tmp_path):
    df = table_for(model, data, np.arange(50))
    path = tmp_path / "preds.parquet"
    write_table(path, df, IDENTITY)
    back = read_table(path)
    pd.testing.assert_frame_equal(back, df)
    other = RunIdentity(schema_version=1, settings_hash="x" * 64, seed=0, data_fingerprint="d", code_sha="c")
    with pytest.raises(ValueError, match="identity_hash"):
        write_table(tmp_path / "bad.parquet", df, other)


class GroupCentred(torch.nn.Module):
    """A cross-section stand-in: each row's output is its first channel minus the mean of that channel over the rows
    of ITS group in the same batch (the group id is the last channel, as in a split.cross_section run)."""

    def forward(self, x):
        v, gid = x[:, -1, 0], x[:, -1, -1].round().long()
        uniq, inv = torch.unique(gid, return_inverse=True)
        mean = torch.zeros(len(uniq)).index_add_(0, inv, v) / torch.bincount(inv).float()
        return {"y_close": (v - mean[inv]).unsqueeze(-1)}


def test_group_batches_never_split_a_group_so_cross_section_outputs_do_not_depend_on_the_batch_size(data):
    """C-01: predict_stage cut whole timestamps, then predict_table re-sliced them into fixed 512-row batches, so a
    cross-section model saw part of a timestamp on the rows at each batch border."""
    idx = np.arange(98)
    x = torch.as_tensor(data["X_1h"][idx], dtype=torch.float32)
    gid = torch.arange(len(idx)) // 7                          # 14 groups of 7 rows, contiguous
    x = torch.cat([x, gid.view(-1, 1, 1).expand(-1, x.shape[1], 1).float()], dim=-1)
    meta = table_meta(data, idx, stage="report", fold=0, train_end=data["split_dates"]["train_end"])
    whole = predict_table(GroupCentred(), x, meta, IDENTITY, batch_size=len(idx))["y_close"].to_numpy()
    grouped = predict_table(GroupCentred(), x, meta, IDENTITY, batch_size=16, groups=gid)["y_close"].to_numpy()
    fixed = predict_table(GroupCentred(), x, meta, IDENTITY, batch_size=16)["y_close"].to_numpy()
    np.testing.assert_allclose(grouped, whole, atol=1e-6)
    assert not np.allclose(fixed, whole, atol=1e-6)            # the old slicing cut groups 2, 4, ... at 16-row borders
