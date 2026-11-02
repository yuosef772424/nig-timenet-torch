"""P28: end-to-end WIRING smoke test on the committed fixture: data_io (P06) -> SplitPlan (P25) -> oracle-config
model (P15, freshly initialised weights) -> predict_table (P27) -> ReturnCodec.decode (P26) -> stats.metrics (P21).

This asserts wiring, not model skill: no metric value or threshold is checked, only that the table validates and
every metric is a finite float. Internal consistency only; there is no TF oracle for this composition.
"""

import math

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.prediction_schema import validate_table
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.split_plan import purge_embargo_split, sample_times
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, own_anchor_and_future
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from tests.contract_doc import DATASET_DIR
from tests.oracle import oracle_loader as ol


def test_full_pipeline_runs_on_fixture():
    seed_everything(0)
    d = npy_dir.load(DATASET_DIR)
    fs, fe, le = sample_times(d)
    sd, bar = d["split_dates"], pd.Timedelta(d["base_tf"])
    plan = purge_embargo_split(le, fs, sd["train_end"], sd["train_end"], sd["val_end"], sd["val_end"],
                               (d["window_sizes"][d["base_tf"]] + d["forecast_horizon"]) * bar, feature_end=fe)

    kw = dict(ol.load_config()["constructor_kwargs"])
    kw.pop("seq_len"), kw.pop("n_features")
    seq_len, n_features = d["X_1h"].shape[1:]
    model = build_nig_timenet_v2(seq_len, n_features, **kw)  # fresh weights (not the oracle's)

    idx = plan.test_idx
    identity = RunIdentity(schema_version=1, settings_hash="0" * 64, seed=0, data_fingerprint="fixture",
                           code_sha="0" * 40)
    df = predict_table(model, torch.as_tensor(d["X_1h"][idx]),
                       table_meta(d, idx, stage="report", fold=0, train_end=sd["train_end"]), identity, batch_size=64)
    validate_table(df)
    assert len(df) == len(idx)

    codec = TARGET_CODECS["return"]()
    state = codec.fit(d)
    values = []
    for t in ("high", "low", "close"):
        raw = own_anchor_and_future(d["last_candles"][idx], t)
        price = codec.decode(df[f"y_{t}"].to_numpy(), state, anchors=raw[:, 0])
        values += [metrics.mae(raw[:, 1], price),
                   metrics.auc(d[f"y_{t}_class"][idx], df[f"y_{t}_class_logits"].to_numpy()),
                   metrics.ece(d[f"y_{t}_class"][idx], df[f"y_{t}_class_logits"].to_numpy())]
    values.append(metrics.rank_ic(d["y_close_reg"][idx], df["y_close"].to_numpy(), groups=df["timestamp"].to_numpy()))
    assert all(isinstance(v, float) and math.isfinite(v) for v in values), values
    assert np.isfinite(df.select_dtypes("number").to_numpy()).all()
