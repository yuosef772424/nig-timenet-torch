"""Shared fixture-model setup for the trainer tests: tests/fixtures/tiny_dataset.dataset (via data_io, P06), a
small NIG-TimeNet v2 with NIG + classification heads on all three price targets, the oracle's trainer config."""

import torch

from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from tests.contract_doc import DATASET_DIR
from tests.oracle import oracle_loader as ol

TARGETS = ("high", "low", "close")
HEAD_TYPES = {t: ["nig_regression", "binary_classification"] for t in TARGETS}


def load_fixture(n: int | None = None):
    d = npy_dir.load(DATASET_DIR)
    x = torch.as_tensor(d["X_1h"][:n], dtype=torch.float32)
    y = {f"y_{t}_{k}": torch.as_tensor(d[f"y_{t}_{k}"][:n], dtype=torch.float32) for t in TARGETS
         for k in ("reg", "class")}
    return x, y


def build_model(seq_len: int, n_features: int, dropout: float = 0.0):
    return build_nig_timenet_v2(seq_len, n_features, d_model=16, num_layers=1, num_heads=4, num_kv_heads=2,
                                head_hidden=16, class_head_hidden=8, dropout=dropout, head_types=HEAD_TYPES)


def trainer_config() -> dict:
    return ol.load_config()["trainer"]
