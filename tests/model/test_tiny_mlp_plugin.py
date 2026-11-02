"""P33: a trivial second architecture plugs in by matching the contract, then trains with the same Trainer and is
evaluated by the same predict_table / metrics path. Wiring smoke test only, no quality claim."""

import math

import numpy as np
import torch

from nig_timenet_torch.core.model_contract import check_contract, contract_of
from nig_timenet_torch.core.prediction_schema import validate_table
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.model.architectures import ARCHITECTURES, ModelSettings, build_model
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.trainer import Trainer
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from tests.contract_doc import DATASET_DIR
from tests.trainer.fixture_setup import load_fixture, trainer_config


def test_tiny_mlp_matches_contract():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 18)
    reference = contract_of(build_model(ModelSettings(), 16, 18), x)
    check_contract(build_model(ModelSettings(architecture="tiny_mlp"), 16, 18), reference, x)
    assert "tiny_mlp" in ARCHITECTURES and ARCHITECTURES.names()[0] == "nig_timenet_v2"


def test_tiny_mlp_one_epoch_smoke():
    seed_everything(0)
    x, y = load_fixture()
    model = build_model(ModelSettings(architecture="tiny_mlp"), x.shape[1], x.shape[2])
    trainer = Trainer(model, trainer_config(), optimizer=lambda ps: torch.optim.AdamW(ps, lr=1e-3))
    losses = []
    for idx in trainer.batch_indices(x.shape[0], 64):
        losses.append(trainer.train_step(x[idx], {k: v[idx] for k, v in y.items()})["total_loss"])
    assert losses and all(math.isfinite(v) for v in losses)

    d = npy_dir.load(DATASET_DIR)
    idx = np.arange(len(d["X_1h"]))
    identity = RunIdentity(schema_version=1, settings_hash="0" * 64, seed=0, data_fingerprint="fixture",
                           code_sha="0" * 40)
    df = predict_table(model, x, table_meta(d, idx, stage="report", fold=0, train_end=d["split_dates"]["train_end"]),
                       identity, batch_size=256)
    validate_table(df)
    assert math.isfinite(metrics.auc(d["y_close_class"], df["y_close_class_logits"].to_numpy()))
