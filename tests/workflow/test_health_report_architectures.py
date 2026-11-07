"""health_report runs end to end for EVERY architecture (hardening H3, HF-09): tiny_mlp and the GRU / TCN variants of
nig_timenet_v2 on the fixture, with the probe points and head modules taken from the model itself.

PURPOSE:  Proves the model_plugin_howto claim: a new architecture needs head_modules() and default_probe_points and
          then runs fit -> predict_table -> health_report unchanged.
TAGS:     health report, architectures, tiny mlp, gru, tcn, head modules, probe points, plugin
PITFALLS: Wiring only (the fixture has 6 synthetic coins): no number here says anything about skill.
"""

import json

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.nig_floor import nig_floors
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow.health_report import REPORT_JSON, HealthRun, health_report
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from tests.contract_doc import DATASET_DIR
from tests.trainer.fixture_setup import trainer_config
from tests.trainer.helpers import identity_for

N_TRAIN, N_VAL = 64, 40
SMALL = {"d_model": 8, "num_layers": 1, "num_heads": 2, "num_kv_heads": 1, "head_hidden": 8, "class_head_hidden": 6}
ARCHS = {"tiny_mlp": ("tiny_mlp", {"hidden": 8}), "gru": ("nig_timenet_v2", {**SMALL, "encoder": "gru"}),
         "tcn": ("nig_timenet_v2", {**SMALL, "encoder": "tcn"}), "transformer": ("nig_timenet_v2", SMALL)}


@pytest.mark.parametrize("name", sorted(ARCHS))
def test_health_report_runs_for_every_architecture(name, tmp_path):
    arch, over = ARCHS[name]
    d = npy_dir.load(DATASET_DIR)
    x = torch.as_tensor(d["X_1h"], dtype=torch.float32)
    y = {k: torch.as_tensor(d[k], dtype=torch.float32) for k in
         [f"y_{t}_{c}" for t in ("high", "low", "close") for c in ("reg", "class")]}
    tr, va = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_VAL)
    torch.manual_seed(0)
    model = build_model(ModelSettings(arch, over), x.shape[1], x.shape[2])
    assert set(nig_floors(model)) == {"high", "low", "close"}
    trainer = Trainer(model, trainer_config(), optimizer=lambda ps: torch.optim.AdamW(ps, lr=1e-3),
                      selection_metric="val_raw_loss")
    settings = FitSettings(epochs=1, batch_size=32, trace=True)
    ident = identity_for(settings)
    trainer.fit(x[tr], {k: v[tr] for k, v in y.items()}, x[va], {k: v[va] for k, v in y.items()}, settings,
                run_dir=tmp_path, identity=ident)
    idx = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    table = predict_table(trainer.model, x[va], table_meta(d, idx, stage="val", fold=0,
                          train_end=d["split_dates"]["train_end"]), ident)
    lc = np.asarray(d["last_candles"])
    run = HealthRun(run_dir=tmp_path, model=trainer.model, x=x[va], predictions=table,
                    labels={k: v[va].numpy() for k, v in y.items()},
                    anchors={t: lc[idx, {"high": 0, "low": 1, "close": 2}[t]] for t in ("high", "low", "close")},
                    last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]],
                    codec_state=TARGET_CODECS["return"]().fit(d),
                    probe_points=tuple(trainer.model.default_probe_points), probe_labels=("y_close_class",))
    health_report(run)
    data = json.loads((tmp_path / REPORT_JSON).read_text(encoding="utf-8"))
    assert set(data["nig_floor"]) == {"high", "low", "close"}
    points = [r["probe_point"] for r in data["probe"]["y_close_class"]]
    assert points == ["raw_input", *trainer.model.default_probe_points]
