"""T4 health_report end to end on the fixture: fit (trace on, run_dir) -> predict_table on validation -> HealthRun ->
JSON + markdown in the run folder. Wiring and contract only (6 synthetic coins say nothing about skill); the numbers
themselves are tested with known answers in tests/eval."""

import json

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.trainer.trainer import FitSettings
from nig_timenet_torch.workflow.health_report import REPORT_JSON, REPORT_MD, HealthRun, health_report
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from tests.contract_doc import DATASET_DIR
from tests.trainer.test_fit import identity_for, make_trainer

N_TRAIN, N_VAL = 96, 60


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = npy_dir.load(DATASET_DIR)
    x = torch.as_tensor(d["X_1h"], dtype=torch.float32)
    idx = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    y = {k: torch.as_tensor(d[k], dtype=torch.float32) for k in
         [f"y_{t}_{c}" for t in ("high", "low", "close") for c in ("reg", "class")]}
    tr, va = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_VAL)
    run_dir = tmp_path_factory.mktemp("run")
    trainer = make_trainer()
    settings = FitSettings(epochs=2, batch_size=32, trace=True)
    ident = identity_for(settings)
    trainer.fit(x[tr], {k: v[tr] for k, v in y.items()}, x[va], {k: v[va] for k, v in y.items()}, settings,
                run_dir=run_dir, identity=ident)
    state = TARGET_CODECS["return"]().fit(d)
    table = predict_table(trainer.model, x[va], table_meta(d, idx, stage="val", fold=0,
                          train_end=d["split_dates"]["train_end"]), ident)
    anchors = {t: np.asarray(d["last_candles"])[idx, {"high": 0, "low": 1, "close": 2}[t]] for t in
               ("high", "low", "close")}
    labels = {k: v[va].numpy() for k, v in y.items()}
    return HealthRun(run_dir=run_dir, model=trainer.model, x=x[va], predictions=table, labels=labels,
                     anchors=anchors, last_close=np.asarray(d["last_candles"])[idx, LAST_COLUMN_INDEX["last_close"]],
                     codec_state=state, probe_points=("trunk_norm",),
                     probe_labels=("y_close_reg", "y_close_class")), run_dir


def test_health_report_writes_json_and_markdown(run):
    hrun, run_dir = run
    rep = health_report(hrun)
    data = json.loads((run_dir / REPORT_JSON).read_text(encoding="utf-8"))
    assert data == json.loads(json.dumps(rep))
    assert set(data) == {"run_dir", "targets", "nig_floor", "probe", "trace"}
    assert set(data["targets"]["targets"]) == {"high", "low", "close"}
    assert set(data["nig_floor"]["close"]) == {"nu", "alpha", "beta"}
    assert set(data["probe"]) == {"y_close_reg", "y_close_class"}
    first = data["probe"]["y_close_class"][0]
    assert first["probe_point"] == "raw_input" and first["delta_vs_raw"] == 0.0
    tr = data["trace"]
    assert tr["trace_enabled"] and tr["epochs_done"] == 2 and len(tr["series"]["val_selection_metric"]) == 2
    assert any(k.startswith("trace_gradshare_") for k in tr["series"])
    md = (run_dir / REPORT_MD).read_text(encoding="utf-8")
    assert md.startswith("# Health report") and "## Targets" in md and "## NIG floor-hit shares" in md
    assert "## Trainer trace" in md and len(md.splitlines()) < 80
    assert "std<move share" in md and "range-share" not in md
    assert md.count("info only (direction suspended)") == 2          # high and low AUC cells, not close
    assert "(informational, O-1)" in md                                # close-head T-sharp verdict
    sh = data["targets"]["targets"]["close"]["sharp"]
    assert "share_std_below_move" in sh and sh["blocking"] is False
    assert all("share_std_below_move" in data["targets"]["targets"][t]["sharp"] for t in ("high", "low"))


def test_missing_checkpoint_and_misaligned_rows_raise(run, tmp_path):
    import dataclasses

    hrun, _ = run
    with pytest.raises(FileNotFoundError):
        health_report(dataclasses.replace(hrun, run_dir=tmp_path))
    with pytest.raises(ValueError):
        health_report(dataclasses.replace(hrun, x=hrun.x[:-1]))
    with pytest.raises(KeyError):
        health_report(dataclasses.replace(hrun, probe_labels=("y_nope_reg",)))
