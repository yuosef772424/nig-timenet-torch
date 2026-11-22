"""PURPOSE: The evaluation steps of a Run on a trained model: the health report on the validation rows and the test
    report on ALL test rows.
TAGS: health step, test step, stage report, target report, test report json, validation subsample
PITFALLS: the test split is read only by ``test_step``, never for training or selection; a cross-section model gets
    whole timestamps in the health subsample (C-01 / C-02); both steps need a trained (or restored) Run.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.groups import row_groups
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.health_report import REPORT_JSON, REPORT_MD, HealthRun, health_report
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["health_step", "stage_report", "test_step", "TEST_REPORT"]

TEST_REPORT = "test_report.json"


def health_step(run: Run) -> Run:
    """Health report on an evenly strided subsample of the validation rows (they must fit on the device); a
    cross-section model gets whole timestamps (C-01 / C-02: a cut timestamp changes its outputs); written to the run
    folder and copied to ``out``."""
    need(run, "result", "train")
    ds, dev = run.dataset, run.device
    val_idx, cap = run.idx["val"], run.settings.evaluation.health_max_rows
    grouped = run.trainer.group_channel
    pick = (run_common.stride_groups(row_groups(run.x["val"]).numpy(), cap) if grouped
            else run_common.stride_rows(np.arange(len(val_idx)), cap))
    idx = val_idx[pick]
    xs = run.x["val"][torch.as_tensor(pick)].to(dev)
    meta = table_meta(ds, idx, stage="val", fold=0, train_end=run.bounds[0])
    table = predict_table(run.trainer.model, xs, meta, run.ident, groups=row_groups(xs).cpu() if grouped else None)
    lc, codec = np.asarray(ds["last_candles"]), run_common.codec_state(ds)
    probe_target = "close" if "close" in run.targets else run.targets[0]
    labels = {k: v[torch.as_tensor(pick)].numpy() for k, v in run.y["val"].items()}
    hr = HealthRun(run_dir=Path(run.settings.run_dir), model=run.trainer.model, x=xs, predictions=table,
                   labels=labels, anchors=flat_run.anchors_by_name(lc, idx, run.targets, codec),
                   last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec,
                   probe_labels=tuple(head_spec(k, probe_target).label_key
                                      for k in ("nig_regression", "binary_classification")))
    health_report(hr)
    run.extra["health_rows"] = int(len(idx))
    if run.out is not None:
        for name in (REPORT_JSON, REPORT_MD):
            shutil.copyfile(Path(run.settings.run_dir) / name, run.out / name)
    return run


def stage_report(run: Run, stage: str, table: pd.DataFrame) -> dict:
    lc, codec = np.asarray(run.dataset["last_candles"]), run_common.codec_state(run.dataset)
    idx = run.idx[stage]
    return target_report(table, {k: v.numpy() for k, v in run.y[stage].items()},
                         anchors=flat_run.anchors_by_name(lc, idx, run.targets, codec),
                         last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec)


def test_step(run: Run) -> dict:
    """target_report on ALL test rows (D-005); writes ``test_report.json`` to ``out``. The test split is never used
    for training or selection. The test table is kept in ``run.extra['tables']`` for the battery."""
    if not len(run.idx["test"]):
        raise ValueError("the split has no test rows")
    table = run.table("test")
    run.extra.setdefault("tables", {})["test"] = table
    rep = stage_report(run, "test", table)
    if run.out is not None:
        run_common.write_json(run.out / TEST_REPORT, rep)
    return rep
