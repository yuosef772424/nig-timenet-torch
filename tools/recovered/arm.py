# Recovered from a past session (scratch file, not a repo module). Ran on the sandbox CPU: one arm of the panel study on a local artifact (E3p slice or E4).
# Original name: scratchpad/arm.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
"""Scratch: one panel arm on CPU. python arm.py ARTIFACT OUT --name B --model '{}' --train '{}' --seed 0"""
import argparse, json, pathlib, time
import torch
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.flat_run import epoch_line
from nig_timenet_torch.workflow.panel_report import panel_step, oracle_frame, stage_frame, panel_report, headline
from nig_timenet_torch.workflow.settings import (BatterySettings, EvalSettings, ModelSettings, RunSettings, SplitSettings,
                                                 TrainSettings, default_model_settings)
p = argparse.ArgumentParser()
p.add_argument("artifact"); p.add_argument("out"); p.add_argument("--name", default="B")
p.add_argument("--model", default="{}"); p.add_argument("--train", default="{}"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch", type=int, default=256); p.add_argument("--selection", default="val_rank_ic_reg")
p.add_argument("--threads", type=int, default=2); p.add_argument("--oracle", action="store_true")
p.add_argument("--patience", type=int, default=4)
a = p.parse_args()
torch.set_num_threads(a.threads)
base = default_model_settings()
out = pathlib.Path(a.out) / f"{a.name}_s{a.seed}"
s = RunSettings(run_dir=str(out / "ckpt"), data=LoaderSettings(name=a.artifact, local_dir=""),
                split=SplitSettings(cross_section=True),
                model=ModelSettings(base.architecture, {**base.overrides, "encoder": "gru", **json.loads(a.model)}),
                train=TrainSettings(epochs=a.epochs, batch_size=a.batch, seed=a.seed, lr=a.lr, device="cpu",
                                    early_stop=EarlyStopSettings(patience=a.patience), selection_metric=a.selection,
                                    **json.loads(a.train)),
                evaluation=EvalSettings(health_max_rows=2000, battery=BatterySettings(enabled=False)))
t0 = time.time()
r = steps.build(steps.prepare(s, out))
steps.train(r, on_epoch=lambda row: print(epoch_line(row, a.epochs), flush=True))
rep = steps.test_step(r)
pr = panel_step(r)
h = headline(pr)
row = {"arm": a.name, "seed": a.seed, "seconds": round(time.time() - t0, 1), "best_epoch": r.result.best_epoch,
       "epochs_done": r.result.epochs_done, "ic_close": rep["targets"]["close"]["skill"]["rank_ic_by_timestamp"],
       "ic_high": rep["targets"]["high"]["skill"]["rank_ic_by_timestamp"], "ic_low": rep["targets"]["low"]["skill"]["rank_ic_by_timestamp"], **h}
(out / "row.json").write_text(json.dumps(row, indent=1, default=float))
print(json.dumps(row, default=float))
if a.oracle:
    o = panel_report(oracle_frame(r, "val"), oracle_frame(r, "test"))
    (out.parent / "oracle.json").write_text(json.dumps(o, indent=1, default=float))
    print("ORACLE", json.dumps(headline(o), default=float))
import resource
print("PEAK_RSS_MB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0), "epoch_s", [round(h["trace_epoch_seconds"], 1) for h in r.result.history])
