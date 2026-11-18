"""Runner of the batch_by walk-forward (docs/audit/2026-11-17-batch-by/PREREGISTERED.md): one arm per call, so the
arms run as separate processes (one CPU thread each). Writes ``<out>/<arm>/`` (the rolling outputs of every fold x
seed) and ``<out>/<arm>/test_loss.jsonl`` (the trainer's raw loss and per-target losses on each fold's test rows).

    python docs/audit/2026-11-17-batch-by/run_arms.py ARTIFACT OUT ARM [--epochs 4] [--max-folds 4] [--seeds 0,1]
        [--device cpu|cuda]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow.rolling_run import run_rolling
from nig_timenet_torch.workflow.settings import (
    BatterySettings,
    EvalSettings,
    RollingSettings,
    RunSettings,
    TrainSettings,
)

ARMS = {"A0": {"batch_by": "rows", "batch_size": 256}, "A1": {"batch_by": "day"},
        "A2": {"batch_by": "timestamp"}, "C60": {"batch_by": "rows", "batch_size": 60},
        "C20": {"batch_by": "rows", "batch_size": 20},
        "C34": {"batch_by": "rows", "batch_size": 34}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("out")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-folds", type=int, default=4)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--device", default="cpu", help="cpu (one thread) or cuda")
    ap.add_argument("--name", default="", help="output folder name (default: the arm), e.g. A2_s0 for one seed")
    a = ap.parse_args(argv)
    if a.device == "cpu":
        torch.set_num_threads(1)
    out = Path(a.out) / (a.name or a.arm)
    out.mkdir(parents=True, exist_ok=True)
    train = TrainSettings(device=a.device, epochs=a.epochs, **ARMS[a.arm])
    settings = RunSettings(run_dir=str(out / "unused"), train=train,
                           data=LoaderSettings(name=a.artifact, mmap=True, local_dir=""),
                           evaluation=EvalSettings(battery=BatterySettings(enabled=False)))
    rolling = RollingSettings(mode="expanding", train_span="365D", test_span="90D", step="90D", embargo="1D",
                              max_folds=a.max_folds, seeds=tuple(int(s) for s in a.seeds.split(",")))
    log = out / "test_loss.jsonl"
    t0 = time.monotonic()

    def on_fold(run, row):
        logs = run.trainer.evaluate(run.x["test"], run.y["test"], 4096)
        rec = {"arm": a.arm, "fold": row["fold"], "seed": row["seed"], "seconds": time.monotonic() - t0,
               "best_epoch": run.result.best_epoch, "epochs_done": run.result.epochs_done,
               "train_rows": int(len(run.idx["train"])), "test_rows": int(len(run.idx["test"])),
               **{f"test_{k}": float(v) for k, v in logs.items()}}
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{a.arm} fold {row['fold']} seed {row['seed']} done at {rec['seconds']:.0f} s", flush=True)

    run_rolling(settings, rolling, out, on_fold=on_fold)
    print(f"{a.arm} finished in {time.monotonic() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
