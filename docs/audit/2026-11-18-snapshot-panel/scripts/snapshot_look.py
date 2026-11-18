"""CPU look (known answer first, PREREGISTERED.md): base panel model vs the snapshot model (joint / hybrid) on an
artifact (E4 or a real slice), whole timestamps, the same loss / selection / rows. One JSON line per arm x seed."""
import json
import sys
import time

import torch

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import ModelSettings, RunSettings, default_model_settings

ARMS = {
    "base": ("nig_timenet_v2", {"cross_input": "relative"}),
    "joint": ("snapshot_panel", {"shared_path": False}),
    "hybrid": ("snapshot_panel", {}),
}


def main(art, root, arm, seed, epochs=8, threads=2, batch=1600):
    torch.set_num_threads(int(threads))
    arch, over = ARMS[arm]
    rd = f"{root}/{arm}_s{seed}"
    s = updated(RunSettings(run_dir=rd), {
        "data.name": art, "train.device": "cpu", "train.epochs": int(epochs), "train.seed": int(seed),
        "train.batch_size": int(batch), "train.lr": 1e-3, "train.early_stop.patience": 3,
        "split.cross_section": True, "split.coin_channel": arch == "snapshot_panel",
        "train.cross_section_loss": "rank_ic", "train.selection_metric": "val_rank_ic_reg",
        "evaluation.battery.enabled": False})
    base = default_model_settings()
    ms = ModelSettings(arch, {**base.overrides, **over}) if arch == "nig_timenet_v2" else ModelSettings(arch, over)
    s = updated(s, {"model": ms})
    run = steps.prepare(s, rd + "_out")
    steps.build(run)
    n_par = sum(p.numel() for p in run.trainer.model.parameters())
    t0 = time.perf_counter()
    steps.train(run)
    wall = time.perf_counter() - t0
    rep = steps.test_step(run)
    h = run.result.history
    out = {"arm": arm, "seed": int(seed), "params": n_par, "train_rows": len(run.x["train"]), "epochs": len(h),
           "best_epoch": run.result.best_epoch, "wall_s": round(wall, 1),
           "val_rank_ic_reg": [round(r["val_selection_metric"], 4) for r in h],
           "train_raw_loss": [round(r["train_raw_loss"], 3) for r in h],
           "val_raw_loss": [round(r["val_raw_loss"], 3) for r in h],
           "test_rank_ic": {t: round(rep["targets"][t]["skill"]["rank_ic_by_timestamp"], 4)
                            for t in ("high", "low", "close")},
           "test_coverage80": {t: round(rep["targets"][t]["sharp"]["coverage"], 3) for t in ("high", "low", "close")}}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
