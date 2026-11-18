"""Speed arms on CPU. Pre-registered (2026-11-18, before any number): a fast arm passes when its wall time per epoch is
<= half the base arm's AND, at the same epoch count, its val rank IC (within timestamp) of every head is >= the base
arm's minus max(0.01, the base seed spread) and its val NIG NLL (raw loss) is not worse by more than 0.02."""
import sys, json, time, numpy as np, torch
from nig_timenet_torch.core.config import updated
from nig_timenet_torch.workflow.settings import RunSettings
from nig_timenet_torch.workflow import run as steps, flat_run
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.metrics import rank_ic

art, root, arm, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 2)
LITE = {"d_model": 32, "num_layers": 1, "head_hidden": 32, "class_head_hidden": 16}
ARMS = {
    "base": ({}, {}),
    "bs1024": ({"train.batch_size": 1024, "train.lr": 2e-3}, {}),
    "bs4096": ({"train.batch_size": 4096, "train.lr": 4e-3}, {}),
    "lite": ({}, LITE),
    "lite_bs1024": ({"train.batch_size": 1024, "train.lr": 2e-3}, LITE),
    "lite_bs4096": ({"train.batch_size": 4096, "train.lr": 4e-3}, LITE),
    # batch composition look (owner 2026-11-18). Pre-registered before any number: time_blocks (one moment of all
    # coins per batch) is kept as a candidate only if its VAL close rank IC beats lite (rows) by more than
    # max(0.01, the lite seed spread) on BOTH seeds and high / low lose no more than 0.01; same rule for +rank_ic.
    "lite_tb": ({"train.batch_order": "time_blocks"}, LITE),
    "lite_tb_rank": ({"train.batch_order": "time_blocks", "train.cross_section_loss": "rank_ic"}, LITE),
}
chg, mo = ARMS[arm]
rd = f"{root}/{arm}_s{seed}"
s = updated(RunSettings(run_dir=rd), {"data.name": art, "train.device": "cpu", "train.epochs": epochs,
                                      "train.seed": seed, "train.early_stop.patience": 99,
                                      "evaluation.battery.enabled": False, **chg})
if mo:
    s = updated(s, {"model.overrides": {**s.model.overrides, **mo}})
run = steps.prepare(s, rd + "_out")
steps.build(run)
n_par = sum(p.numel() for p in run.trainer.model.parameters())
t0 = time.perf_counter()
steps.train(run)
wall = time.perf_counter() - t0
hist = run.result.history
m = run.trainer.model.eval()
x = run.x["val"]
with torch.no_grad():
    parts = [m(xb) for xb in x.split(8192)]
o = {k: torch.cat([p[k] for p in parts]).squeeze(-1).numpy() for k in parts[0]}
tid = flat_run.time_ids(run.art.dataset, run.idx["val"])
ic = {t: rank_ic(run.y["val"][f"y_{t}_reg"].numpy(), o[head_spec("nig_regression", t).output_keys["mu"]], tid)
      for t in ("high", "low", "close")}
res = {"arm": arm, "seed": seed, "params": n_par, "train_rows": len(run.x["train"]), "epochs": len(hist),
       "wall_s": round(wall, 1), "epoch_s": [round(h.get("trace_epoch_seconds", float("nan")), 1) for h in hist],
       "val_raw_loss": [round(h["val_raw_loss"], 4) for h in hist], "best_epoch": run.result.best_epoch,
       "val_rank_ic_best_weights": {k: round(float(v), 4) for k, v in ic.items()}}
print(json.dumps(res))
