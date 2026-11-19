import json, sys
from pathlib import Path
import torch
R = Path("/content/drive/MyDrive/nig_timenet_runs")
ck = next((R / "panel-rolling-ref-002/train").glob("fold_000_seed_0/ckpt/checkpoint.pt"))
h = torch.load(ck, map_location="cpu", weights_only=False)["history"]
print([k for k in h[0] if not k.startswith("trace") and ("val" in k or k.startswith("train_loss") or k == "loss" or "total" in k)][:40])
out = {}
for job in ("panel-rolling-ref-002", "panel-rolling-std-001", "panel-rolling-ls-001", "panel-rolling-base-001"):
    for c in sorted((R / job / "train").glob("fold_*_seed_*/ckpt/checkpoint.pt")):
        hist = torch.load(c, map_location="cpu", weights_only=False)["history"]
        keep = [k for k in hist[0] if k in ("train_total_loss", "val_total_loss", "val_raw_loss", "val_selection_metric", "train_cross_section_loss", "val_cross_section_loss", "train_close_reg_loss", "val_close_reg_loss",
                "train_raw_loss", "train_close_reg_loss", "val_close_reg_loss", "val_rank_ic_reg", "val_ls_sharpe_close",
                "train_cross_section_loss", "val_cross_section_loss")]
        out[f"{job}/{c.parent.parent.name}"] = [{k: h_.get(k) for k in keep} for h_ in hist]
json.dump(out, open(R / "memorisation-check-001/curves.json", "w"))
print(len(out))
