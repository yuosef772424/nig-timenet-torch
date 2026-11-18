"""Memorisation in the walk-forward (owner 2026-11-19): per-fold learning curves from the checkpoints, the year
composition and coin set of each fold's TRAIN rows (the row selection of the jobs), coin overlap between folds."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.jobs import rolling as rj
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.rolling import fold_with_val, rolling_folds

R = Path("/content/drive/MyDrive/nig_timenet_runs")
OUT = R / "memorisation-check-001"
OUT.mkdir(exist_ok=True)
res = {"curves": {}, "folds": []}

for job in ("panel-rolling-ref-002", "panel-rolling-std-001", "panel-rolling-ls-001", "panel-rolling-base-001"):
    for ck in sorted((R / job / "train").glob("fold_*_seed_*/ckpt/checkpoint.pt")):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        hist = st.get("history") or []
        rows = [{k: h.get(k) for k in ("epoch", "train_loss", "val_loss", "val_selection", "val_raw_loss")
                 if k in h} for h in hist]
        meta = json.loads((ck.parent / "best_meta.json").read_text()) if (ck.parent / "best_meta.json").is_file() else {}
        res["curves"][f"{job}/{ck.parent.parent.name}"] = {"history": rows, "best": meta,
                                                         "keys": sorted(hist[0].keys()) if hist else []}
print("curves", len(res["curves"]), flush=True)

cfg = json.load(open("/content/nig-timenet-torch/docs/jobs/panel-rolling-ref-002.json"))["config"]
cfg["artifact"] = "/content/art1"
c = from_dict(rj.RollingConfig, cfg)
s = rj.run_settings(c, code_sha="x" * 40)
r = rj.rolling_settings(c)
art = run_common.load_data(s)
ds = art.dataset
ts = pd.to_datetime(np.asarray(ds["last_candles"])[:, TS_COL].astype("int64"), utc=True)
names = np.array([b["name"] for b in ds["asset_bounds"]])
coin_of = np.empty(len(ts), dtype=object)
for b in ds["asset_bounds"]:
    coin_of[b["start"]:b["end"]] = b["name"]
first_seen = pd.Series(ts).groupby(coin_of).min()
sets = []
for fold in rolling_folds(ds, r):
    plan, cut_v = fold_with_val(fold, r.embargo, r.val_share)
    tr, va, te = run_common.select_rows(ds, s.split, (plan.train_idx, plan.val_idx, plan.test_idx))
    yrs = pd.Series(ts[tr].year).value_counts(normalize=True).sort_index()
    coins = set(coin_of[tr])
    sets.append(coins)
    listed_2020 = sum(first_seen[c_] < pd.Timestamp("2021-01-01", tz="UTC") for c_ in coins)
    first_era = (ts[tr] < pd.Timestamp("2021-07-01", tz="UTC")).mean()
    # how many times one calendar day of the first era is present across the training sets so far
    res["folds"].append({"fold": fold.index, "train_rows": int(len(tr)), "val_rows": int(len(va)),
                         "test_rows": int(len(te)), "train_start": str(ts[tr].min()), "train_end": str(ts[tr].max()),
                         "year_share": {int(k): round(float(v), 3) for k, v in yrs.items()},
                         "share_rows_before_2021_07": round(float(first_era), 3), "coins": len(coins),
                         "coins_listed_before_2021": int(listed_2020),
                         "timestamps": int(len(np.unique(ts[tr]))),
                         "rows_per_timestamp": round(len(tr) / max(1, len(np.unique(ts[tr]))), 1),
                         "test_coins_in_train": round(len(set(coin_of[te]) & coins) / max(1, len(set(coin_of[te]))), 3)})
    print(res["folds"][-1], flush=True)
res["coin_jaccard_vs_fold0"] = [round(len(a & sets[0]) / len(a | sets[0]), 3) for a in sets]
(OUT / "memorisation.json").write_text(json.dumps(res, indent=1, default=str))
print("done", res["coin_jaccard_vs_fold0"])
