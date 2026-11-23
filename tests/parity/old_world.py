"""The synthetic world of the parity test in the OLD project's array layout (not a test).

The same arrays feed both projects: ``tests/eval/verification_world.make_world`` (this project's datasets / tables) is
written as the old splits (train / val / test dicts with ``y_*_class`` coded +1 / -1 as the old pipeline did) and the
old model's outputs (the PredictionTable columns of val and test)."""

import json

import numpy as np

from nig_timenet_torch.model.heads import head_spec
from tests.eval.verification_world import SCALE, TARGETS, make_world

#: the two cases of the parity test: golden folder -> base timeframe (daily bars: a day is one timestamp; 6 h bars:
#: four timestamps per UTC day, where the old unit (timestamp) and the torch default (day) differ)
CASES = {"daily": "1D", "sixhour": "6h"}
#: the keyword arguments both projects get (old ``run_full_verification`` has no min_n: it uses 200 inside)
OLD_ARGS = {"target_acc": 0.65, "rr": 2.0, "stop_mode": "nig", "n_boot": 100, "run_candle": True}
MIN_N = 200


def world_arrays(tf: str = "1D") -> dict:
    ds, idx, tables = make_world(kind="skilled", bar=tf)
    out = {"tf": np.array(tf), "args": np.array(json.dumps(OLD_ARGS)), "horizon": np.array(int(ds["forecast_horizon"])),
           "scale": np.array(SCALE)}
    for stage in ("train", "val", "test"):
        rows = np.asarray(idx[stage])
        out[f"{stage}_X"] = ds[f"X_{tf}"][rows]
        out[f"{stage}_lc"] = ds["last_candles"][rows]
        for t in TARGETS:
            out[f"{stage}_reg_{t}"] = ds[head_spec("nig_regression", t).label_key][rows]
            out[f"{stage}_cls_{t}"] = 2.0 * ds[head_spec("binary_classification", t).label_key][rows] - 1.0
    for stage in ("val", "test"):
        for t in TARGETS:
            for spec in (head_spec("nig_regression", t), head_spec("binary_classification", t)):
                for key in spec.output_keys.values():
                    if key in tables[stage].columns:
                        out[f"{stage}_out_{key}"] = tables[stage][key].to_numpy(dtype=np.float64)
    return out
