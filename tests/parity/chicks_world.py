"""The small synthetic world of the chicks parity test, in this project's layout and in the OLD project's (not a test).

One world with a KNOWN answer (``tests.eval.verification_world`` helpers: a dataset with ``last_candles`` and
return-coded regression labels, the PredictionTables of a "skilled" model) cut into train / val / test by day. The
same arrays feed both projects: ``small_world`` is what this project's ``stage_from_dataset`` reads, ``old_arrays``
is the old project's ``test_dict`` / ``val_dict`` layout (one split per coin) and the fake model's outputs, which
``tests/parity/run_old_chicks.py`` replays through the old ``run_full_analysis``."""

import json

import numpy as np

from nig_timenet_torch.model.heads import head_spec
from tests.eval.verification_world import SCALE, TARGETS, make_dataset, make_table

#: the cases of the parity test: golden folder -> (bar timeframe, days, coins, the chicks arguments both projects get)
CASES = {
    "daily_default": ("1D", 160, 4, {}),
    "sixhour_platt_neutral": ("6h", 160, 4, {"calibration_method": "platt", "pnl_market_neutral": True,
                                              "trade_min_edge_ratio": 0.5, "trade_sort_by": "confidence",
                                              "trade_top_n": 5, "batch_analysis_size": 5}),
}


def small_world(tf: str, n_days: int, n_coins: int):
    """-> (dataset, idx, tables) with day-based splits: train < 50 %, val 50-75 % (a gap), test from 82 %."""
    ds, _ = make_dataset(0, n_days=n_days, n_coins=n_coins, bar=tf)
    day = np.arange(len(ds["last_candles"])) // n_coins
    idx = {"train": np.flatnonzero(day < n_days // 2), "val": np.flatnonzero((day >= n_days * 5 // 8)
                                                                               & (day < n_days * 3 // 4)),
           "test": np.flatnonzero(day >= n_days * 41 // 50)}
    maj = {t: bool(np.mean(ds[head_spec("binary_classification", t).label_key][idx["train"]] > 0) >= 0.5)
           for t in TARGETS}
    tables = {s: make_table(ds, idx[s], kind="skilled", seed=1 + i, majority=maj)
              for i, s in enumerate(("val", "test"))}
    rng = np.random.default_rng(7)
    for table in tables.values():                  # a varying NIG scale: a constant one makes ranks of rounding noise
        for t in TARGETS:
            keys = head_spec("nig_regression", t).output_keys
            table[keys["beta"]] = table[keys["beta"]] * rng.uniform(0.5, 1.5, len(table))
            table[keys["nu"]] = table[keys["nu"]] * rng.uniform(0.8, 1.6, len(table))
    return ds, idx, tables


def old_arrays(tf: str, n_days: int, n_coins: int, args: dict) -> dict:
    """The world as the old project holds it: per stage ``X`` / ``lc`` / ``reg_<t>`` / ``asset`` and the model's outputs
    ``out_<key>`` (all in table row order), plus the scale, the timeframe and the arguments."""
    ds, idx, tables = small_world(tf, n_days, n_coins)
    out = {"tf": np.array(tf), "scale": np.array(SCALE), "args": np.array(json.dumps(args)),
           "horizon": np.array(int(ds["forecast_horizon"]))}
    for stage in ("val", "test"):
        rows = np.asarray(idx[stage])
        out[f"{stage}_X"] = ds[f"X_{tf}"][rows]
        out[f"{stage}_lc"] = ds["last_candles"][rows]
        out[f"{stage}_asset"] = np.asarray(tables[stage]["asset"].astype(str).tolist(), dtype="U16")
        for t in TARGETS:
            out[f"{stage}_reg_{t}"] = ds[head_spec("nig_regression", t).label_key][rows]
            for spec in (head_spec("nig_regression", t), head_spec("binary_classification", t)):
                for key in spec.output_keys.values():
                    if key in tables[stage].columns:
                        out[f"{stage}_out_{key}"] = tables[stage][key].to_numpy(dtype=np.float64)
    return out
