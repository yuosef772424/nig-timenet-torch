"""PURPOSE: The only bridge from this project's outputs to the backtest contract: a battery table (mu, sigma, p_dir,
    y_ret, coin, end_time, optional predicted extremes) becomes a signal table; any other model writes the contract.
TAGS: backtest adapter, battery table, signal table, model outputs, stage column, predicted extremes
PITFALLS: ``end_time`` of a battery table is the prediction time (the close of the last input bar), which is what
    the contract calls ``time``; pred_high / pred_low of a battery table are already returns from the last close
    (eval/battery_adapter.py), clipped here to their side of 0; a table without them leaves the feature absent (a
    'pred' exit then raises). Confidence is not in a battery table (pass ``confidence`` yourself).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.backtest.contract import check_signals

__all__ = ["from_battery_table"]


def from_battery_table(table: pd.DataFrame, stage: str | None = None, confidence=None) -> pd.DataFrame:
    """A contract signal table from a battery table: score = mu, sigma, p_up = p_dir, y_true = y_ret, time = end_time,
    ``stage`` (a name for every row) and an optional ``confidence`` array."""
    out = pd.DataFrame({"coin": table["coin"].astype(str).to_numpy(), "time": table["end_time"].to_numpy(),
                        "score": table["mu"].to_numpy()})
    for src, dst in (("sigma", "sigma"), ("p_dir", "p_up"), ("y_ret", "y_true")):
        if src in table:
            out[dst] = table[src].to_numpy()
    if {"pred_high", "pred_low"} <= set(table.columns):
        out["pred_high"] = np.maximum(table["pred_high"].to_numpy(), 0.0)
        out["pred_low"] = np.minimum(table["pred_low"].to_numpy(), 0.0)
    if stage is not None:
        out["stage"] = stage
    if confidence is not None:
        out["confidence"] = np.asarray(confidence, dtype=float)
    return check_signals(out)
