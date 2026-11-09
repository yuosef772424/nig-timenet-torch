"""PURPOSE: Runs the deep-side check (eval/deep_side.py) on the VAL and TEST stages of a finished run (model holding
    its best weights, e.g. after workflow.run.restore) and writes deep_side_report.txt and deep_side.json.
TAGS: deep side, run reading, derived target, val and test, report writer
PITFALLS: needs the high, low and close heads and X channels high / low / close as log(price / prev close) (the
    last-bar asymmetry baseline is read from them); predicts each stage once (cached in run.extra by stage_inputs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nig_timenet_torch.eval import deep_side as ds
from nig_timenet_torch.workflow.reading import stage_data

__all__ = ["deep_frames", "deep_side_step", "REPORT", "JSON"]

REPORT, JSON = "deep_side_report.txt", "deep_side.json"


def deep_frames(run) -> dict:
    """{"VAL": frame, "TEST": frame}: ``eval.deep_side.deep_frame`` of each stage; ``last_bar_deep`` = log(H / C) +
    log(L / C) of the last input bar = x_high + x_low - 2 x_close."""
    out = {}
    for split, stage in (("VAL", "val"), ("TEST", "test")):
        st = stage_data(run, stage)
        x = st["x_last"]
        last = np.asarray(x["high"], np.float64) + np.asarray(x["low"], np.float64) - 2.0 * np.asarray(x["close"],
                                                                                                         np.float64)
        out[split] = ds.deep_frame(st["battery"], last)
    return out


def deep_side_step(run, out_dir, seed: int = 0) -> dict:
    """The deep-side check of ``run`` into ``out_dir``; returns the result dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = deep_frames(run)
    res = ds.deep_side(f["VAL"], f["TEST"], seed=seed)
    (out / REPORT).write_text(ds.render(res) + "\n", encoding="utf-8")
    (out / JSON).write_text(json.dumps(res, indent=1, default=_json), encoding="utf-8")
    return res


def _json(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.generic):
        return v.item()
    return str(v)
