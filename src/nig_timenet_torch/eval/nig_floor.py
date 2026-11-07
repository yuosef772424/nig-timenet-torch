"""T4 NIG floor-hit shares: how often each NIG parameter of each head sits on its floor (B-001 evidence).

nu / alpha / beta = softplus(raw) + floor, so a value within ``rel_tol * floor`` of the floor means softplus(raw) ~ 0:
the head is pinned and its gradient through that parameter is ~ 0. Floors are read from the model's own ``NIGHead``
modules (never re-typed here); the values come from a PredictionTable (columns of the head registry).

PURPOSE:  T4 NIG floor-hit shares: how often each NIG parameter of each head sits on its floor; floors are read from
          the model's own ``head_modules()`` (architecture-neutral).
TAGS:     nig floor, floor hit, t4 diagnostics, head modules, nu alpha beta, health report
PITFALLS: The model must provide ``head_modules() -> {target: NIGHead}`` (every ARCHITECTURES model does); there is
          no module-name convention any more. A floor <= 0 raises (the rule is relative to it).
"""

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.model.heads import head_spec

__all__ = ["nig_floors", "nig_floor_hit_report"]

PARAMS = ("nu", "alpha", "beta")


def nig_floors(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """{target: {"nu": nu_min, "alpha": alpha_min, "beta": beta_min}} per NIGHead of ``model.head_modules()``."""
    if not hasattr(model, "head_modules"):
        raise ValueError(f"{type(model).__name__} has no head_modules() (every architecture must provide it)")
    out = {t: {"nu": m.nu_min, "alpha": m.alpha_min, "beta": m.beta_min} for t, m in model.head_modules().items()}
    if not out:
        raise ValueError("the model has no NIGHead")
    return out


def nig_floor_hit_report(table: pd.DataFrame, floors: dict[str, dict[str, float]], rel_tol: float = 0.05) -> dict:
    """{target: {param: {"floor", "share_at_floor", "min", "median"}}}: the share of rows with value - floor <=
    rel_tol * floor. A floor that is 0 or negative raises (the rule is relative to it)."""
    if rel_tol <= 0:
        raise ValueError("rel_tol must be > 0")
    out = {}
    for t, fl in floors.items():
        keys = head_spec("nig_regression", t).output_keys
        out[t] = {}
        for p in PARAMS:
            if fl[p] <= 0:
                raise ValueError(f"floor of {t}.{p} is {fl[p]}; the relative rule needs a positive floor")
            if keys[p] not in table.columns:
                raise KeyError(f"table has no column {keys[p]!r}")
            v = table[keys[p]].to_numpy(dtype=np.float64)
            out[t][p] = {"floor": float(fl[p]), "share_at_floor": float(np.mean(v - fl[p] <= rel_tol * fl[p])),
                         "min": float(np.min(v)), "median": float(np.median(v))}
    return out
