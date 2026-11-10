"""PURPOSE: The reporting half of grad_stack: learning curves (where memorization starts), the comparison files and
    the evaluation battery of every variant of a finished run folder.
TAGS: grad stack report, learning curves, overfit, memorization, battery, test report, gate verdicts
PITFALLS: ``grad_stack_battery`` reads the battery tables ``run_grad_stack`` wrote per variant folder (a folder from
    an older code version has none); ``n_variants`` = the number of variants found (multiple-testing correction).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.workflow import battery_run

__all__ = ["VARIANTS", "learning_curves", "write_report", "grad_stack_battery"]

VARIANTS = ("A", "data_h2", "grad", "grad_data", "grad_full", "full")   # every variant of grad_stack


def _flat(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def learning_curves(models: dict, out: Path) -> pd.DataFrame:
    """Where memorization starts: per variant and epoch the train / val raw loss (``learning_curves.csv``) and a
    digest: best epoch, val at best / last, ``overfit`` = val_last - val_best (> 0: val got worse after the best
    epoch), ``gap`` = val_last - train_last (train much lower than val: the model fits rows it cannot generalise)."""
    rows = [{"variant": n, "epoch": h["epoch"], "train": h.get("train_raw_loss"), "val": h.get("val_raw_loss")}
            for n, r in models.items() for h in r.result.history]
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "learning_curves.csv", index=False)
    digest = {}
    for n, c in curves.groupby("variant", sort=False):
        best = int(models[n].result.best_epoch or c["epoch"].iloc[-1])
        vb = float(c.loc[c["epoch"] == best, "val"].iloc[0])
        digest[n] = {"epochs": len(c), "best_epoch": best, "train_first": c["train"].iloc[0],
                     "train_last": c["train"].iloc[-1], "val_best": vb, "val_last": c["val"].iloc[-1],
                     "overfit": c["val"].iloc[-1] - vb, "gap": c["val"].iloc[-1] - c["train"].iloc[-1]}
    dg = pd.DataFrame(digest).T
    print("\n== learning curves (raw loss; lower = better) ==\n", dg.round(4).to_string(), flush=True)
    return dg


def write_report(results: dict, rows: list[dict], out: Path, meta: dict) -> dict:
    """The comparison files (test report per variant, gate rows / verdicts, summary.json) and a printed digest."""
    summary = pd.DataFrame({n: _flat(r) for n, r in results.items()})
    summary.to_csv(out / "test_report_by_variant.csv")
    gdf = pd.DataFrame(rows)
    gdf.to_csv(out / "gate_rows.csv", index=False)
    blocking = gdf[gdf["blocking"].astype(bool)]
    verdicts = blocking.groupby(["variant", "verdict"]).size().unstack(fill_value=0).reindex(list(results))
    verdicts.to_csv(out / "gate_verdicts.csv")
    failed = blocking[blocking.verdict == "fail"]
    meta["gate_fails"] = {v: sorted(f"{r.gate}|{r.check}|{r.head}" for r in failed[failed.variant == v].itertuples())
                          for v in results}
    (out / "summary.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    key_rows = [k for k in summary.index if k.endswith(("rank_ic_by_timestamp", "spearman_per_coin.mean",
                                                         "class.auc", "decile_error_spearman", "sharp.coverage"))]
    print("\n== gate verdicts (blocking rows) ==\n", verdicts)
    print("\n== key test metrics ==\n", summary.loc[key_rows].round(4).to_string())
    print("\nfailed blocking checks per variant:", {v: len(f) for v, f in meta["gate_fails"].items()},
          "(list: summary.json)")
    return meta


def grad_stack_battery(out, *, cost: float = 0.0008, show: bool = True) -> dict:
    """The evaluation battery (T-profit after ``cost`` per side) of every variant of a finished ``run_grad_stack``
    folder; ``n_variants`` = the number of variants (the battery's multiple-testing correction). Writes
    ``battery_report.txt`` / ``battery_profit.json`` into each variant folder; -> {variant: battery sections}."""
    runs = Path(out) / "runs"
    names = [v for v in VARIANTS if battery_run.table_path(runs / v, "test").is_file()]
    if not names:
        raise FileNotFoundError(f"no battery tables under {runs}: run run_grad_stack again with this code version")
    res = {}
    for v in names:
        res[v] = battery_run.battery_from_folder(runs / v, cost=cost, n_variants=len(names), out_dir=runs / v)
        if show:
            print(f"\n{'=' * 30} battery: {v} {'=' * 30}\n{battery_run.render(res[v])}", flush=True)
    return res
