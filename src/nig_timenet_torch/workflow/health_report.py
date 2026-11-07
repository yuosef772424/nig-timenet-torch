"""health_report(run): every T4 diagnostic of a run in one call

PURPOSE:  T4 health_report(run): one call writes every diagnostic of a run (targets, NIG floors, layer probe, trainer
          trace) as JSON and markdown next to the checkpoint.
TAGS:     health report, diagnostics, t-sharp, probe, trainer trace, markdown report
PITFALLS: ``last_close`` must be the last CLOSE price per row (not a head's own anchor). High/low AUC is information
          only (direction suspended). The close-head T-sharp verdict is informational only (O-1 closed).

T4 ``health_report(run)``: ONE call that writes every T4 diagnostic of a run next to its checkpoint.

Input contract (what T5r's job passes): the run folder ``Trainer.fit(run_dir=...)`` wrote (``checkpoint.pt`` with the
history and the Kendall state, ``best.pt`` / ``best_meta.json``) plus the predictions of the model on VALIDATION:
``predictions`` (a PredictionTable from ``workflow.prediction_table.predict_table``), its ``labels`` / ``anchors``
(aligned to the table rows) and the ``codec_state`` the labels were encoded with, and the ``model`` holding the weights
that made the predictions (``fit`` restores the best epoch) with the validation inputs ``x`` in table order.

Output: ``health_report.json`` (all numbers) and ``health_report.md`` (short, for reading) in the run folder.
Sections: target_report (T-skill / T-conf / T-sharp / coverage), nig_floor (floor-hit shares), probe (per-layer ridge
probe with the raw input next to every layer), trace (per-head loss, Kendall weights, schedule values, gradient
share; the ``trace_*`` columns exist only when the run was fitted with ``FitSettings.trace=True``)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.eval.nig_floor import nig_floor_hit_report, nig_floors
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.model.probe import layer_probe_report
from nig_timenet_torch.trainer import checkpoint

__all__ = ["HealthRun", "health_report", "REPORT_JSON", "REPORT_MD"]

REPORT_JSON, REPORT_MD = "health_report.json", "health_report.md"


@dataclass(frozen=True)
class HealthRun:
    run_dir: Path
    model: torch.nn.Module
    x: torch.Tensor                  # validation inputs, row i <-> predictions row i
    predictions: pd.DataFrame
    labels: dict
    anchors: dict
    last_close: np.ndarray           # last CLOSE price per row (reference of the predicted move in T-sharp)
    codec_state: dict
    probe_points: tuple[str, ...]    # module names inside ``model``; "raw_input" is always added
    probe_labels: tuple[str, ...]    # label keys to probe (a binary label gives AUC, a regression label R^2)
    probe_max_rows: int = 2000       # an evenly strided, order-preserving subsample (folds are contiguous in time)


def _trace_section(run_dir: Path) -> dict:
    ckpt = checkpoint.load_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"{run_dir}: no {checkpoint.CHECKPOINT}; health_report reads what Trainer.fit wrote")
    history = ckpt["history"]
    keep = [k for k in history[0] if k in ("epoch", "lr", "val_selection_metric") or k.endswith("_loss")
            or k.startswith("trace_")] if history else []
    series = {k: [row.get(k) for row in history] for k in keep}
    kendall = ckpt.get("kendall")
    final_kendall = None if kendall is None else {k.removeprefix("log_var_"): float(v) for k, v in kendall.items()}
    best_meta = run_dir / checkpoint.BEST_META
    return {"epochs_done": int(ckpt["epochs_done"]),
            "best": json.loads(best_meta.read_text(encoding="utf-8")) if best_meta.is_file() else None,
            "trace_enabled": any(k.startswith("trace_") for k in keep), "final_kendall_logvar": final_kendall,
            "series": series}


def _probe_section(run: HealthRun) -> dict:
    n = run.x.shape[0]
    ix = np.unique(np.linspace(0, n - 1, min(n, run.probe_max_rows)).astype(np.int64))
    out = {}
    for key in run.probe_labels:
        if key not in run.labels:
            raise KeyError(f"probe label {key!r} is not in labels")
        y = np.asarray(run.labels[key], dtype=np.float64)[ix]
        out[key] = layer_probe_report(run.model, run.x[torch.as_tensor(ix)], y,
                                      probe_points=list(run.probe_points)).to_dict("records")
    return out


def build_report(run: HealthRun) -> dict:
    n = len(run.predictions)
    if run.x.shape[0] != n:
        raise ValueError(f"x has {run.x.shape[0]} rows, predictions has {n}")
    return {"run_dir": str(run.run_dir),
            "targets": target_report(run.predictions, run.labels, anchors=run.anchors, last_close=run.last_close,
                                     codec_state=run.codec_state),
            "nig_floor": nig_floor_hit_report(run.predictions, nig_floors(run.model)),
            "probe": _probe_section(run), "trace": _trace_section(Path(run.run_dir))}


def _clean(o):
    """JSON-safe copy: numpy scalars -> python, NaN / inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.generic,)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _f(v, nd=3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def render_markdown(rep: dict) -> str:
    L = [f"# Health report — {rep['run_dir']}", "", f"Rows: {rep['targets']['n_rows']}", "",
         "## Targets (validation)", "",
         "| head | rank IC (ts) | per-coin rho | AUC (const / shuffled) | T-conf rho | gap CI95 | conf ok | "
         "std<move share | coverage80 | sharp ok |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in rep["targets"]["targets"].items():
        sk, c, s = r["skill"], r["conf"], r["sharp"]
        cl = sk.get("class")
        auc = "n/a" if cl is None else (f"{_f(cl['auc'])} ({_f(cl['auc_constant_baseline'], 2)} / "
                                        f"{_f(cl['auc_shuffled_baseline'])})"
                                        + (" info only (direction suspended)" if cl.get("information_only") else ""))
        ok = s.get("meets_target", "n/a")
        if not s.get("blocking", True):
            ok = f"{ok} (informational, O-1)"
        L.append(f"| {t} | {_f(sk['rank_ic_by_timestamp'])} | {_f(sk['spearman_per_coin']['mean'])} | {auc} | "
                 f"{_f(c['decile_error_spearman'])} | [{_f(c['gap_ci95'][0])}, {_f(c['gap_ci95'][1])}] | "
                 f"{c['meets_target']} | {_f(s.get('share_std_below_move'))} | {_f(s['coverage'])} | "
                 f"{ok} |")
    L += ["", "Baselines: constant confidence scores 0; a zero std scores 1.0 on std<move share (read it with "
          "coverage); constant AUC 0.5.", "", "## NIG floor-hit shares", "",
          "| head | nu | alpha | beta |", "|---|---|---|---|"]
    for t, p in rep["nig_floor"].items():
        L.append(f"| {t} | " + " | ".join(_f(p[k]["share_at_floor"], 2) for k in ("nu", "alpha", "beta")) + " |")
    for key, rows in rep["probe"].items():
        L += ["", f"## Layer probe — {key}", "", "| point | metric | mean | raw | delta vs raw | features |",
              "|---|---|---|---|---|---|"]
        L += [f"| {r['probe_point']} | {r['metric']} | {_f(r['mean'])} | {_f(r['raw_mean'])} | "
              f"{_f(r['delta_vs_raw'])} | {r['n_features']} |" for r in rows]
    tr = rep["trace"]
    L += ["", "## Trainer trace", "", f"Epochs: {tr['epochs_done']}; best: {tr['best']}; trace columns: "
          f"{'yes' if tr['trace_enabled'] else 'NO (fit without FitSettings.trace)'}", ""]
    last = {k: v[-1] for k, v in tr["series"].items() if v}
    L += [f"- {k}: {_f(v, 4)}" for k, v in last.items() if k.startswith(("trace_", "val_", "train_")) and v is not None]
    return "\n".join(L) + "\n"


def health_report(run: HealthRun) -> dict:
    """Build the report, write ``health_report.json`` and ``health_report.md`` into ``run.run_dir``, return it."""
    rep = _clean(build_report(run))
    run_dir = Path(run.run_dir)
    (run_dir / REPORT_JSON).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / REPORT_MD).write_text(render_markdown(rep), encoding="utf-8")
    return rep
