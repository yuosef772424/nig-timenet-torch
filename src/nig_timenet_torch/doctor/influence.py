"""PURPOSE: ``python -m nig_timenet_torch.doctor.influence``: train a configuration with the batch-influence monitor
    on, then write the batch report (by coin, by month) and the permutation importance of every input feature on VAL.
TAGS: doctor, influence entry, batch influence, feature importance, cli, report writer, e1 known answer, coins, months
PITFALLS: the monitor adds one probe forward per step per probe (and a backward every ``first_order_every`` steps):
    a diagnosis run, not a production run; with ``--trained`` (a finished run restored) there is no training, so only
    the feature importance is written; importance is measured on VAL rows (never the training rows).

Outputs under ``--out``: ``batch_influence/`` (``batch_influence.csv`` one row per step, ``batch_rows.npz`` the
composition of every batch, ``batch_influence_report.md`` / ``.json`` grouped by coin, ``by_month/`` the same grouped by
the month of the training row) and ``feature_importance.md`` / ``.csv`` / ``.json`` (features, clusters, bar blocks;
with ``--within-time`` also ``feature_importance_within_time``). ``python -m nig_timenet_torch.doctor.influence e1``
writes a synthetic E1 artifact first (known answer: the close signal sits in the last closes, the volatility in the
high / low ranges).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.doctor.common import make_context, row_times, stride
from nig_timenet_torch.doctor.feature_importance import ImportanceSettings, feature_importance, write_importance
from nig_timenet_torch.eval.influence_report import write_influence_report
from nig_timenet_torch.trainer.batch_influence import BatchInfluence, InfluenceSettings
from nig_timenet_torch.workflow import flat_run

__all__ = ["monitor_for", "batch_reports", "importance_reports", "main"]


def monitor_for(run, settings: InfluenceSettings, train_probe: bool = True) -> BatchInfluence:
    """The monitor of a built Run: the primary probe on VAL rows, a second one on TRAINING rows (memorisation)."""
    probes = {"val": (run.x["val"], run.y["val"])}
    if train_probe:
        probes["train"] = (run.x["train"], run.y["train"])
    return BatchInfluence(probes, settings, group_channel=run.settings.split.cross_section)


def batch_reports(out, run, monitor: BatchInfluence, n_perm: int = 1000) -> dict:
    """Save the record and write the report grouped by coin (``out``) and by month of the training row
    (``out/by_month``); returns the coin report."""
    out = Path(out)
    monitor.save(out)
    frame, (steps, rows) = monitor.frame(), monitor.rows_of_steps()
    idx = run.idx["train"]
    coins = flat_run.row_coins(run.dataset, idx).astype(str)
    months = pd.to_datetime(row_times(run.dataset, idx), utc=True).strftime("%Y-%m").to_numpy()
    res = write_influence_report(out, frame, steps, rows, coins, n_perm=n_perm, title="Batch influence by coin")
    write_influence_report(out / "by_month", frame, steps, rows, months, n_perm=n_perm,
                           title="Batch influence by month")
    return res


def _names(dataset: dict, n_channels: int) -> list[str]:
    fo = list(dataset["feature_order"]) if "feature_order" in dataset else []
    return fo if len(fo) == n_channels else [f"f{j}" for j in range(n_channels)]


def importance_reports(out, run, settings: ImportanceSettings, max_rows: int | None = 50_000,
                       within_time: bool = False) -> dict:
    """Permutation importance of the trained model on (evenly strided) VAL rows: per feature, per cluster and per
    block of bars (``feature_importance``) and, with ``within_time``, across coins of the same timestamp only
    (``feature_importance_within_time``). Returns {name: summary}."""
    idx = stride(run.idx["val"], max_rows)
    cross = run.settings.split.cross_section
    x, y = flat_run.flat_inputs(run.dataset, idx, cross)
    times, names = row_times(run.dataset, idx), _names(run.dataset, x.shape[2] - int(cross))
    jobs = {"feature_importance": settings}
    if within_time:
        jobs["feature_importance_within_time"] = ImportanceSettings(**{**settings.__dict__,
                                                                      "permutation": "within_time"})
    res = {}
    for name, s in jobs.items():
        table, summary = feature_importance(run.trainer, x, y, times, names, s)
        write_importance(out, table, summary, name=name, title=name.replace("_", " ").capitalize())
        res[name] = summary
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.doctor.influence",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("artifact", help="a contract artifact folder, or 'e1' to write and use a synthetic E1")
    ap.add_argument("--out", default="influence_out")
    ap.add_argument("--max-rows", type=int, default=60_000, help="rows per split (evenly strided)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="{}", help="JSON model overrides over the default preset")
    ap.add_argument("--train", default="{}", help="JSON TrainSettings overrides")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--trained", action="store_true", help="restore --run-dir's best weights: importance only")
    ap.add_argument("--probe-rows", type=int, default=1024)
    ap.add_argument("--first-order-every", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fi-rows", type=int, default=50_000, help="VAL rows of the feature importance")
    ap.add_argument("--within-time", action="store_true", help="also permute across coins of one timestamp only")
    a = ap.parse_args(argv)
    out = Path(a.out)
    artifact = a.artifact
    if artifact.lower() == "e1":
        artifact = str(out / "e1_artifact")
        write_synthetic(artifact, SyntheticSettings(n_coins=16, n_bars=2000), overwrite=True)
    bi = InfluenceSettings(probe_rows=a.probe_rows, first_order_every=a.first_order_every, seed=a.seed)
    monitors = []

    def monitor_of(r):
        monitors.append(monitor_for(r, bi))
        return monitors[-1]

    t0 = time.monotonic()
    ctx = make_context(artifact, out, run_dir=a.run_dir, model=json.loads(a.model), train=json.loads(a.train),
                       max_rows=a.max_rows, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
                       trained=a.trained, monitor_of=None if a.trained else monitor_of)
    print(f"trained / restored in {time.monotonic() - t0:.0f} s", flush=True)
    if monitors:
        res = batch_reports(out / "batch_influence", ctx.run, monitors[0], n_perm=a.n_perm)
        sh = res["split_half"]
        print(f"batch influence: {res['n_steps']} steps, total change {res['total_delta']:.4g}, harmful "
              f"{res['harmful_share']:.1%}, split-half reliability {sh['reliability']:.3f}", flush=True)
    fi = importance_reports(out, ctx.run, ImportanceSettings(repeats=a.repeats, bar_blocks=4, seed=a.seed), a.fi_rows,
                            a.within_time)
    for name, s in fi.items():
        print(f"{name}: uses {s['uses']}, hurts {s['hurts']} of {s['groups']} groups, split-half rho "
              f"{s['split_half_rho']:.3f}", flush=True)
    print(f"reports under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

