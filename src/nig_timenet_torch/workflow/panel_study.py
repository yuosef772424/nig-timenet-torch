"""PURPOSE: The panel (multi-coin) study: named arms x seeds through the run steps on whole-timestamp batches, each
    scored with the day metrics and compared with the baseline arm on paired days; resumable arm by arm.
TAGS: panel study, multi coin arms, cross section, day metrics, winning days, paired days, accept rule, resumable,
    deadline, arm overrides, study json
PITFALLS: exploratory (rule C5): a CPU run is a look; the accept rule needs >= 2 seeds (rule C4) and is the
    pre-registered one (docs/audit/2026-11-16-panel/PREREGISTERED.md): a win over EVERY baseline of
    ``eval/day_metrics.py`` on every seed; ``vs_baseline_arm`` is extra (an arm against the baseline arm on the same
    days). A finished arm leaves ``arm.json`` + its frames under ``ckpt_root``: a later call reads them instead of
    training again. A deadline stops BEFORE an arm that cannot start and inside an arm at an epoch border
    (``finished`` False: call again to continue). The gates of rule C6 are not run here.

An arm = ``{"model": {model overrides}, "train": {TrainSettings fields}}`` over the study's base settings (which must
set ``split.cross_section``). Outputs in ``out``: ``study.json``, ``study.md`` and one folder per arm x seed with the
summary, the test report, the panel report, the two frames and, when the settings enable it, the battery report.
"""

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.eval import day_metrics
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.workflow import battery_run, panel_report
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["ARMS", "BASELINE_ARM", "arm_settings", "run_arm", "run_panel_study", "compare", "render", "ARM_JSON"]

BASELINE_ARM = "per_coin"
#: the named arms (docs/audit/2026-11-16-panel): the per-coin model on whole-timestamp batches (the baseline arm),
#: + relative inputs, + the cross-section rank loss, + the close mean centred inside its timestamp (the four arms of
#: panel-001); the last one selects the epoch by the long / short book of day metric 2 instead
_REL = {"cross_input": "relative"}
ARMS = {
    "per_coin": {},
    "relative": {"model": _REL},
    "relative_rank_loss": {"model": _REL, "train": {"cross_section_loss": "rank_ic"}},
    "relative_rank_loss_center": {"model": {**_REL, "cross_center": True}, "train": {"cross_section_loss": "rank_ic"}},
    "relative_rank_loss_ls": {"model": _REL,
                              "train": {"cross_section_loss": "rank_ic", "selection_metric": "val_ls_sharpe_close"}},
}
ARM_JSON = "arm.json"
_KEPT = (panel_report.REPORT_JSON, panel_report.REPORT_MD, "panel_frame_val.parquet", "panel_frame_test.parquet",
         steps.SUMMARY, steps.TEST_REPORT, battery_run.REPORT_TXT, battery_run.PROFIT_JSON)


def arm_settings(settings: RunSettings, arm: dict, seed: int, run_dir) -> RunSettings:
    """``settings`` with the arm's model overrides and train fields, the seed and the run folder."""
    unknown = sorted(set(arm) - {"model", "train"})
    if unknown:
        raise ValueError(f"an arm may hold 'model' and 'train', not {unknown}")
    model = ModelSettings(settings.model.architecture, {**settings.model.overrides, **arm.get("model", {})})
    changes = {**{f"train.{k}": v for k, v in arm.get("train", {}).items()}, "train.seed": seed,
               "run_dir": str(run_dir), "model": model}
    return updated(settings, changes)


def _peak_mb(device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None


def run_arm(settings: RunSettings, name: str, arm: dict, seed: int, out, *, ckpt_root=None,
            deadline: float | None = None, on_epoch=None) -> dict:
    """One arm x seed -> its row. A finished arm (``arm.json`` under the checkpoint root) is read back, its files
    copied to ``out``; else it trains (resuming its checkpoint) until done or the deadline."""
    folder = Path(out) / f"{name}_s{seed}"
    keep = Path(ckpt_root or out) / f"{name}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (keep / ARM_JSON).is_file():
        for f in _KEPT:
            if (keep / f).is_file() and keep != folder:
                shutil.copyfile(keep / f, folder / f)
        return json.loads((keep / ARM_JSON).read_text())
    t0 = time.monotonic()
    r = steps.build(steps.prepare(arm_settings(settings, arm, seed, keep / "ckpt"), folder))
    if r.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(r.device)
    steps.train(r, deadline=deadline, on_epoch=on_epoch)
    res = r.result
    row = {"arm": name, "seed": seed, "epochs_done": res.epochs_done, "best_epoch": res.best_epoch,
           "best_value": res.best_value, "finished": res.epochs_done >= r.fit_settings.epochs or res.stopped_early,
           "rows": {k: int(len(v)) for k, v in r.idx.items()}, "data_fingerprint": r.art.manifest.data_fingerprint,
           "epoch_seconds": [h["trace_epoch_seconds"] for h in res.history if "trace_epoch_seconds" in h],
           "peak_memory_mb": _peak_mb(r.device), "selection_metric": r.settings.train.selection_metric}
    if not row["finished"]:
        return row
    rep = steps.test_step(r)
    row["test_rank_ic"] = {t: rep["targets"][t]["skill"]["rank_ic_by_timestamp"] for t in r.targets}
    row.update(panel_report.headline(panel_report.panel_step(r)))
    steps.write_summary(r)
    verdict = steps.battery_step(r).get("verdict")      # {} when the settings switch the battery off
    if verdict is not None:
        row["battery_tradeable"] = verdict.get("tradeable")
    row["seconds"] = time.monotonic() - t0
    keep.mkdir(parents=True, exist_ok=True)
    for f in _KEPT:
        if (folder / f).is_file() and keep != folder:
            shutil.copyfile(folder / f, keep / f)
    (keep / ARM_JSON).write_text(json.dumps(row, indent=1, default=float), encoding="utf-8")
    return row


def _days(folder: Path, top_share: float = 0.2) -> dict:
    frame = pd.read_parquet(panel_report.frame_path(folder, "test"))
    mu = frame["mu"].to_numpy()
    return {"absolute": day_metrics.absolute_days(frame, mu), "neutral": day_metrics.neutral_days(frame, mu, top_share)}


def compare(out, arms, seeds, baseline: str = BASELINE_ARM) -> dict:
    """{arm: {metric: [paired_days(arm, baseline arm) per seed]}} from the TEST frames of the finished arms."""
    res = {}
    for seed in seeds:
        base = Path(out) / f"{baseline}_s{seed}"
        if not panel_report.frame_path(base, "test").is_file():
            continue
        b = _days(base)
        for name in arms:
            folder = Path(out) / f"{name}_s{seed}"
            if name == baseline or not panel_report.frame_path(folder, "test").is_file():
                continue
            a = _days(folder)
            for metric in ("absolute", "neutral"):
                res.setdefault(name, {}).setdefault(metric, []).append(
                    {"seed": seed, **day_metrics.paired_days(a[metric], b[metric], seed=seed)})
    return res


def run_panel_study(settings: RunSettings, arms: dict, seeds, out, *, ckpt_root=None, deadline: float | None = None,
                    on_epoch=None) -> dict:
    """Every seed x arm (seeds outer, so two seeds of the first arms exist before the last arm starts on none);
    writes ``study.json`` / ``study.md``. ``finished`` False = the deadline came first: call again to continue."""
    if not settings.split.cross_section:
        raise ValueError("the panel study needs split.cross_section (whole timestamps)")
    rows, finished, longest = {name: [] for name in arms}, True, 0.0
    for seed in seeds:
        for name, arm in arms.items():
            if deadline is not None and longest and time.monotonic() + 0.25 * longest > deadline:
                finished = False
                continue
            t0 = time.monotonic()
            row = run_arm(settings, name, arm, seed, out, ckpt_root=ckpt_root, deadline=deadline, on_epoch=on_epoch)
            longest = max(longest, time.monotonic() - t0)
            rows[name].append(row)
            finished = finished and row["finished"]
            state = _line(row) if row["finished"] else f"cut by the deadline after {row['epochs_done']} epoch(s)"
            print(f"arm {name} seed {seed}: {state}", flush=True)
    done = {name: [r for r in rs if r["finished"]] for name, rs in rows.items()}
    study = {"seeds": list(seeds), "arms": rows, "arm_overrides": arms, "finished": finished,
             "data_fingerprint": next((r["data_fingerprint"] for rs in rows.values() for r in rs), ""),
             "accept": {name: _accept(rs) for name, rs in done.items() if rs},
             "vs_baseline_arm": compare(out, arms, seeds) if BASELINE_ARM in arms else {}}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "study.json").write_text(json.dumps(study, indent=1, default=float), encoding="utf-8")
    (Path(out) / "study.md").write_text(render(study), encoding="utf-8")
    return study


def _accept(rows: list[dict]) -> dict:
    """The pre-registered accept rule on the finished seeds of one arm, per metric."""
    out = {}
    for metric in ("absolute", "neutral"):
        wins = [bool(r[metric]["beats_every_baseline"]) for r in rows]
        out[metric] = {"seeds": len(wins), "seeds_beating_every_baseline": int(sum(wins)),
                       "accepted": len(wins) >= 2 and all(wins)}
    return out


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}"


def _line(r: dict) -> str:
    a, n = r["absolute"], r["neutral"]
    return (f"best epoch {r['best_epoch']}/{r['epochs_done']}, metric 1 {_pct(a['win_share'])} % winning days "
            f"(net mean {a['mean_net'] * 100:+.4f} %), metric 2 {_pct(n['win_share'])} % (net "
            f"{n['mean_net'] * 100:+.4f} %), relative rank IC {r['relative_rank_ic']:+.4f}")


def render(study: dict) -> str:
    """study.md: one row per arm x seed (TEST), the accept rule per arm, the paired comparison with the baseline
    arm."""
    text = ["# Panel study (exploratory, rule C5)", "",
            f"Data `{study['data_fingerprint'][:12]}`; seeds {study['seeds']}; "
            f"{'finished' if study['finished'] else 'NOT finished (deadline): run the job again to continue'}.", "",
            "| arm | seed | best epoch / epochs | metric 1: winning days [CI], gross | mean day net | beats every "
            "baseline | metric 2: winning days [CI], gross | mean day net | beats every baseline | relative rank IC | "
            "rank IC high / low / close | s / epoch | peak MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rows in study["arms"].items():
        for r in rows:
            if not r["finished"]:
                text.append(f"| {name} | {r['seed']} | cut at {r['epochs_done']} | | | | | | | | | | |")
                continue
            a, n, ic = r["absolute"], r["neutral"], r["test_rank_ic"]
            secs = sorted(r["epoch_seconds"])[len(r["epoch_seconds"]) // 2] if r["epoch_seconds"] else float("nan")
            mem = "" if r["peak_memory_mb"] is None else f"{r['peak_memory_mb']:.0f}"
            text.append(
                f"| {name} | {r['seed']} | {r['best_epoch']} / {r['epochs_done']} | {_pct(a['win_share'])} % "
                f"[{_pct(a['win_lo'])}, {_pct(a['win_hi'])}] | {a['mean_net'] * 100:+.4f} % | "
                f"{a['beats_every_baseline']} | {_pct(n['win_share'])} % [{_pct(n['win_lo'])}, {_pct(n['win_hi'])}] | "
                f"{n['mean_net'] * 100:+.4f} % | {n['beats_every_baseline']} | {r['relative_rank_ic']:+.4f} | "
                + " / ".join(f"{ic[t]:.3f}" for t in TARGET_COLUMNS if t in ic) + f" | {secs:.0f} | {mem} |")
    text += ["", "## Accept rule (a paired win over EVERY baseline on >= 2 seeds)", ""]
    for name, acc in study["accept"].items():
        text.append(f"- {name}: " + "; ".join(
            f"{m} {'ACCEPTED' if v['accepted'] else 'not accepted'} ({v['seeds_beating_every_baseline']}/{v['seeds']} "
            f"seeds)" for m, v in acc.items()))
    if study["vs_baseline_arm"]:
        text += ["", f"## Against the baseline arm `{BASELINE_ARM}` (paired days, TEST)", "",
                 "| arm | metric | seed | days | mean day difference [CI] | winning-day difference [CI] | wins |",
                 "|---|---|---|---|---|---|---|"]
        for name, per in study["vs_baseline_arm"].items():
            for metric, rows in per.items():
                for p in rows:
                    if p["n_days"] < 2:
                        continue
                    text.append(
                        f"| {name} | {metric} | {p['seed']} | {p['n_days']} | {p['d_mean'] * 100:+.4f} % "
                        f"[{p['d_mean_lo'] * 100:+.4f}, {p['d_mean_hi'] * 100:+.4f}] | {p['d_win'] * 100:+.1f} pp "
                        f"[{p['d_win_lo'] * 100:+.1f}, {p['d_win_hi'] * 100:+.1f}] | {p['wins']} |")
    return "\n".join(text) + "\n"
