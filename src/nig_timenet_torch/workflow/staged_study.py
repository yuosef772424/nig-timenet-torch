"""PURPOSE: The staged-training experiment of ROADMAP ST1 (card F section 5): per seed one staged run and the
    equal-step baselines B1 / B2 / B3 through the run steps, the per-stage tables, the comparison and the accept rule.
TAGS: staged study, baselines b1 b2 b3, equal steps, accept rule, seeds, stage table, steps to the sign, st1
PITFALLS: exploratory (rule C5): a CPU smoke is not a result; an arm is compared only at EQUAL optimizer steps, the
    same seed (same initial weights) and the same fixed validation rows (``comparison`` warns when the steps differ);
    the accept rule needs >= 2 seeds (rule C4) and the ``val_raw_loss`` selection metric, otherwise the verdict says
    it cannot judge; every arm trains through ``run.train_staged`` (the one Trainer), the baselines with a plan from
    ``trainer.staged.baseline_stages`` and no sign, no restart.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.eval.target_report import INFORMATION_ONLY_DIRECTION_HEADS
from nig_timenet_torch.trainer.selection_metric import SELECTION_METRICS
from nig_timenet_torch.trainer.staged import baseline_stages
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings
from nig_timenet_torch.workflow.staged_report import comparison, stage_table

__all__ = ["BASELINES", "run_arm", "run_staged_study", "verdict", "render"]

BASELINES = ("b1", "b2", "b3")


def _ic(report: dict) -> dict:
    """{flat key: value} of every by-timestamp rank IC of a test report (the trivial shuffled baseline excluded)."""
    return {k: float(v) for k, v in steps.flat_scalars(report) if k.endswith("rank_ic_by_timestamp")}


def run_arm(settings: RunSettings, arm: str, seed: int, root, *, plan_from=None, test: bool = True,
            on_eval=None, ckpt_root=None, deadline: float | None = None) -> tuple:
    """One arm x seed: ``arm`` is ``staged`` or a baseline kind; ``plan_from`` (the finished staged Run of this seed)
    gives a baseline its step count and lr boundaries. The outputs go to ``root/<arm>_s<seed>``, the checkpoint (the
    resume point) to ``<ckpt_root or root>/<arm>_s<seed>/ckpt``. Returns (Run, row); an arm cut by ``deadline`` (a
    ``time.monotonic()`` value) returns its row with ``finished`` False and no test numbers."""
    folder = Path(root) / f"{arm}_s{seed}"
    s = updated(settings, {"train.seed": seed, "run_dir": str(Path(ckpt_root or root) / f"{arm}_s{seed}" / "ckpt")})
    r = steps.build(steps.prepare(s, folder))
    plan = None
    if arm != "staged":
        plan = baseline_stages(arm, plan_from.plan, plan_from.staged, s.train.staged, s.train.batch_size)
    steps.train_staged(r, stages=plan, on_eval=on_eval, deadline=deadline)
    row = {"seed": seed, "steps": r.staged.total_steps, "best_value": r.staged.best_value,
           "finished": r.staged.finished, "best_step": r.staged.best_step, "stages": r.extra["stages"]}
    if not row["finished"]:
        return r, row
    if test and len(r.idx["test"]):
        row["test_rank_ic"] = _ic(steps.test_step(r))
    steps.write_summary(r)
    return r, row


def run_staged_study(settings: RunSettings, seeds, out, *, baselines=BASELINES, test: bool = True,
                     on_eval=None, ckpt_root=None, deadline: float | None = None) -> dict:
    """Every seed: the staged arm, then ``baselines`` at its number of steps. Writes ``study.json`` and ``study.md``
    to ``out`` and returns the study dict ``{"seeds", "arms": {arm: [row per seed]}, "finished", "verdict", ...}``.
    ``ckpt_root`` puts the checkpoints elsewhere (a Drive folder) than the outputs. ``deadline`` (a
    ``time.monotonic()`` value): no arm starts after it, a running arm stops at an evaluation point before it;
    ``finished`` False = cut (run again: every arm resumes from its checkpoint)."""
    if settings.train.staged is None:
        raise ValueError("run_staged_study needs settings.train.staged")
    arms: dict[str, list] = {"staged": [], **{b: [] for b in baselines}}
    fingerprint, finished, base = "", True, None    # base: this seed's finished staged Run (the baselines' plan)
    kw = dict(test=test, on_eval=on_eval, ckpt_root=ckpt_root, deadline=deadline)
    for seed, arm in ((s, a) for s in seeds for a in ("staged", *baselines)):
        if not finished or (deadline is not None and time.monotonic() >= deadline):
            finished = False
            break
        r, row = run_arm(settings, arm, seed, out, plan_from=None if arm == "staged" else base, **kw)
        if arm == "staged":
            base, fingerprint = r, r.art.manifest.data_fingerprint
        arms[arm].append(row)
        finished = row["finished"]
    study = {"seeds": list(seeds), "data_fingerprint": fingerprint, "arms": arms, "finished": finished,
             "selection_metric": settings.train.selection_metric,
             "settings": dataclasses.asdict(settings.train.staged)}
    study["verdict"] = verdict(study)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "study.json").write_text(json.dumps(study, indent=1, default=float), encoding="utf-8")
    (Path(out) / "study.md").write_text(render(study), encoding="utf-8")
    return study


def verdict(study: dict) -> str:
    """The accept rule of card F section 5 in one sentence: on EVERY seed the staged best ``val_raw_loss`` beats B1 and
    B2 by more than the seed spread of B1, and its test rank IC is not lower than B1's. Anything else (or not enough
    arms / seeds) is "no gain" or "cannot judge"."""
    arms = study["arms"]
    if not study.get("finished", True):
        return "cannot judge: the study was cut by the deadline (finished false); run the job again to continue."
    if len(study["seeds"]) < 2 or not {"b1", "b2"} <= set(arms):
        return "cannot judge: the accept rule needs >= 2 seeds and the baselines b1 and b2 (rule C4)."
    if SELECTION_METRICS.entry(study["selection_metric"]).meta["higher_is_better"]:
        return "cannot judge: the accept rule is stated for a lower-is-better selection metric (val_raw_loss)."
    b1 = [r["best_value"] for r in arms["b1"]]
    spread = float(max(b1) - min(b1))
    wins, ics = [], []
    for st, r1, r2 in zip(arms["staged"], arms["b1"], arms["b2"], strict=True):
        wins.append(st["best_value"] < min(r1["best_value"], r2["best_value"]) - spread)
        ics.append(all(v >= r1.get("test_rank_ic", {}).get(k, -np.inf) for k, v in st.get("test_rank_ic", {}).items()
                       if any(h in k for h in INFORMATION_ONLY_DIRECTION_HEADS)))
    if all(wins) and all(ics):
        return (f"staged training helps by the accept rule (every seed beats B1 and B2 by more than the B1 seed spread "
                f"{spread:.4g}; test rank IC not lower): run the battery on the winner and on B1.")
    return (f"no gain by the accept rule (seeds beating B1 and B2 by more than the B1 seed spread {spread:.4g}: "
            f"{sum(wins)}/{len(wins)}; test rank IC not lower on {sum(ics)}/{len(ics)}); see the per-stage tables.")


def render(study: dict) -> str:
    """study.md: the verdict, the comparison table, then the stage table of the staged arm of every seed."""
    text = ["# Staged training study (exploratory, rule C5)", "", f"Verdict: {study['verdict']}", "",
            comparison({a: [{k: v for k, v in r.items() if k != "stages"} for r in rows]
                        for a, rows in study["arms"].items()})]
    for r in study["arms"]["staged"]:
        text += ["", f"## Stage table, staged arm, seed {r['seed']}", "", stage_table(r["stages"])]
    return "\n".join(text) + "\n"
