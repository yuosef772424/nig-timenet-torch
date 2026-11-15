"""PURPOSE: Card E — the readiness report: ONE command runs cards A-D on a configuration and prints, per check: stage,
    check, value, baseline, verdict and what to change; then the diagnosis: which stage to work on first.
TAGS: doctor, card e, readiness report, entry point, diagnosis, which stage, cli
PITFALLS: the default mode trains the configuration itself for a few epochs on strided rows (CPU quick mode), so its
    numbers describe the METHOD on a short run, not a finished long run: pass ``--run-dir`` + ``--trained`` to examine
    a finished run's best weights instead; the diagnosis order is data -> trainer -> model -> metrics (a data fault
    makes every later number meaningless).

``python -m nig_timenet_torch.doctor ARTIFACT|e1 [--out DIR] [--cards A,B,C,D] [--full] [--max-rows N] [--epochs N]
[--model JSON] [--train JSON] [--no-preset] [--architecture NAME] [--run-dir DIR --trained] [--include-test]``

Outputs in ``--out``: ``readiness.md`` (diagnosis, failing checks, every check), ``readiness.json`` (every check with
its detail numbers) and ``info_flow.csv`` (card A's stage table). The cards under docs/audit/*/cards say how to read
each section and what to change.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.doctor import data_checks, info_flow, metric_checks, trainer_checks
from nig_timenet_torch.doctor.common import Check, DoctorContext, make_context, render_table, worst, write_report

__all__ = ["run_cards", "diagnosis", "render_report", "main", "CARDS", "CARD_FILES"]

CARDS = ("C", "B", "A", "D")       # run order: data first (its ceiling feeds D), then trainer, model, metrics
CARD_FILES = {"A": "A_information_flow.md", "B": "B_trainer.md", "C": "C_data.md", "D": "D_metrics.md",
              "E": "E_readiness.md"}
CARD_TITLES = {"A": "model: information flow", "B": "trainer", "C": "data", "D": "metrics and profit"}
_HARD_DATA = ("data/alignment", "data/split", "data/coverage")


def run_cards(ctx: DoctorContext, cards=CARDS, *, include_test: bool = False) -> dict:
    """{"checks": [Check], "info_table": DataFrame | None, "seconds": {card: s}} for the cards asked (any order in,
    the fixed CARDS order out)."""
    checks: list[Check] = []
    table, secs = None, {}
    for card in [c for c in CARDS if c in cards]:
        t0 = time.monotonic()
        if card == "C":
            checks += data_checks.data_checks(ctx)
        elif card == "B":
            checks += trainer_checks.trainer_checks(ctx)
        elif card == "A":
            got, table = info_flow.info_flow(ctx)
            checks += got
        elif card == "D":
            checks += metric_checks.metric_checks(ctx, include_test=include_test)
        secs[card] = round(time.monotonic() - t0, 1)
        print(f"card {card}: {secs[card]} s, worst verdict {worst([c for c in checks if c.card == card])}", flush=True)
    return {"checks": checks, "info_table": table, "seconds": secs}


def diagnosis(checks: list[Check]) -> list[str]:
    """The reading order of a failed report: the first line that applies is the stage to work on first."""
    fails = [c for c in checks if c.verdict == "fail"]
    by = lambda card: [c for c in fails if c.card == card]  # noqa: E731
    lines = []
    hard = [c for c in by("C") if c.stage in _HARD_DATA]
    if hard:
        lines.append(f"1. DATA is not sound ({len(hard)} check(s): {'; '.join(c.check for c in hard[:3])}). Fix the "
                     "data first: every other number is measured on it.")
    units = [c for c in by("C") if c.stage == "data/units"]
    if units:
        lines.append(f"{len(lines) + 1}. UNITS: {len(units)} scale check(s) fail (labels or inputs are not in the "
                     "units the model's constants assume): set the fitted scalers (card C, units).")
    if any(c.card == "B" for c in checks):
        v, names = trainer_checks.verdict(checks)
        if v != "learns":
            lines.append(f"{len(lines) + 1}. TRAINER: {v.upper()} — {'; '.join(names[:4])} (card B).")
        elif names:
            lines.append(f"{len(lines) + 1}. TRAINER learns, with failing health checks: {'; '.join(names[:4])}.")
    a = by("A")
    if a:
        lines.append(f"{len(lines) + 1}. MODEL loses information: {'; '.join(f'{c.stage}: {c.check}' for c in a[:3])} "
                     "(card A, stage table in info_flow.csv).")
    no_info = [c for c in checks if c.card == "C" and c.stage == "data/information" and c.verdict == "warn"]
    if no_info:
        lines.append(f"{len(lines) + 1}. INPUTS hold no readable signal for: "
                     f"{', '.join(c.check.split(':')[0] for c in no_info)}. No model or trainer change fixes this; "
                     "the data needs inputs that carry it (card C, input information).")
    goal = [c for c in checks if c.card == "D" and c.check.startswith("GOAL")]
    if goal:
        ready = all(c.verdict == "pass" for c in goal)
        lines.append(f"{len(lines) + 1}. GOAL (profit after cost): {'MET on ' if ready else 'NOT met on '}"
                     f"{', '.join(c.stage.split('/')[1] for c in goal)}"
                     + ("" if ready else ". The lines above say which stage to work on first."))
    d = [c for c in by("D") if not c.check.startswith("GOAL") and not c.stage.startswith("profit")]
    if d:
        lines.append(f"{len(lines) + 1}. METRICS below their trivial baseline: "
                     f"{'; '.join(f'{c.stage}: {c.check}' for c in d[:4])}.")
    return lines or ["No failing check."]


def render_report(ctx: DoctorContext, result: dict, title: str) -> str:
    checks = result["checks"]
    r = ctx.run
    cfg = {"model": r.ms.overrides, "architecture": r.ms.architecture,
           "train": {k: getattr(r.settings.train, k) for k in ("epochs", "batch_size", "lr", "weight_decay", "seed")}}
    L = [f"# Readiness report — {title}", "",
         f"Rows: {len(r.idx['train'])} train / {len(r.idx['val'])} val / {len(r.idx['test'])} test "
         f"(max_rows {r.settings.split.max_rows}); epochs done {r.result.epochs_done}, best epoch "
         f"{r.result.best_epoch}; mode {'quick' if ctx.quick else 'full'}; seconds per card {result['seconds']}.", "",
         f"Configuration: `{json.dumps(cfg, default=str)}`", "", "## Diagnosis (read top down)", ""]
    L += diagnosis(checks)
    L += ["", "## Verdict per card", "", "| card | what | pass | warn | fail | read |", "|---|---|---|---|---|---|"]
    for card in ("A", "B", "C", "D"):
        mine = [c for c in checks if c.card == card]
        if mine:
            n = {v: sum(c.verdict == v for c in mine) for v in ("pass", "warn", "fail")}
            L.append(f"| {card} | {CARD_TITLES[card]} | {n['pass']} | {n['warn']} | {n['fail']} | "
                     f"cards/{CARD_FILES[card]} |")
    L += ["", "## Failing and warning checks", "", render_table(checks, only={"fail", "warn"}), "",
          "## All checks", "", render_table(checks), ""]
    if result["info_table"] is not None:
        L += ["## Card A stage table (probe score per stage; d_ = change against the stage's parents)", "",
              "```", result["info_table"].round(3).to_string(index=False), "```", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.doctor", description=__doc__.split("\n\n")[0])
    ap.add_argument("artifact", help="a contract artifact folder, or 'e1' to write and use a synthetic E1")
    ap.add_argument("--out", default="doctor_out")
    ap.add_argument("--cards", default="A,B,C,D")
    ap.add_argument("--full", action="store_true", help="more rows and steps (minutes on a CPU; GPU smoke job)")
    ap.add_argument("--max-rows", type=int, default=None, help="rows per split (evenly strided); default 60000 quick")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="{}", help="JSON model overrides over the default preset")
    ap.add_argument("--train", default="{}", help="JSON TrainSettings overrides")
    ap.add_argument("--architecture", default=None)
    ap.add_argument("--no-preset", action="store_true")
    ap.add_argument("--run-dir", default=None, help="a run folder (with --trained: a finished run to examine)")
    ap.add_argument("--trained", action="store_true", help="restore --run-dir's best weights instead of training")
    ap.add_argument("--include-test", action="store_true", help="card D also scores the test split")
    a = ap.parse_args(argv)
    out = Path(a.out)
    artifact = a.artifact
    if artifact.lower() == "e1":
        artifact = str(out / "e1_artifact")
        write_synthetic(artifact, SyntheticSettings() if a.full else SyntheticSettings(n_coins=16, n_bars=2000),
                        overwrite=True)
    quick = not a.full
    max_rows = a.max_rows if a.max_rows is not None else (60_000 if quick else None)
    t0 = time.monotonic()
    ctx = make_context(artifact, out, run_dir=a.run_dir, model=json.loads(a.model), train=json.loads(a.train),
                       architecture=a.architecture, no_preset=a.no_preset, max_rows=max_rows,
                       epochs=a.epochs or (3 if quick else 10), batch_size=a.batch_size, seed=a.seed, quick=quick,
                       trained=a.trained)
    print(f"context ready in {time.monotonic() - t0:.0f} s ({'restored' if a.trained else 'trained'})", flush=True)
    result = run_cards(ctx, tuple(c.strip().upper() for c in a.cards.split(",")), include_test=a.include_test)
    report = render_report(ctx, result, a.artifact)
    write_report(out, "readiness", result["checks"])
    (out / "readiness.md").write_text(report, encoding="utf-8")
    if result["info_table"] is not None:
        result["info_table"].to_csv(out / "info_flow.csv", index=False)
    print("\n".join(report.split("\n## All checks")[0].splitlines()))
    print(f"\nfull report: {out / 'readiness.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
