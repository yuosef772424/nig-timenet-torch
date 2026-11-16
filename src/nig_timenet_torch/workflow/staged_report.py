"""PURPOSE: The inputs and the report of staged training (ROADMAP ST1): the growth context of the training rows, the
    rough independent-label count of a stage, and the per-stage / per-arm tables as markdown.
TAGS: staged training, stage table, growth context, independent labels, steps to the sign, equal budget comparison
PITFALLS: ``independent_labels`` is the rough estimate of NOTES N-10 (rows x (1 - label overlap) x effective coins /
    coins per timestamp, from doctor.data_checks.overlap on the FULL training rows), a scale for the table, not a
    measurement; the comparison table only means something when every arm ran the same number of optimizer steps on
    the same fixed validation rows (``comparison`` checks the steps and says so when they differ).
"""

import math

import numpy as np

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.trainer.staged import GrowthContext
from nig_timenet_torch.workflow import flat_run

__all__ = ["growth_context", "independent_labels", "stage_table", "comparison", "eval_line"]


def growth_context(dataset: dict, idx: np.ndarray, seed: int = 0) -> GrowthContext:
    """Coin id and time of every TRAIN row ``idx`` (the order of the training tensors)."""
    _, coin = np.unique(flat_run.row_coins(dataset, idx).astype(str), return_inverse=True)
    return GrowthContext(coin=coin.astype(np.int64), time=np.asarray(dataset["last_candles"])[idx, TS_COL].astype(
        np.int64), seed=seed)


def independent_labels(dataset: dict, idx: np.ndarray, stage_rows: list[int]) -> list[int]:
    """Rough number of independent labels in each stage (N-10): rows x (1 - label overlap) x effective coins per
    timestamp. One ``overlap`` call on the full training rows gives the factors."""
    from nig_timenet_torch.doctor.data_checks import overlap  # lazy: doctor imports workflow.run (cycle)

    o = overlap(dataset, idx)
    eff, per = o.get("effective_coins", float("nan")), o.get("coins_per_timestamp", float("nan"))
    coin_factor = eff / per if math.isfinite(eff) and math.isfinite(per) and per > 0 else 1.0
    return [max(1, int(round(r * (1.0 - o["label_overlap"]) * coin_factor))) for r in stage_rows]


def _f(v, fmt=".4f") -> str:
    return "-" if v is None else format(v, fmt)


def stage_table(stages: list[dict]) -> str:
    """The per-stage table of card F improvement 7 (markdown): rows, independent labels, lr, steps to the sign, best
    validation value, gap at the end of the stage, rank IC per head on the fixed validation rows."""
    heads = sorted({h for s in stages for h in s.get("rank_ic", {})})
    lines = ["| stage | rows | indep. labels | lr | steps | steps to sign | ended by | best val | gap at end | "
             + " | ".join(f"rank IC {h}" for h in heads) + " |",
             "|" + "---|" * (9 + len(heads))]
    for s in stages:
        lines.append(f"| {s['stage']} | {s['rows']} | {s.get('independent_labels', '-')} | {s['lr']:.2e} | "
                     f"{s['steps']} | {_f(s['steps_to_sign'], 'd')} | {s['ended_by']} | {_f(s['best_value'])} | "
                     f"{_f(s['gap_at_end'])} | " + " | ".join(_f(s["rank_ic"].get(h)) for h in heads) + " |")
    return "\n".join(lines)


def comparison(arms: dict[str, list[dict]]) -> str:
    """Markdown table of the arms (``staged``, ``b1``, ``b2``, ``b3``), one row per (arm, seed): total steps, best
    validation value over the run, and the test rank IC per head when the entry has it. ``arms[arm]`` is a list of
    dicts with keys ``seed``, ``steps``, ``best_value`` and optional ``test_rank_ic`` ({head: value})."""
    heads = sorted({h for rows in arms.values() for r in rows for h in r.get("test_rank_ic", {})})
    lines = ["| arm | seed | steps | best val | " + " | ".join(f"test rank IC {h}" for h in heads) + " |",
             "|" + "---|" * (4 + len(heads))]
    for arm, rows in arms.items():
        for r in rows:
            lines.append(f"| {arm} | {r['seed']} | {r['steps']} | {_f(r['best_value'])} | "
                         + " | ".join(_f(r.get("test_rank_ic", {}).get(h)) for h in heads) + " |")
    by_seed: dict = {}
    for rows in arms.values():
        for r in rows:
            by_seed.setdefault(r["seed"], set()).add(r["steps"])
    bad = sorted(seed for seed, steps in by_seed.items() if len(steps) > 1)
    note = (f"\n\nWARNING: seed(s) {bad}: the arms did not run the same number of optimizer steps, so they are not "
            f"comparable (card F section 9)." if bad else "")
    return "\n".join(lines) + note


def eval_line(row: dict) -> str:
    """One progress line per evaluation point of a staged run (the Colab output and the notebook)."""
    return (f"stage {row['stage']} step {row['step']} lr {row['lr']:.2e} val_raw {row['val_raw_loss']:.4f} "
            f"probe_raw {row['probe_raw_loss']:.4f} gap {row['gap']:+.4f}" + ("  SIGN" if row["sign"] else ""))
