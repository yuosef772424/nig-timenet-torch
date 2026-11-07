"""Battery run: the one call that turns two stages of a run into the printed/written evaluation battery

PURPOSE: Adapts the VAL and TEST stage (table, labels, anchors, last_close, codec state) to battery tables and runs
    the full battery with the T-profit sections; battery_step does it for a finished Run and writes the reports.
TAGS: battery run, stage inputs, t-profit, battery report, adapter entry point, battery step, profit json
PITFALLS: VAL thresholds are applied to TEST (no leakage), so val and test must be different stages of ONE run;
    n_variants is the number of variants tried so far and widens the CI, state it in the result file.

``battery_step(run)`` (after ``workflow.run.test_step``) writes ``battery_report.txt`` (every section) and
``battery_profit.json`` (the T-profit sections with their numbers and the verdict) into the run's output folder.
``StageInputs`` bundles exactly what ``eval.target_report.target_report`` takes for one stage (plus the optional
``prev_close`` for the persistence baseline). ``run_battery`` returns ``{section: {title, rows, note, ...}}``;
``render`` prints it the way the battery does; ``battery_report.txt`` is written when ``out_dir`` is given.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.eval.battery import full_battery, show
from nig_timenet_torch.eval.battery_adapter import to_battery_table
from nig_timenet_torch.workflow import flat_run, run_common

__all__ = ["StageInputs", "run_battery", "render", "battery_step", "stage_inputs", "profit_json", "REPORT_TXT",
           "PROFIT_JSON"]

REPORT_TXT = "battery_report.txt"
PROFIT_JSON = "battery_profit.json"


@dataclass(frozen=True)
class StageInputs:
    table: pd.DataFrame
    labels: dict
    anchors: dict
    last_close: np.ndarray
    codec_state: dict
    prev_close: np.ndarray | None = None

    def battery_table(self) -> pd.DataFrame:
        return to_battery_table(self.table, self.labels, anchors=self.anchors, last_close=self.last_close,
                                codec_state=self.codec_state, prev_close=self.prev_close)


def render(sections: dict) -> str:
    """The text of every section, in order, exactly as ``battery.show`` prints it."""
    lines: list[str] = []
    for section in sections.values():
        show(section, echo=lines.append)
    return "\n".join(lines) + "\n"


def run_battery(val: StageInputs, test: StageInputs, *, cost: float = 0.0008, n_variants: int = 1,
                out_dir=None) -> dict:
    """The full battery (14 sections, 3 T-profit sections, verdict) for one run; writes ``battery_report.txt``
    into ``out_dir`` when given."""
    sections = full_battery(val.battery_table(), test.battery_table(), cost, n_variants)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / REPORT_TXT).write_text(render(sections))
    return sections


def stage_inputs(run, stage: str) -> StageInputs:
    """The battery's inputs for ONE stage of a finished ``workflow.run.Run`` (the table is predicted once and kept)."""
    tables = run.extra.setdefault("tables", {})
    if stage not in tables:
        tables[stage] = run.table(stage)
    lc = np.asarray(run.dataset["last_candles"])
    idx = run.idx[stage]
    return StageInputs(tables[stage], {k: v.numpy() for k, v in run.y[stage].items()},
                       flat_run.anchors_by_name(lc, idx, run.targets),
                       lc[idx, LAST_COLUMN_INDEX["last_close"]], run_common.codec_state(run.dataset))


def profit_json(sections: dict) -> dict:
    """The T-profit sections (title, rows, note, stats) and the verdict, as plain JSON-able data."""
    keep = {k: v for k, v in sections.items() if k.startswith("profit_") or k == "verdict"}
    return {k: {f: v[f] for f in ("title", "rows", "note", "stats", "best", "tradeable") if f in v}
            for k, v in keep.items()}


def battery_step(run) -> dict:
    """Run the battery on the val and test stage of ``run`` with the run's ``evaluation.battery`` settings; writes
    ``battery_report.txt`` and ``battery_profit.json`` to the run's output folder (when it has one)."""
    b = run.settings.evaluation.battery
    if not b.enabled:
        return {}
    sections = run_battery(stage_inputs(run, "val"), stage_inputs(run, "test"), cost=b.cost_per_side,
                           n_variants=b.n_variants, out_dir=run.out)
    if run.out is not None:
        (run.out / PROFIT_JSON).write_text(json.dumps({"cost_per_side": b.cost_per_side, "n_variants": b.n_variants,
                                                       **profit_json(sections)}, indent=1, default=float),
                                           encoding="utf-8")
    return sections
