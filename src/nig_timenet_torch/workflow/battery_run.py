"""Battery run: the one call that turns two stages of a run into the printed/written evaluation battery

PURPOSE: Adapts the VAL and TEST stage (table, labels, anchors, last_close, codec state) to battery tables and runs
    the full battery with the T-profit sections; optionally writes the text report next to the run.
TAGS: battery run, stage inputs, t-profit, battery report, adapter entry point
PITFALLS: VAL thresholds are applied to TEST (no leakage), so val and test must be different stages of ONE run;
    n_variants is the number of variants tried so far and widens the CI, state it in the result file.

``StageInputs`` bundles exactly what ``eval.target_report.target_report`` takes for one stage (plus the optional
``prev_close`` for the persistence baseline). ``run_battery`` returns ``{section: {title, rows, note, ...}}``;
``render`` prints it the way the battery does; ``battery_report.txt`` is written when ``out_dir`` is given.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery import full_battery, show
from nig_timenet_torch.eval.battery_adapter import to_battery_table

__all__ = ["StageInputs", "run_battery", "render", "REPORT_TXT"]

REPORT_TXT = "battery_report.txt"


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
