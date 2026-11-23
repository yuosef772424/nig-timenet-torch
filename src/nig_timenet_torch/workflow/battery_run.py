"""PURPOSE: Adapts the VAL and TEST stage (table, labels, anchors, last_close, codec state) to battery tables and runs
    the full battery with the T-profit sections; battery_step does it for a finished Run and writes the reports.
TAGS: battery run, stage inputs, t-profit, battery report, adapter entry point, battery step, profit json
PITFALLS: battery_step skips (and says so in run.extra['battery_skipped']) a run whose target mode does not decode
    to a price forecast (magnitude, entry_range, '+relative'); VAL thresholds are applied to TEST (no leakage), so
    val and test must be different stages of ONE run; n_variants is the number of variants tried so far and widens
    the CI, state it in the result file; stage_atr reads only bars closed by the prediction time (window high / low,
    else the last NATR_14), battery tables saved before 2026-11-22 have no atr column (the take-profit-at-ATR section
    is skipped on them); account_from_folders puts the TEST of SEVERAL fold folders in ONE account (each fold's own
    VAL threshold): the folds must not overlap.

With ``evaluation.battery.day_metrics`` (D-032, off by default) ``battery_step`` also writes the day metrics of a
multi-coin run (``workflow.panel_report.panel_step``: ``panel_report.json`` / ``.md`` and the two frames).
``battery_step(run)`` (after ``workflow.run.test_step``) writes ``battery_report.txt`` (every section),
``battery_profit.json`` (the T-profit sections with their numbers and the verdict) and the two battery tables
(``battery_table_val.parquet`` / ``battery_table_test.parquet``) into the run's output folder. ``battery_from_folder``
re-runs the battery from those tables alone (any model that writes them, other costs or more variants), no model needed;
``account_from_folders`` runs the capital account (``eval.account``) over the TEST of several fold folders (a rolling
run), the view of the audit scripts a3-a6: ``python -m nig_timenet_torch.workflow.battery_run <fold folders...>``.
``StageInputs`` bundles exactly what ``eval.target_report.target_report`` takes for one stage (plus the optional
``prev_close`` for the persistence baseline). ``run_battery`` returns ``{section: {title, rows, note, ...}}``;
``render`` prints it the way the battery does; ``battery_report.txt`` is written when ``out_dir`` is given.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data.scaling import PERCENT_SCALE
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.account import account_section
from nig_timenet_torch.eval.battery import full_battery, show
from nig_timenet_torch.eval.battery_adapter import to_battery_table
from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.eval.vol_baseline import window_ranges
from nig_timenet_torch.workflow import flat_run, run_common

__all__ = ["StageInputs", "run_battery", "run_tables", "render", "battery_step", "battery_from_folder", "stage_inputs",
           "account_from_folders", "main",
           "stage_atr", "NATR_FEATURE",
           "profit_json", "table_path", "REPORT_TXT", "PROFIT_JSON"]

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
    atr: np.ndarray | None = None              # mean true range of the last 14 input bars (log units per bar)

    def battery_table(self) -> pd.DataFrame:
        return to_battery_table(self.table, self.labels, anchors=self.anchors, last_close=self.last_close,
                                codec_state=self.codec_state, prev_close=self.prev_close, atr=self.atr)


def render(sections: dict) -> str:
    """The text of every section, in order, exactly as ``battery.show`` prints it."""
    lines: list[str] = []
    for section in sections.values():
        show(section, echo=lines.append)
    return "\n".join(lines) + "\n"


def table_path(folder, stage: str) -> Path:
    return Path(folder) / f"battery_table_{stage}.parquet"


def run_battery(val: StageInputs, test: StageInputs, *, cost: float = DEFAULT_COST_PER_SIDE, n_variants: int = 1,
                out_dir=None) -> dict:
    """The full battery (14 sections, 3 T-profit sections, verdict) for one run; writes ``battery_report.txt``
    into ``out_dir`` when given. ``cost`` is charged per side of a trade."""
    return run_tables(val.battery_table(), test.battery_table(), cost=cost, n_variants=n_variants, out_dir=out_dir)


def run_tables(val: pd.DataFrame, test: pd.DataFrame, *, cost: float = DEFAULT_COST_PER_SIDE, n_variants: int = 1,
               out_dir=None, trim: str = "best", drop_flat: bool = True, account_books: int = 0) -> dict:
    """The same from two ready battery tables (``eval.battery_adapter.to_battery_table`` output); ``trim``,
    ``drop_flat`` and ``account_books`` (0 = read the horizon) as in ``eval.battery.full_battery``."""
    sections = full_battery(val, test, cost, n_variants, trim=trim, drop_flat=drop_flat,
                            account_books=account_books or None)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / REPORT_TXT).write_text(render(sections))
    return sections


#: the input channel the ATR falls back to when the window has no high / low channels: pandas_ta's NATR (Wilder ATR /
#: close, in percent), stored / ``data.scaling.PERCENT_SCALE``
NATR_FEATURE = "NATR_14"


def stage_atr(dataset: dict, idx, timeframes=()) -> np.ndarray | None:
    """The ATR of each row in log units per bar, from bars closed by the row's prediction time only: the mean true
    range of the last 14 bars of its input window when the window has high / low price channels
    (``eval.vol_baseline.window_ranges``), else the window's last ``NATR_14`` (Wilder's ATR over the close, the TF
    feature set the artifacts carry), else None (the take-profit-at-ATR section then says it cannot run)."""
    key = run_common.input_keys(dataset, timeframes)[0]
    fo, rows = list(dataset["feature_order"]), np.asarray(idx)
    try:
        return window_ranges(np.asarray(dataset[key])[rows], fo)["atr14"]
    except ValueError:          # no high / low channel, or the channels are not log(price / prev close)
        pass
    if NATR_FEATURE in fo:
        natr = np.asarray(dataset[key])[rows, -1, fo.index(NATR_FEATURE)].astype(np.float64) * PERCENT_SCALE / 100.0
        return np.log1p(np.clip(natr, 0.0, None))
    return None


def stage_inputs(run, stage: str) -> StageInputs:
    """The battery's inputs for ONE stage of a finished ``workflow.run.Run`` (the table is predicted once and kept)."""
    tables = run.extra.setdefault("tables", {})
    if stage not in tables:
        tables[stage] = run.table(stage)
    lc = np.asarray(run.dataset["last_candles"])
    idx = run.idx[stage]
    return StageInputs(tables[stage], {k: v.numpy() for k, v in run.y[stage].items()},
                       flat_run.anchors_by_name(lc, idx, run.targets, run_common.codec_state(run.dataset),
                                                dataset=run.dataset),
                       lc[idx, LAST_COLUMN_INDEX["last_close"]], run_common.codec_state(run.dataset),
                       atr=stage_atr(run.dataset, idx, run.settings.data.input_timeframes))


def profit_json(sections: dict) -> dict:
    """The T-profit sections (title, rows, note, stats), the verdict and the scorecard verdict, as plain JSON-able
    data."""
    keep = {k: v for k, v in sections.items()
            if k.startswith("profit_") or k in ("verdict", "scorecard_verdict", "account")}
    return {k: {f: v[f] for f in ("title", "rows", "note", "stats", "best", "tradeable", "passed") if f in v}
            for k, v in keep.items()}


def battery_step(run) -> dict:
    """Run the battery on the val and test stage of ``run`` with the run's ``evaluation.battery`` settings; writes
    ``battery_report.txt`` and ``battery_profit.json`` to the run's output folder (when it has one)."""
    b = run.settings.evaluation.battery
    state = run_common.codec_state(run.dataset)
    if not TARGET_CODECS[state["mode"]]().forecasts_price(state):   # magnitude / entry_range / '+relative'
        run.extra["battery_skipped"] = (f"target mode {state['target_mode']!r}: its decoded price is not a forecast "
                                        "(no sign or no market level); the trading rules need a signed target")
        print(f"battery and panel report skipped: {run.extra['battery_skipped']}", flush=True)
        return {}
    if b.day_metrics:
        from nig_timenet_torch.workflow.panel_report import panel_step  # it imports this module

        panel_step(run)
    if not b.enabled:
        return {}
    val, test = stage_inputs(run, "val").battery_table(), stage_inputs(run, "test").battery_table()
    sections = run_tables(val, test, cost=b.cost_per_side, n_variants=b.n_variants, out_dir=run.out,
                          trim=b.verdict_trim, drop_flat=b.drop_flat, account_books=b.account_books)
    if run.out is not None:
        val.to_parquet(table_path(run.out, "val"), index=False)
        test.to_parquet(table_path(run.out, "test"), index=False)
        (run.out / PROFIT_JSON).write_text(json.dumps({"cost_per_side": b.cost_per_side, "n_variants": b.n_variants,
                                                       **profit_json(sections)}, indent=1, default=float),
                                           encoding="utf-8")
    return sections


def _tables(folder) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [str(table_path(folder, st)) for st in ("val", "test") if not table_path(folder, st).is_file()]
    if missing:
        raise FileNotFoundError(f"not a finished run folder (no battery tables): {missing}")
    return pd.read_parquet(table_path(folder, "val")), pd.read_parquet(table_path(folder, "test"))


def battery_from_folder(folder, *, cost: float = DEFAULT_COST_PER_SIDE, n_variants: int = 1, out_dir=None,
                        trim: str = "best", drop_flat: bool = True, account_books: int = 0) -> dict:
    """The battery of a FINISHED run folder: reads its two battery tables, no model and no artifact needed (any model
    that writes ``battery_table_{val,test}.parquet`` can use it). ``out_dir`` None = write nothing."""
    val, test = _tables(folder)
    sections = run_tables(val, test, cost=cost, n_variants=n_variants, out_dir=out_dir, trim=trim,
                          drop_flat=drop_flat, account_books=account_books)
    if out_dir is not None:
        (Path(out_dir) / PROFIT_JSON).write_text(json.dumps({"cost_per_side": cost, "n_variants": n_variants,
                                                             **profit_json(sections)}, indent=1, default=float),
                                                 encoding="utf-8")
    return sections


def account_from_folders(folders, *, cost: float = DEFAULT_COST_PER_SIDE, account_books: int = 0, **kw) -> dict:
    """The ACCOUNT section (``eval.account.account_section``; ``kw`` its other options) over the TEST tables of several
    finished fold folders in time order, each fold's long book cut at its own VAL quantile: the account of a rolling
    run (the audit's a3-a6 on rolling-dw1-r-001 were this over the 7 folds of one seed)."""
    return account_section([_tables(f) for f in folders], cost, n_books=account_books or None, **kw)


def main(argv=None) -> int:
    """Print the ACCOUNT section of fold folders: ``python -m nig_timenet_torch.workflow.battery_run F0 F1 ...``."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("folders", nargs="+", help="finished run folders with battery_table_{val,test}.parquet")
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_PER_SIDE, help="cost per side")
    ap.add_argument("--books", type=int, default=0, help="sub-books (0 = read the label horizon from the table)")
    ap.add_argument("--placebo", type=int, default=None, help="placebo draws (default eval.account.N_PLACEBO)")
    a = ap.parse_args(argv)
    kw = {} if a.placebo is None else {"n_placebo": a.placebo}
    print(render({"account": account_from_folders(a.folders, cost=a.cost, account_books=a.books, **kw)}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
