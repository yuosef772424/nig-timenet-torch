"""PURPOSE: run_chicks(run): the TF chicks report (trust, calibration, trades, patterns, integrity, tearsheet, figures,
    printed text) for a finished Run, written to an output folder; plus the CLI.
TAGS: chicks run, chicks report, run full analysis, tearsheet, trust report, calibration, run step, cli, tf parity,
    notebook step
PITFALLS: the test split is read only here and by test_step / battery_step; calibrators are fitted on the VAL stage
    only (``calibrate``) and never on test; a target mode whose decoded price is not a forecast (magnitude,
    entry_range, '+relative') is skipped with a printed line, as the TF section 6 was; the defaults of
    ``ChicksSettings`` are the TF defaults (calibration isotonic, edge ratio 1.0, top 20, cost 0.08 % round trip,
    Arabic text); ``trade_cost_pct`` is a ROUND-TRIP percent (0.08 = 0.04 % a side, unlike the battery's per-side
    ``cost_per_side``); the report is NOT part of ``run_all``; the CLI rebuilds the run from its settings JSON and the
    run folder's checkpoint (the folder holds no settings), so it needs the artifact the run used.
SOURCE: crypto-signal-prediction ``workflow/run.py::run_chicks`` (the call) and ``evaluation/full_analysis.py``.

``ChicksSettings`` is the strict settings object (unknown keys raise, every field is read); ``run_chicks(run,
settings=None, **changes)`` takes a ChicksSettings and/or keyword changes (``run_chicks(run, calibrate=False,
trade_top_n=10)``). ``stage_data(run, stage)`` is what one stage holds for the report. The report is printed (the
language of ``lang``), the files go to ``out_dir`` (default ``<run.out>/analysis_outputs``) and the result dict of the
TF ``run_full_analysis`` is returned, plus ``files`` (every file written).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nig_timenet_torch.core.config import from_dict, updated
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.calibrator import CALIBRATORS
from nig_timenet_torch.eval.chicks_decode import DIRECTION_TOLERANCE, StageData
from nig_timenet_torch.eval.chicks_full import BATCH_ANALYSIS_SIZE, TRADE_COST_PCT, run_full_analysis
from nig_timenet_torch.eval.chicks_text import LANGS, Say
from nig_timenet_torch.eval.chicks_trades import MIN_EDGE_RATIO, SORT_BY, TOP_N
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import NIG_UNC_MAX
from nig_timenet_torch.workflow import battery_run, flat_run, run_common
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["ChicksSettings", "run_chicks", "stage_data", "stage_from_dataset", "CALIBRATORS_JSON", "OUT_FOLDER",
           "main"]

OUT_FOLDER = "analysis_outputs"
CALIBRATORS_JSON = "confidence_calibrators.json"


@dataclass(frozen=True)
class ChicksSettings:
    """The chicks report's settings; every default is the TF default (``main.ipynb`` section 6)."""
    out_dir: str | None = None           # None = <run.out>/analysis_outputs; a relative name is under run.out
    calibrate: bool = True               # fit the post-hoc confidence calibration on VAL and apply it to TEST
    calibration_method: str = "isotonic"  # eval.calibrator.CALIBRATORS: "isotonic" | "platt"
    market_neutral: bool = False         # tearsheet trade = true change minus the mean of all coins at the same time
    run_trade_selection: bool = True     # the edge-vs-uncertainty trade selection
    trade_min_edge_ratio: float = MIN_EDGE_RATIO   # keep trades with |move| > this x uncertainty
    trade_sort_by: str = "both"          # chicks_trades.SORT_BY: confidence | move | both | edge
    trade_top_n: int | None = TOP_N      # rows of the trade table (None = every qualified trade)
    trade_cost_pct: float = TRADE_COST_PCT   # round-trip cost in percent (0.08 = 0.04 % a side)
    run_integrity_check: bool = True     # naive baseline x lag scan x bias per coin and head
    batch_analysis_size: int = BATCH_ANALYSIS_SIZE   # samples per batch of the batch-regression report
    eval_all_rows: bool = True           # every coin and row of the test split (False: skip coins under 20 rows)
    n_display: int = 1                   # latest samples printed per head and coin (display only)
    range_frac: float = DIRECTION_TOLERANCE   # the "within tolerance" share of the true high - low
    unc_max: float = NIG_UNC_MAX         # cap of the recomputed NIG epistemic / aleatoric (label units)
    class_head: bool = False             # also read the class head's P(up) (pattern feature, calibration fallback)
    make_plots: bool = True              # draw and save the figures
    verbose: bool = True                 # print the report
    lang: str = "ar"                     # printed text: "ar" (the TF text) | "en"
    save_calibrators: bool = True        # write the fitted calibrators' state (rule B4) next to the report

    def __post_init__(self):
        if self.calibration_method not in CALIBRATORS.names():
            raise ValueError(f"calibration_method must be one of {sorted(CALIBRATORS.names())}, "
                             f"got {self.calibration_method!r}")
        if self.trade_sort_by not in SORT_BY:
            raise ValueError(f"trade_sort_by must be one of {SORT_BY}, got {self.trade_sort_by!r}")
        if self.lang not in LANGS:
            raise ValueError(f"lang must be one of {LANGS}, got {self.lang!r}")
        if self.trade_top_n is not None and self.trade_top_n < 1:
            raise ValueError("trade_top_n must be None or >= 1")
        if min(self.trade_min_edge_ratio, self.trade_cost_pct, self.range_frac) < 0 or self.batch_analysis_size < 1:
            raise ValueError("trade_min_edge_ratio, trade_cost_pct, range_frac >= 0 and batch_analysis_size >= 1")
        if self.n_display < 0 or not self.unc_max > 0:
            raise ValueError("n_display >= 0 and unc_max > 0")


def stage_from_dataset(dataset: dict, idx, table, targets, state: dict | None = None) -> StageData:
    """One stage from its parts: the artifact ``dataset``, the row indices ``idx`` of the stage, its PredictionTable
    and the heads in the table (the rows of the table are aligned to ``idx``)."""
    state = run_common.codec_state(dataset) if state is None else state
    idx = np.asarray(idx)
    lc = np.asarray(dataset["last_candles"])
    heads = tuple(t for t in targets if head_spec("nig_regression", t).output_keys["mu"] in table.columns)
    keys = [head_spec("nig_regression", t).label_key for t in heads]
    labels = {k: np.asarray(dataset[k])[idx] for k in keys}
    anchors = flat_run.anchors_by_name(lc, idx, heads, state, dataset=dataset)
    return StageData(table, labels, anchors, lc[idx], state, heads)


def stage_data(run: Run, stage: str) -> StageData:
    """What the report reads of ONE stage of a finished run: its PredictionTable (predicted once, kept in
    ``run.extra['tables']``), labels, anchors, last candles and the codec state."""
    need(run, "result", "train")
    si = battery_run.stage_inputs(run, stage)
    heads = tuple(t for t in run.targets if head_spec("nig_regression", t).output_keys["mu"] in si.table.columns)
    lc = np.asarray(run.dataset["last_candles"])[run.idx[stage]]
    return StageData(si.table, si.labels, si.anchors, lc, si.codec_state, heads)


def _out_dir(run: Run, s: ChicksSettings) -> str:
    name = OUT_FOLDER if s.out_dir is None else s.out_dir
    return str(Path(name) if run.out is None or Path(name).is_absolute() else Path(run.out) / name)


def _write_calibrators(calibrators: dict | None, out_dir: str) -> str | None:
    """The calibrators' plain state (method, signal, the fitted state, the val diagnostics) as JSON."""
    if not calibrators:
        return None
    path = Path(out_dir) / CALIBRATORS_JSON
    path.write_text(json.dumps(calibrators, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    return str(path)


def run_chicks(run: Run, settings: ChicksSettings | None = None, **changes) -> dict | None:
    """The chicks report of a finished ``run`` (see the module doc). Returns the ``run_full_analysis`` dict plus
    ``out_dir``, ``files`` and ``settings``; None when the run's target mode does not decode to a price forecast."""
    s = updated(settings or ChicksSettings(), changes)
    say = Say(s.lang)
    state = run_common.codec_state(run.dataset)
    if not TARGET_CODECS[state["mode"]]().forecasts_price(state):
        name = state.get("target_mode", state["mode"])
        say("chicks_skipped", mode=name)
        run.extra["chicks_skipped"] = name
        return None
    if not len(run.idx["test"]):
        raise ValueError("the split has no test rows")
    out_dir = _out_dir(run, s)
    test = stage_data(run, "test")
    val = stage_data(run, "val") if s.calibrate and len(run.idx["val"]) else None
    res = run_full_analysis(
        test, out_dir=out_dir, val_stage=val, calibration_method=s.calibration_method,
        batch_analysis_size=s.batch_analysis_size, make_plots=s.make_plots, run_integrity_check=s.run_integrity_check,
        verbose=s.verbose, n_display=s.n_display, run_trade_selection=s.run_trade_selection,
        trade_min_edge_ratio=s.trade_min_edge_ratio, trade_sort_by=s.trade_sort_by, trade_top_n=s.trade_top_n,
        trade_cost_pct=s.trade_cost_pct, pnl_market_neutral=s.market_neutral, eval_all_rows=s.eval_all_rows,
        range_frac=s.range_frac, unc_max=s.unc_max, class_head=s.class_head, say=say)
    if s.save_calibrators:
        _write_calibrators(res.get("calibrators"), out_dir)
    res.update(out_dir=out_dir, settings=s,
               files=sorted(str(p.relative_to(out_dir)) for p in Path(out_dir).rglob("*") if p.is_file()))
    run.extra["chicks"] = res
    return res


def main(argv=None) -> int:
    """``python -m nig_timenet_torch.workflow.chicks_run RUN_FOLDER --settings run_settings.json [--out DIR]
    [--set key=json ...]``: rebuild the run from its RunSettings JSON, restore the weights of RUN_FOLDER (a run's
    train folder holding ``best.pt``), and write the chicks report."""
    from nig_timenet_torch.workflow import run as steps
    from nig_timenet_torch.workflow.settings import RunSettings

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("folder", help="the finished run's train folder (checkpoint.pt, best.pt)")
    ap.add_argument("--settings", required=True, help="the run's RunSettings as JSON (core.config.to_dict)")
    ap.add_argument("--out", default=None, help="the run's output folder (the report goes to <out>/analysis_outputs)")
    ap.add_argument("--weights", default="best", choices=("best", "last"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="a ChicksSettings change, e.g. --set calibrate=false --set lang='\"en\"'")
    a = ap.parse_args(argv)
    changes = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    settings = from_dict(RunSettings, json.loads(Path(a.settings).read_text(encoding="utf-8")))
    run = steps.build(steps.prepare(settings, a.out))
    steps.restore(run, weights=a.weights, from_dir=a.folder)
    run_chicks(run, **changes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
