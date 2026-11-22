"""PURPOSE: The panel step of a finished run: the day-metric frame of a stage (battery table + trailing 24 h return
    + last-bar asymmetry) and panel_step, which writes the day metrics, the uncertainty report and the frames.
TAGS: panel report, day metrics frame, winning days, trailing return, last bar asymmetry, uncertainty use, oracle
    frame, run step, panel frame parquet
PITFALLS: the trailing return of a row is read from the FULL artifact (the coin's close ``bars_per_day`` bars before),
    not from the stage rows: a strided stage (``max_rows``) has no consecutive timestamps; a row whose earlier bar
    is missing gets NaN (its coin does not trade the momentum baselines that day). VAL fits what needs fitting
    (baseline orientation, confidence thresholds); TEST is only scored. ``oracle_frame`` needs an environment with
    ``oracle_*`` columns (E1 / E4): it is the known answer, never a model.

``panel_step(run)`` (after ``workflow.run.test_step``) writes ``panel_report.json`` / ``panel_report.md`` and
``panel_frame_{val,test}.parquet`` into the run's output folder; ``panel_from_folder`` recomputes the report from the
two frames alone (other cost, other top share), no model and no artifact needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TS_COL
from nig_timenet_torch.data_io.synthetic import oracle_keys
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval import day_metrics, uncertainty_use
from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.workflow import battery_run, flat_run, run_common

__all__ = ["trailing_return", "stage_frame", "oracle_frame", "panel_report", "panel_step", "panel_from_folder",
           "panel_from_folders", "frame_path", "REPORT_JSON", "REPORT_MD"]

REPORT_JSON, REPORT_MD = "panel_report.json", "panel_report.md"
_KEEP = ("mu", "sigma", "p_dir", "y_ret", "coin", "pred_high", "pred_low", "fut_high", "fut_low", "sigma_high",
         "sigma_low")


def trailing_return(dataset: dict, idx: np.ndarray, bars: int) -> np.ndarray:
    """close(t) / close(t - ``bars`` bars) - 1 of the rows ``idx``, the earlier close read from the same coin's rows
    of the whole dataset; NaN when that bar is not in the artifact."""
    lc = np.asarray(dataset["last_candles"])
    ts, close = lc[:, TS_COL].astype(np.int64), lc[:, LAST_COLUMN_INDEX["last_close"]]
    want = ts - bars * int(pd.Timedelta(dataset["base_tf"]).value)
    out = np.full(len(ts), np.nan)
    for b in dataset["asset_bounds"]:
        sl = slice(b["start"], b["end"])
        t = ts[sl]
        j = np.searchsorted(t, want[sl])
        ok = (j < len(t)) & (t[np.minimum(j, len(t) - 1)] == want[sl])
        prev = close[sl][np.minimum(j, len(t) - 1)]
        out[sl] = np.where(ok, close[sl] / prev - 1.0, np.nan)
    return out[np.asarray(idx)]


def _extras(dataset: dict, idx: np.ndarray) -> dict:
    lc = np.asarray(dataset["last_candles"])[np.asarray(idx)]
    close = lc[:, LAST_COLUMN_INDEX["last_close"]]
    bars_per_day = max(1, int(round(pd.Timedelta("1D") / pd.Timedelta(dataset["base_tf"]))))
    return {"ret_1d": trailing_return(dataset, idx, bars_per_day),
            "asym": np.log(lc[:, LAST_COLUMN_INDEX["last_high"]] / close)
            + np.log(lc[:, LAST_COLUMN_INDEX["last_low"]] / close)}


def stage_frame(run, stage: str) -> pd.DataFrame:
    """The day-metric frame of one stage of a finished Run (``eval.day_metrics.FRAME_COLUMNS`` + the battery's
    p_dir and high / low columns)."""
    bt = battery_run.stage_inputs(run, stage).battery_table()
    frame = bt[[c for c in _KEEP if c in bt.columns]].copy()
    frame.insert(0, "time", bt["end_time"].to_numpy())
    for k, v in _extras(run.dataset, run.idx[stage]).items():
        frame[k] = v
    return frame


def oracle_frame(run, stage: str) -> pd.DataFrame:
    """The frame of the KNOWN answer of a synthetic environment: mu / sigma = the close label's exact conditional
    mean / sd (``oracle_*`` columns), decoded like a model's; the realised columns are the stage's own."""
    ds, idx = run.dataset, run.idx[stage]
    mean_key, sd_key = oracle_keys("close")
    if mean_key not in ds:
        raise KeyError("oracle_frame: the dataset has no oracle columns (E1 / E4 only)")
    frame = stage_frame(run, stage)
    state = run_common.codec_state(ds)
    codec = TARGET_CODECS[state["mode"]]()
    lc = np.asarray(ds["last_candles"])
    anchor = flat_run.anchors_by_name(lc, idx, ("close",), state)["close"]
    last_close = lc[idx, LAST_COLUMN_INDEX["last_close"]]
    mean, sd = np.asarray(ds[mean_key], np.float64)[idx], np.asarray(ds[sd_key], np.float64)[idx]
    dec = lambda v: codec.decode(v, state, anchor)  # noqa: E731
    frame["mu"] = dec(mean) / last_close - 1.0
    frame["sigma"] = (dec(mean + sd) - dec(mean)) / last_close
    return frame[[c for c in frame.columns if not c.startswith(("pred_", "fut_", "sigma_"))]]


def panel_report(val: pd.DataFrame, test: pd.DataFrame, *, cost: float = DEFAULT_COST_PER_SIDE, top_share: float = 0.2,
                 seed: int = 0) -> dict:
    """{"val", "test": day_report, "uncertainty": uncertainty_report of TEST with VAL thresholds}."""
    return {"val": day_metrics.day_report(val, None, cost=cost, top_share=top_share, seed=seed),
            "test": day_metrics.day_report(test, val, cost=cost, top_share=top_share, seed=seed),
            "uncertainty": uncertainty_use.uncertainty_report(test, val, cost=cost)}


def render(report: dict, title: str = "") -> str:
    return "\n".join([f"## {title}".rstrip(), "", day_metrics.render(report["test"], "TEST"),
                      day_metrics.render(report["val"], "VAL"), "### Uncertainty use (TEST, thresholds from VAL)", "",
                      uncertainty_use.render(report["uncertainty"])])


def frame_path(folder, stage: str) -> Path:
    return Path(folder) / f"panel_frame_{stage}.parquet"


def _write(out: Path, report: dict, title: str) -> None:
    run_common.write_json(out / REPORT_JSON, report)
    (out / REPORT_MD).write_text(render(report, title), encoding="utf-8")


def panel_step(run, *, top_share: float = 0.2) -> dict:
    """The panel report of a finished run (cost = the run's battery cost per side); written to the run's output
    folder with the two frames when it has one. The numbers land in ``run.extra['panel']`` (the summary keeps the
    headline rows)."""
    val, test = stage_frame(run, "val"), stage_frame(run, "test")
    cost = run.settings.evaluation.battery.cost_per_side
    report = panel_report(val, test, cost=cost, top_share=top_share, seed=run.settings.train.seed)
    run.extra["panel"] = headline(report)
    if run.out is not None:
        val.to_parquet(frame_path(run.out, "val"), index=False)
        test.to_parquet(frame_path(run.out, "test"), index=False)
        _write(run.out, report, f"Panel report, seed {run.settings.train.seed}")
    return report


def headline(report: dict) -> dict:
    """The few numbers of a panel report a summary table shows (TEST): per metric the model's winning-day share
    with its CI, the mean gross / net day, whether it beats every baseline, the best baseline's share; the relative
    rank IC; the sd-vs-error rank correlations of the close head."""
    out = {}
    for metric in ("absolute", "neutral"):
        sec = report["test"][metric]
        m = sec["model"]
        base = {k: v["gross"].get("win_share", float("nan")) for k, v in sec.items() if k != "model"}
        out[metric] = {"days": m["gross"].get("n_days", 0), "win_share": m["gross"].get("win_share"),
                       "win_lo": m["gross"].get("win_lo"), "win_hi": m["gross"].get("win_hi"),
                       "mean_gross": m["gross"].get("mean"), "mean_net": m["net"].get("mean"),
                       "beats_every_baseline": m["beats_every_baseline"], "baseline_win_share": base}
    out["relative_rank_ic"] = report["test"]["relative_rank_ic"]["model"]
    out["sd_ranks_error_close"] = report["uncertainty"]["sd_ranks_error"]["close"]
    return out


def panel_from_folder(folder, *, cost: float = DEFAULT_COST_PER_SIDE, top_share: float = 0.2, out_dir=None) -> dict:
    """The panel report of a FINISHED run folder from its two frames (no model, no artifact)."""
    missing = [str(frame_path(folder, st)) for st in ("val", "test") if not frame_path(folder, st).is_file()]
    if missing:
        raise FileNotFoundError(f"not a finished panel run folder (no frames): {missing}")
    report = panel_report(pd.read_parquet(frame_path(folder, "val")), pd.read_parquet(frame_path(folder, "test")),
                          cost=cost, top_share=top_share)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        _write(Path(out_dir), report, "Panel report")
    return report


def panel_from_folders(folders, *, cost: float = DEFAULT_COST_PER_SIDE, top_share: float = 0.2, out_dir=None) -> dict:
    """ONE panel report over the frames of several run folders whose test windows do not overlap (the folds of a
    rolling study, ``workflow/rolling_run.py``): the TEST frames are concatenated (every day of every fold is scored
    by the model that was trained before it), and so are the VAL frames (they orient the asymmetry baseline and fit
    the confidence thresholds). Folders without frames are skipped; none with frames raises."""
    have = [f for f in folders if frame_path(f, "test").is_file() and frame_path(f, "val").is_file()]
    if not have:
        raise FileNotFoundError("no folder holds panel frames (run with evaluation.battery.day_metrics)")
    val, test = (pd.concat([pd.read_parquet(frame_path(f, st)) for f in have], ignore_index=True)
                 for st in ("val", "test"))
    if test.duplicated(["time", "coin"]).any():
        raise ValueError("the test frames overlap (a row is scored twice): pool only folds with disjoint test windows")
    report = panel_report(val, test, cost=cost, top_share=top_share)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        _write(Path(out_dir), report, f"Panel report, {len(have)} folders pooled")
    return report
