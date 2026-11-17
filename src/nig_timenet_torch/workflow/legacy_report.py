"""PURPOSE: The TF evaluation reports and charts that had no counterpart here, for a finished run: reads the run's
    battery tables (VAL, TEST), writes legacy_report.md / .json, the figures and the worst / best samples as a
    sample_charts CSV.
TAGS: legacy report, tf evaluation, tearsheet, equity chart, portfolio chart, best worst predictions, calibration,
    lag scan, report writer, cli
PITFALLS: reads ``battery_table_{val,test}.parquet`` (written by ``workflow.battery_run.battery_step``): a run with the
    battery switched off has none, so this report cannot be made for it. Off the run path: nothing in the default run
    calls it (``python -m nig_timenet_torch.workflow.legacy_report FOLDER``). ``plots`` needs matplotlib (extra
    ``plots``). Exploratory (rule C5); the samples CSV feeds ``eval/sample_charts.py``, whose candles need the network.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from nig_timenet_torch.eval import legacy_reports as lr
from nig_timenet_torch.workflow.battery_run import table_path

__all__ = ["stage_report", "legacy_report", "legacy_from_folder", "render", "REPORT_MD", "REPORT_JSON", "SAMPLES_CSV",
           "main"]

REPORT_MD, REPORT_JSON, SAMPLES_CSV = "legacy_report.md", "legacy_report.json", "legacy_samples.csv"
_TEAR = ("n", "win_rate", "mean", "median", "sharpe", "sortino", "omega", "profit_factor", "sqn", "payoff",
         "max_win_streak", "max_loss_streak", "final_equity", "max_drawdown")


def stage_report(table: pd.DataFrame, cost: float, top_share: float = lr.TOP_SHARE, n_samples: int = 10) -> dict:
    """Every legacy report of one stage's battery table (frames kept as frames for the figures)."""
    tr = lr.trades(table, cost, top_share)
    daily = lr.daily_portfolio(tr)
    return {"trade_rows": tr, "daily": daily, "tearsheet_trades": lr.tearsheet(tr["net"].to_numpy()),
            "tearsheet_days": lr.tearsheet(daily["ret"].to_numpy(), risk=1.0), "integrity": lr.integrity(table),
            "calibration": lr.calibration(table), "conditioned": lr.conditioned(table),
            "coins": lr.coin_ranking(table), "extremes": lr.extreme_samples(tr, n_samples)}


def _jsonable(rep: dict) -> dict:
    out = {}
    for k, v in rep.items():
        if k in ("trade_rows",):
            continue
        if isinstance(v, pd.DataFrame):
            out[k] = json.loads(v.reset_index().to_json(orient="records", date_format="iso"))
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _md(frame: pd.DataFrame) -> str:
    """A markdown table of ``frame`` (no tabulate dependency)."""
    cols = list(frame.columns)
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols), *body])


def render(reports: dict, cost: float, top_share: float) -> str:
    """The markdown report of ``legacy_report``'s stages."""
    lines = ["# Legacy evaluation report (TF reports ported; exploratory, rule C5)", "",
             f"Trades: the top {top_share:.0%} |mu| rows per stage, side from p_dir, net of {cost} per side. Per-trade "
             "numbers are unannualised; the daily portfolio splits the whole capital over each day's trades.", ""]
    for stage, rep in reports.items():
        lines += [f"## {stage.upper()}", "", "| tearsheet | " + " | ".join(_TEAR) + " |",
                  "|---|" + "---|" * len(_TEAR)]
        for name in ("tearsheet_trades", "tearsheet_days"):
            lines.append(f"| {name.split('_')[1]} | " + " | ".join(_fmt(rep[name].get(k)) for k in _TEAR) + " |")
        integ, cal = rep["integrity"], rep["calibration"]
        lines += ["", "Integrity: " + ", ".join(f"{k} {_fmt(v)}" for k, v in integ.items() if k != "lag_scan"),
                  "", "Lag scan (mean r per coin of mu(t) with y(t+k)): " + ", ".join(
                      f"k={r['lag']}: {_fmt(r['mean_r'])}" for r in integ["lag_scan"]),
                  "", f"Calibration of p_dir: ECE {_fmt(cal['ece'])}, Brier {_fmt(cal['brier'])} (base rate "
                  f"{_fmt(cal['brier_base_rate'])}), AUC {_fmt(cal['auc'])}, rows {cal['rows']}", ""]
        for key, frame in rep["conditioned"].items():
            lines += [f"Hit rate {key.replace('_', ' ')}:", "", _md(frame), ""]
        coins = rep["coins"]
        lines += [f"Best coins (of {len(coins)}):", "", _md(coins.head(5)), "",
                  "Worst coins:", "", _md(coins.tail(5)), ""]
        for key in ("worst", "best"):
            lines += [f"{key.capitalize()} trades:", "", _md(rep["extremes"][key]),
                      ""]
    return "\n".join(lines) + "\n"


def _samples_csv(reports: dict) -> pd.DataFrame:
    rows = []
    for stage, rep in reports.items():
        for key in ("worst", "best"):
            for r in rep["extremes"][key].to_dict("records"):
                rows.append({"coin": r["coin"], "date": pd.Timestamp(r["end_time"]).isoformat(),
                             "side": "long" if r["side"] > 0 else "short", "pnl": 100 * r["net"],
                             "source": f"{stage} {key} trade", "note": f"mu {r['mu']:+.4f}", "time": True})
    return pd.DataFrame(rows, columns=["coin", "date", "side", "pnl", "source", "note", "time"])


def legacy_report(tables: dict, out_dir=None, *, cost: float = 0.0008, top_share: float = lr.TOP_SHARE,
                  plots: bool = True) -> dict:
    """``tables`` = {stage: battery table}; writes the markdown, the JSON, the samples CSV and (``plots``) the figures
    into ``out_dir`` (None = write nothing). Returns {stage: report}."""
    reports = {stage: stage_report(t, cost, top_share) for stage, t in tables.items()}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / REPORT_MD).write_text(render(reports, cost, top_share), encoding="utf-8")
        (out / REPORT_JSON).write_text(json.dumps({s: _jsonable(r) for s, r in reports.items()}, indent=1,
                                                  default=str), encoding="utf-8")
        _samples_csv(reports).to_csv(out / SAMPLES_CSV, index=False)
        if plots:
            from nig_timenet_torch.eval.legacy_plots import figures
            for stage, rep in reports.items():
                figures(stage, tables[stage], rep, out / "figures")
    return reports


def legacy_from_folder(folder, out_dir=None, **kw) -> dict:
    """``legacy_report`` on the battery tables a run wrote into ``folder`` (output next to them by default)."""
    tables = {st: pd.read_parquet(table_path(folder, st)) for st in ("val", "test") if table_path(folder, st).is_file()}
    if not tables:
        raise FileNotFoundError(f"no battery_table_{{val,test}}.parquet in {folder} (was the battery switched off?)")
    return legacy_report(tables, folder if out_dir is None else out_dir, **kw)


def main(argv=None) -> None:
    """``python -m nig_timenet_torch.workflow.legacy_report FOLDER [--out DIR] [--cost C] [--top-share S]``"""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("folder")
    ap.add_argument("--out")
    ap.add_argument("--cost", type=float, default=0.0008)
    ap.add_argument("--top-share", type=float, default=lr.TOP_SHARE)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    legacy_from_folder(a.folder, a.out, cost=a.cost, top_share=a.top_share, plots=not a.no_plots)
    print(f"wrote {REPORT_MD}, {REPORT_JSON}, {SAMPLES_CSV} into {a.out or a.folder}")


if __name__ == "__main__":
    main()
