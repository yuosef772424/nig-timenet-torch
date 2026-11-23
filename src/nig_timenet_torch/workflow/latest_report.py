"""PURPOSE: The latest-samples report of a battery table: the last N samples per coin in price units, the trades the
    selection keeps (move over uncertainty) and their charts, as CSV / markdown / PNG; a CLI over a run folder.
TAGS: latest report, latest samples, live table, trade selection, entry target chart, cli, report writer, battery table
PITFALLS: pure read of tables (no model); ``stage`` picks the battery table of a finished run (``test`` is the judged
    stage: reading it to tune the selection is the selection bias the protocol exists to stop, so the CLI
    defaults to ``val``); the same function reads the table the live run builds (``workflow.live_run``), whose last row
    has no label. Selection and ranking never use a truth column. Exploratory (rule C5).
"""

import argparse
from pathlib import Path

import pandas as pd

from nig_timenet_torch.eval import latest_plots, latest_table, trade_selection
from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.workflow.battery_run import table_path

__all__ = ["latest_report", "latest_from_folder", "LATEST_CSV", "SELECTION_CSV", "REPORT_MD", "main"]

LATEST_CSV, SELECTION_CSV, REPORT_MD = "latest_table.csv", "latest_selection.csv", "latest_report.md"


def _md(frame: pd.DataFrame) -> str:
    text = latest_table.format_latest_table(frame)
    cols = list(text.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in text.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols), *body])


def latest_report(table: pd.DataFrame, out_dir=None, *, n: int = 5, coins=None, k: float = 1.0,
                  rank_by: str = "combined", top: int | None = None, cost: float = DEFAULT_COST_PER_SIDE,
                  now=None, candles: dict | None = None, charts: bool = True, title: str = "", echo=print,
                  show_latest: bool = True) -> dict:
    """{"latest": the last ``n`` rows per coin, "selection": the rows ``trade_selection.select_trades`` keeps,
    "charts": [PNG paths]}; prints both tables through ``echo`` and, with ``out_dir``, writes ``latest_table.csv``,
    ``latest_selection.csv``, ``latest_report.md`` and the charts (``charts`` False: none; ``candles``: {coin: OHLCV
    frame by bar open time} draws the candles under each chart; ``show_latest`` False prints only the selection)."""
    latest = latest_table.latest_table(table, n=n, coins=coins, now=now)
    selection = trade_selection.select_trades(latest, k=k, rank_by=rank_by, top=top, cost=cost)
    if show_latest:
        latest_table.print_latest_table(latest, title=f"{title} latest {n} samples per coin".strip(), echo=echo)
    latest_table.print_latest_table(selection, title=f"selected: edge > {k} x uncertainty, ranked by {rank_by}",
                                    echo=echo)
    paths: list[Path] = []
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        latest.to_csv(out / LATEST_CSV, index=False)
        selection.to_csv(out / SELECTION_CSV, index=False)
        if charts and len(latest):
            paths = latest_plots.latest_charts(latest, out / "figures", candles, title=title)
        note = (f"Last {n} rows per coin (exploratory, rule C5). Confidence = P(signalled side); uncertainty = "
                "predictive sd of the close return; move and error in percent of the entry; `live` rows have no "
                "truth yet.")
        lines = [f"# Latest samples {title}".strip(), "", note, "", _md(latest), "",
                 f"## Selected: |move| > {k} x uncertainty, ranked by {rank_by}", "", _md(selection), ""]
        (out / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")
    return {"latest": latest, "selection": selection, "charts": paths}


def latest_from_folder(folder, stage: str = "val", out_dir=None, **kw) -> dict:
    """``latest_report`` on the ``stage`` battery table a run wrote into ``folder`` (output next to it by default)."""
    path = table_path(folder, stage)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (was the battery switched off, or is the stage not val / test?)")
    return latest_report(pd.read_parquet(path), folder if out_dir is None else out_dir, title=stage.upper(), **kw)


def main(argv=None) -> None:
    """``python -m nig_timenet_torch.workflow.latest_report FOLDER [--stage val|test] [--n 5] [--coins A,B]
    [--k 1.0] [--rank-by combined] [--top N] [--no-charts] [--out DIR]``"""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("folder")
    ap.add_argument("--stage", default="val", choices=("val", "test"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--coins", default="")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--rank-by", default="combined", choices=trade_selection.RANK_BY)
    ap.add_argument("--top", type=int)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_PER_SIDE)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    latest_from_folder(a.folder, a.stage, a.out, n=a.n, coins=[c for c in a.coins.split(",") if c] or None, k=a.k,
                       rank_by=a.rank_by, top=a.top, cost=a.cost, charts=not a.no_charts)


if __name__ == "__main__":
    main()
