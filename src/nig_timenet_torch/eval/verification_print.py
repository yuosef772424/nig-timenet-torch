"""Verification printing: the sub-reports and the verdict table as the old TF project printed them

PURPOSE: The printed output of ``run_full_verification``: the selective-direction, R:R trades and candle-baseline
    reports and the verdict table with its counts and conclusion, in the old layout and (``lang="ar"``) wording.
TAGS: verification print, selective report, trades report, candle report, verdict table, old project parity
PITFALLS: every function returns the text it prints (so the parity test compares it with the old project's stdout);
    the layout is the old one exactly: float format 3 digits and width 200 / 220 / 250, the pivot of the candle report
    rounded to 4 digits; the label-check line of the candle report is the share of rows whose class label equals
    "future price above the head's last price" (not a model number); the summary's conclusion reads only the verdict
    symbols; a report that was not computed (empty frame) prints nothing.
SOURCE: print statements of crypto-signal-prediction workflow/selective_eval.py, candle_baseline.py, verification.py.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import FUTURE_OF, LAST_COLUMN_INDEX, LAST_OF
from nig_timenet_torch.eval.verification_text import t
from nig_timenet_torch.model.heads import head_spec

__all__ = ["print_selective", "print_trades", "print_candle", "summary_text", "label_check", "PASS", "FAIL"]

PASS, FAIL = "✅", "❌"
RULE = "═" * 100            # the old table's banner rule (═)


def _show(frame: pd.DataFrame, width: int) -> str:
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", width):
        return frame.to_string(index=False)


def print_selective(sel: pd.DataFrame, target_acc: float, lang: str = "en", echo=print) -> str:
    """The selective-direction report as the old ``selective_direction_report`` printed it."""
    text = "\n".join([t(lang, "p_sel_title", acc=target_acc), t(lang, "p_sel_1"), t(lang, "p_sel_2"),
                      _show(sel, 200)])
    echo(text)
    return text


def print_trades(trd: pd.DataFrame, rr: float, stop_mode: str, cost_round_trip_pct: float, lang: str = "en",
                 echo=print) -> str:
    """The R:R trades report as the old ``rr_trading_report`` printed it (``cost_round_trip_pct`` in percent)."""
    text = "\n".join([t(lang, "p_tr_title", rr=rr, stop=stop_mode, cost=cost_round_trip_pct), t(lang, "p_tr_1"),
                      _show(trd, 220)])
    echo(text)
    return text


def label_check(dataset: dict, rows, targets) -> dict:
    """{target: share of rows whose class label equals "the future price is above the head's last price"}, as python
    floats (the old candle report printed this dict)."""
    rows = np.asarray(rows)
    lc = np.asarray(dataset["last_candles"], dtype=np.float64)[rows]
    out = {}
    for tg in targets:
        y = np.asarray(dataset[head_spec("binary_classification", tg).label_key])[rows] > 0
        out[tg] = float(np.mean(y == (lc[:, LAST_COLUMN_INDEX[FUTURE_OF[tg]]] > lc[:, LAST_COLUMN_INDEX[LAST_OF[tg]]])))
    return out


def print_candle(cb: dict, checks: dict | None, lang: str = "en", echo=print) -> str:
    """The candle-baseline report as the old ``candle_baseline_report`` printed it; ``checks`` = {"val": dict, "test":
    dict} of ``label_check`` (None leaves the line out, as the old one did for a non-return target mode)."""
    table = cb["table"]
    lines = [t(lang, "p_cb_title")]
    if checks is not None:
        lines.append(t(lang, "p_cb_check", val=checks["val"], test=checks["test"]))
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        lines.append(table.pivot_table(index=["target", "score"], columns="split",
                                       values=["auc", "accuracy"]).round(4).to_string())
        lines.append(t(lang, "p_cb_maj", d=table.groupby(["target", "split"])["majority_baseline"].first()
                       .round(4).to_dict()))
    for tg, v in cb["verdicts"].items():
        d, lo, hi = v["auc_diff_vs_best"]
        lines.append(t(lang, "p_cb_best", t=tg, best=v["best_baseline"], d=d, lo=lo, hi=hi, days=v["n_test_days"]))
        lines.append(f"   [{tg}] {v['verdict']}")
    text = "\n".join(lines)
    echo(text)
    return text


def summary_text(summary: pd.DataFrame, lang: str = "en", out_dir=None, sections_trade=()) -> str:
    """The verdict table with its banner, counts and conclusion (and, with ``out_dir``, the saved-to line)."""
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        body = summary.to_string(index=False)
    n_ok, n_bad = int((summary["verdict"] == PASS).sum()), int((summary["verdict"] == FAIL).sum())
    trade_ok = summary[summary["section"].isin(list(sections_trade))]["verdict"].eq(PASS).any()
    end = t(lang, "p_end_ok" if trade_ok else "p_end_bad")
    head = f"\n{RULE}\n{t(lang, 'p_banner')}\n{RULE}"
    text = "\n".join([head, body, t(lang, "p_counts", ok=n_ok, bad=n_bad, rest=len(summary) - n_ok - n_bad),
                      f"{t(lang, 'p_concl')} {end}"])
    if out_dir is not None:
        text += "\n" + t(lang, "p_saved", out=out_dir)
    return text
