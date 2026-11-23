"""Parity of the torch ``run_full_verification(match_old=True)`` with the OLD project's, number by number.

The golden files (tests/parity/verification_golden/<case>/) were written by the old project's own code
(``python -m tests.parity.make_verification_golden``) on the arrays of ``tests.parity.old_world``; this test rebuilds
the same arrays, runs the torch verification in match mode and compares the printed text, the verdict table, every
sub-report, the candle verdicts and the saved files. Anything that differs, or any section missing, fails.
Two cases: daily bars (one timestamp per day) and 6 h bars (four per day: the old bootstrap unit, the raw timestamp,
and the torch default, the UTC day, then give different intervals, so the match mode is really exercised)."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import verification as ver
from nig_timenet_torch.eval.verification_text import OLD_COST_PER_SIDE, Variant
from tests.eval.verification_world import TARGETS, make_world
from tests.parity.old_world import CASES, MIN_N, OLD_ARGS

GOLDEN = Path(__file__).parent / "verification_golden"
pytestmark = pytest.mark.skipif(not (GOLDEN / "daily" / "summary.csv").is_file(), reason="no golden files")


def old_read(case: str, name: str) -> pd.DataFrame:
    return pd.read_csv(GOLDEN / case / name, dtype=str, keep_default_na=False)


def same_frame(new: pd.DataFrame, old: pd.DataFrame, what: str, rel: float = 1e-9) -> None:
    """Same columns in the same order, same rows; text cells equal, numeric cells equal to ``rel`` (relative)."""
    assert list(new.columns) == list(old.columns), f"{what}: columns {list(new.columns)} != {list(old.columns)}"
    assert len(new) == len(old), f"{what}: {len(new)} rows != {len(old)}"
    for col in old.columns:
        for i, (a, b) in enumerate(zip(new[col].astype(str), old[col].astype(str), strict=True)):
            try:
                fa, fb = float(a), float(b)
            except ValueError:
                assert a == b, f"{what}[{i}].{col}: {a!r} != {b!r}"
                continue
            assert fa == pytest.approx(fb, rel=rel, abs=1e-12, nan_ok=True), f"{what}[{i}].{col}: {a} != {b}"


def run_new(case: str, out, **kw):
    tf = CASES[case]
    ds, idx, tables = make_world(kind="skilled", bar=tf)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = ver.run_full_verification(
            ds, idx, tables, targets=TARGETS, x_keys=[f"X_{tf}"], cost_per_side=0.0008, n_boot=OLD_ARGS["n_boot"],
            target_acc=OLD_ARGS["target_acc"], rr=OLD_ARGS["rr"], stop_mode=OLD_ARGS["stop_mode"], min_n=MIN_N,
            out_dir=out / "analysis_outputs", verbose=True, **kw)
    return res, buf.getvalue()


@pytest.fixture(scope="module", params=list(CASES))
def match(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("new")
    res, printed = run_new(request.param, out, match_old=True)
    return request.param, res, printed, out


def test_the_verdict_table_equals_the_old_one_row_by_row(match):
    case, res, _, _ = match
    same_frame(res["summary"], old_read(case, "summary.csv"), "summary")


def test_the_printed_output_equals_the_old_stdout_line_by_line(match):
    case, _, printed, _ = match
    old = (GOLDEN / case / "printed.txt").read_text(encoding="utf-8")
    saved = "حُفظ في "
    new_lines, old_lines = printed.splitlines(), old.splitlines()
    assert len(new_lines) == len(old_lines), f"{len(new_lines)} printed lines != {len(old_lines)}"
    for i, (a, b) in enumerate(zip(new_lines, old_lines, strict=True)):
        if saved in a:                       # the folder the files went to differs; the rest of the line is the same
            assert a.split(saved)[0] == b.split(saved)[0] and a.endswith("/verification_summary.csv و.json")
        else:
            assert a == b, f"printed line {i}: {a!r} != {b!r}"


def test_the_sub_reports_equal_the_old_ones(match):
    case, res, _, _ = match
    same_frame(res["details"]["selective"], old_read(case, "selective.csv"), "selective")
    # the realised high / low / close come from the float32 labels here and from the raw float64 prices in the old
    # project: the R of a trade differs by ~1e-8 relative (the verdicts and counts are equal)
    same_frame(res["details"]["trades"], old_read(case, "trades.csv"), "trades", rel=1e-6)
    same_frame(res["details"]["candle"]["table"], old_read(case, "candle_table.csv"), "candle table")


def test_the_candle_verdicts_equal_the_old_ones(match):
    case, res, _, _ = match
    old = json.loads((GOLDEN / case / "candle_verdicts.json").read_text(encoding="utf-8"))
    new = res["details"]["candle"]["verdicts"]
    assert set(new) == set(old)
    for tg, v in old.items():
        n = new[tg]
        assert n["best_baseline"] == v["best"] and n["verdict"] == v["verdict"] and n["n_test_days"] == v["days"]
        np.testing.assert_allclose(n["auc_diff_vs_best"], v["diff"], rtol=1e-9)
        np.testing.assert_allclose(n["auc_gain_when_combined"], v["gain"], rtol=1e-9)


def test_the_two_saved_files_equal_the_old_ones_and_nothing_else_is_written(match):
    case, _, _, out = match
    folder = out / "analysis_outputs"
    assert sorted(p.name for p in folder.iterdir()) == ["verification_summary.csv", "verification_summary.json"]
    old = GOLDEN / case / "old_out"
    read = {"dtype": str, "keep_default_na": False, "encoding": "utf-8-sig"}
    same_frame(pd.read_csv(folder / "verification_summary.csv", **read),
               pd.read_csv(old / "verification_summary.csv", **read), "saved csv")
    new_json = json.loads((folder / "verification_summary.json").read_text(encoding="utf-8"))
    old_json = json.loads((old / "verification_summary.json").read_text(encoding="utf-8"))
    assert [(r["section"], r["check"], r["verdict"], r["note"]) for r in new_json] == \
        [(r["section"], r["check"], r["verdict"], r["note"]) for r in old_json]


def test_the_old_project_has_no_plots_and_no_other_files(match):
    """run_full_verification of the old project draws nothing and saves exactly the two summary files."""
    case, _, _, _ = match
    names = sorted(p.name for p in (GOLDEN / case / "old_out").iterdir())
    assert names == ["verification_summary.csv", "verification_summary.json"]


def test_without_match_the_six_hour_numbers_differ_from_the_old_ones(tmp_path):
    """The options are real: on 6 h bars the default (UTC-day blocks, the battery's cost, the NIG sd) gives other
    numbers than the old project."""
    res, _ = run_new("sixhour", tmp_path)
    old = old_read("sixhour", "summary.csv")
    days = res["summary"].query("check == 'independent test days'")["value"].iloc[0]
    old_days = int(old[old["check"].str.contains("أيام test المستقلة")]["value"].iloc[0])
    assert days != old_days                                                      # days != timestamps
    assert not np.allclose(res["details"]["trades"]["test_exp_R"].astype(float),
                           old_read("sixhour", "trades.csv")["test_exp_R"].astype(float))      # cost + stop width


def test_each_option_moves_only_its_own_numbers():
    """Every difference is its own switch: cost and stop width move R, the bootstrap unit moves t_days, the language
    moves only the text."""
    ds, idx, tables = make_world(kind="skilled", bar="6h")
    args = {"targets": TARGETS, "x_keys": ["X_6h"], "n_boot": 100, "min_n": MIN_N, "run_candle": False,
            "latest_n": 0}

    def trades(cost=0.0008, **variant):
        return ver.run_full_verification(ds, idx, tables, cost_per_side=cost, variant=Variant(**variant),
                                         **args)["details"]["trades"]
    base = trades()
    assert not np.allclose(trades(OLD_COST_PER_SIDE)["test_exp_R"].astype(float), base["test_exp_R"].astype(float))
    assert not np.allclose(trades(stop_width="student_t")["test_exp_R"].astype(float),
                           base["test_exp_R"].astype(float))
    by_ts, arabic = trades(block="timestamp"), trades(lang="ar")
    np.testing.assert_allclose(by_ts["test_exp_R"].astype(float), base["test_exp_R"].astype(float), rtol=1e-12)
    assert not np.allclose(by_ts["test_t_days"].astype(float), base["test_t_days"].astype(float))
    np.testing.assert_allclose(arabic["test_t_days"].astype(float), base["test_t_days"].astype(float), rtol=1e-12)
    assert list(arabic["verdict"]) != list(base["verdict"])                       # the text only
