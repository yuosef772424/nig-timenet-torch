"""Parity of the torch chicks report (``eval.chicks_full.run_full_analysis``, Arabic) with the OLD project's
``run_full_analysis``, number by number, line by line.

The golden files (tests/parity/chicks_golden/<case>/) were written by the old project's own code
(``python -m tests.parity.make_chicks_golden``) on the arrays of ``tests.parity.chicks_world``: the printed text, every
file the old code saved (but the PNGs), a signature of each figure it saved (titles, labels, the data of lines, bars and
scatters) and the structures it returned. This test rebuilds the same world, runs the torch report and compares them
all. Anything that differs, a missing file or an extra one fails. The two cases cover the defaults (isotonic, daily
bars) and the other switches (Platt, market neutral, other trade rule, 6 h bars)."""

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import chicks_plots
from nig_timenet_torch.eval.chicks_full import run_full_analysis
from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.workflow.chicks_run import stage_from_dataset
from tests.eval.verification_world import TARGETS
from tests.parity import figsig
from tests.parity.chicks_world import CASES, small_world
from tests.parity.run_old_chicks import _jsonable

GOLDEN = Path(__file__).parent / "chicks_golden"
pytestmark = pytest.mark.skipif(not (GOLDEN / "daily_default" / "printed.txt").is_file(), reason="no golden files")
RTOL = 1e-6
PLATT_RTOL = 1e-4        # scikit-learn's lbfgs stops at its own tolerance; this project's Platt fit is Newton to 1e-12


def same(new, old, path: str = "", rel: float = RTOL) -> None:
    """Deep equality: text exact, numbers to ``RTOL`` (relative, tiny absolute floor), None / NaN alike."""
    if isinstance(old, dict):
        assert isinstance(new, dict) and list(new) == list(old), f"{path}: keys {list(new)} != {list(old)}"
        for k in old:
            same(new[k], old[k], f"{path}.{k}", rel)
    elif isinstance(old, list):
        assert isinstance(new, list) and len(new) == len(old), f"{path}: length {len(new)} != {len(old)}"
        for i, (a, b) in enumerate(zip(new, old, strict=True)):
            same(a, b, f"{path}[{i}]", rel)
    elif isinstance(old, bool) or old is None or isinstance(old, str):
        assert new == old, f"{path}: {new!r} != {old!r}"
    else:
        assert new == pytest.approx(old, rel=rel, abs=1e-9), f"{path}: {new} != {old}"


@pytest.fixture(scope="module", params=list(CASES))
def run(request, tmp_path_factory):
    case = request.param
    tf, n_days, n_coins, args = CASES[case]
    ds, idx, tables = small_world(tf, n_days, n_coins)
    test = stage_from_dataset(ds, idx["test"], tables["test"], TARGETS)
    val = stage_from_dataset(ds, idx["val"], tables["val"], TARGETS)
    sigs, original = {}, chicks_plots.save_figure

    def recording(fig, name, out_dir="analysis_outputs", dpi=130):
        sigs[name] = figsig.fig_signature(fig)
        return original(fig, name, out_dir, dpi)

    work = tmp_path_factory.mktemp(case)
    here, buf = os.getcwd(), io.StringIO()
    os.chdir(work)
    chicks_plots.save_figure = recording
    try:
        with contextlib.redirect_stdout(buf):
            res = run_full_analysis(test, out_dir="analysis_outputs", val_stage=val, say=Say("ar"), **args)
    finally:
        chicks_plots.save_figure = original
        os.chdir(here)
    printed = buf.getvalue().replace(str(work / "analysis_outputs"), "<OUT>")
    return case, res, printed, sigs, work / "analysis_outputs"


def test_the_printed_text_equals_the_old_stdout_line_by_line(run):
    case, _, printed, _, _ = run
    old = (GOLDEN / case / "printed.txt").read_text(encoding="utf-8")
    new_lines, old_lines = printed.splitlines(), old.splitlines()
    for i, (a, b) in enumerate(zip(new_lines, old_lines, strict=False)):
        assert a == b, f"printed line {i}: {a!r} != {b!r}"
    assert len(new_lines) == len(old_lines), f"{len(new_lines)} printed lines != {len(old_lines)}"


def test_the_saved_files_are_the_same_set_and_have_the_same_numbers(run):
    case, _, _, sigs, out = run
    old_dir = GOLDEN / case / "analysis_outputs"
    names = lambda root: sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())  # noqa: E731
    new_files = [n for n in names(out) if not n.endswith(".png")]
    assert new_files == [n for n in names(old_dir) if not n.endswith(".png")]
    old_figs = json.loads((GOLDEN / case / "figures.json").read_text(encoding="utf-8"))
    assert sorted(n for n in names(out) if n.endswith(".png")) == sorted(f"figures/{n}.png" for n in old_figs)
    assert all((out / n).stat().st_size > 0 for n in names(out))
    for name in new_files:
        if name.endswith(".csv"):
            read = {"dtype": str, "keep_default_na": False, "encoding": "utf-8-sig"}
            a, b = pd.read_csv(out / name, **read), pd.read_csv(old_dir / name, **read)
            assert list(a.columns) == list(b.columns) and len(a) == len(b), name
            for col in b.columns:
                for i, (x, y) in enumerate(zip(a[col], b[col], strict=True)):
                    try:
                        fx, fy = float(x), float(y)
                    except ValueError:
                        assert x == y, f"{name}[{i}].{col}: {x!r} != {y!r}"
                        continue
                    assert fx == pytest.approx(fy, rel=RTOL, abs=1e-9, nan_ok=True), f"{name}[{i}].{col}: {x} != {y}"
        else:
            assert (out / name).read_text(encoding="utf-8") == (old_dir / name).read_text(encoding="utf-8"), name


def test_every_figure_has_the_same_title_labels_and_data(run):
    case, _, _, sigs, _ = run
    old = json.loads((GOLDEN / case / "figures.json").read_text(encoding="utf-8"))
    assert sorted(sigs) == sorted(old)
    for name, sig in old.items():
        same(json.loads(json.dumps(sigs[name])), sig, f"figure {name}")


def test_the_returned_tables_equal_the_old_ones(run):
    case, res, _, _, _ = run
    old = json.loads((GOLDEN / case / "tables.json").read_text(encoding="utf-8"))
    pat = res["pattern_discovery"]
    new = {k: res[k] for k in ("asset_summary", "asset_ci", "coverage", "verification_summary", "aggregated_results",
                               "movement_analysis", "confidence_analysis", "batch_analysis", "trade_selection",
                               "integrity_diagnostics", "trading_metrics")}
    new["trade_candidates_n"] = None if res["trade_candidates"] is None else len(res["trade_candidates"])
    new["trust"] = {k: res["trust_report"][k] for k in ("overall", "by_group", "reliability_curve", "calibration")}
    new["calibrators"] = {t: {k: v for k, v in c.items() if k not in ("predict", "state")}
                          for t, c in res["calibrators"].items()}
    new["patterns"] = {"summary": pat["summary"], "heads": pat["heads"], "skipped": pat["skipped"],
                       "per_head": {h: {k: r[k] for k in ("n", "features", "tree_rules", "tree_accuracy",
                                                           "tree_naive_baseline_accuracy", "tree_balanced_accuracy",
                                                           "feature_importance", "clusters")}
                                    for h, r in pat["per_head"].items()}}
    new["flat_columns"] = list(res["flat_df"].columns)
    new = json.loads(json.dumps(_jsonable(new)))
    for k in old:
        same(new[k], old[k], k, rel=PLATT_RTOL if k in ("trust", "calibrators") and "platt" in case else RTOL)


def test_nothing_in_the_figures_is_empty_or_unreadable(run):
    _, _, _, sigs, _ = run
    for name, sig in sigs.items():
        titles = [a["title"] for a in sig["axes"]]
        assert any(titles) and all(isinstance(t, str) for t in titles), name
        assert not any(any("؀" <= ch <= "ۿ" for ch in a["title"] + a["xlabel"] + a["ylabel"])
                       for a in sig["axes"]), f"{name}: Arabic in a figure label (matplotlib cannot shape it)"
    assert np.isfinite(len(sigs))
