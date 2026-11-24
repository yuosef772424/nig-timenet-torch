"""workflow/chicks_run.py: the strict settings, ``run_chicks`` on a finished Run of the fixture artifact (files, result,
language), the skip of a target mode that is not a price forecast, and the command line."""

import json
import shutil
from pathlib import Path

import pytest

from nig_timenet_torch.core.config import from_dict, to_dict, updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.target_codec import ReturnCodec
from nig_timenet_torch.workflow import chicks_run
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.chicks_run import CALIBRATORS_JSON, ChicksSettings, run_chicks
from nig_timenet_torch.workflow.settings import RunSettings
from tests.contract_doc import DATASET_DIR
from tests.data_io.helpers import write_manifest


def test_the_defaults_are_the_tf_defaults():
    s = ChicksSettings()
    assert (s.calibrate, s.calibration_method, s.market_neutral) == (True, "isotonic", False)
    assert (s.trade_min_edge_ratio, s.trade_sort_by, s.trade_top_n, s.trade_cost_pct) == (1.0, "both", 20, 0.08)
    assert (s.run_trade_selection, s.run_integrity_check, s.batch_analysis_size) == (True, True, 100)
    assert (s.eval_all_rows, s.n_display, s.range_frac, s.unc_max, s.lang) == (True, 1, 0.2, 20.0, "ar")
    assert (s.make_plots, s.verbose, s.class_head, s.out_dir) == (True, True, False, None)


def test_settings_are_strict():
    with pytest.raises(ValueError, match="unknown key"):
        from_dict(ChicksSettings, {"calibrte": False})
    with pytest.raises(ValueError, match="calibration_method"):
        ChicksSettings(calibration_method="spline")
    with pytest.raises(ValueError, match="trade_sort_by"):
        ChicksSettings(trade_sort_by="best")
    with pytest.raises(ValueError, match="lang"):
        ChicksSettings(lang="fr")
    with pytest.raises(ValueError, match="trade_top_n"):
        updated(ChicksSettings(), {"trade_top_n": 0})
    assert from_dict(ChicksSettings, to_dict(updated(ChicksSettings(), {"lang": "en"}))).lang == "en"


def _settings(art_dir, tmp_path):
    base = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art_dir)))
    return updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "split.max_rows": 60,
                          "evaluation.health_max_rows": 40, "evaluation.verification.enabled": False})


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    work = tmp_path_factory.mktemp("chicks")
    art_dir = work / "artifact.dataset"
    shutil.copytree(DATASET_DIR, art_dir)
    write_manifest(art_dir)
    out = work / "out"
    return steps.run_all(_settings(art_dir, work), out), work


@pytest.mark.slow
def test_run_chicks_writes_the_report_files_of_a_finished_run(finished, capsys):
    run, _ = finished
    res = run_chicks(run, lang="en", n_display=0)
    out = Path(run.out) / "analysis_outputs"
    assert res["out_dir"] == str(out) and res["settings"].lang == "en"
    assert {"flat_df", "trust_report", "trading_metrics", "pattern_discovery", "integrity_diagnostics"} <= set(res)
    assert "flat_predictions_all.csv" not in res["files"] or (out / "flat_predictions_all.csv").is_file()
    assert any(f.startswith("figures/calibration_curve_") for f in res["files"])
    assert all((out / f).stat().st_size > 0 for f in res["files"])
    assert "Trust & Calibration Report" in capsys.readouterr().out
    if res["calibrators"]:
        saved = json.loads((out / CALIBRATORS_JSON).read_text(encoding="utf-8"))
        assert set(saved) == set(res["calibrators"])
    assert run.extra["chicks"] is res


@pytest.mark.slow
def test_run_chicks_can_switch_calibration_and_text_off_and_skips_a_non_forecast_mode(finished, monkeypatch, tmp_path):
    run, _ = finished
    res = run_chicks(run, calibrate=False, make_plots=False, out_dir=str(tmp_path / "plain"), verbose=False,
                     run_integrity_check=False, run_trade_selection=False)
    assert res["calibrators"] is None and res["trust_report"]["calibration"] is None
    assert not any(f.endswith(".png") for f in res["files"]) and res["trade_selection"] is None
    monkeypatch.setattr(ReturnCodec, "forecasts_price", lambda self, state: False)
    assert run_chicks(run) is None and run.extra["chicks_skipped"]


@pytest.mark.slow
def test_the_command_line_rebuilds_the_run_and_writes_the_report(finished, tmp_path, capsys):
    run, work = finished
    settings_json = tmp_path / "settings.json"
    settings_json.write_text(json.dumps(to_dict(run.settings), default=str), encoding="utf-8")
    out = tmp_path / "cli_out"
    train_dir = Path(run.settings.run_dir)
    assert chicks_run.main([str(train_dir), "--settings", str(settings_json), "--out", str(out),
                            "--set", "make_plots=false", "--set", 'lang="en"']) == 0
    assert (out / "analysis_outputs" / "flat_predictions_all.csv").is_file() or any(
        (out / "analysis_outputs").iterdir())
    assert "Trust & Calibration Report" in capsys.readouterr().out
