"""workflow/live_model.py + workflow/live_run.py: a model exported from a finished run is loaded from its folder, the
live candles (a fake API) are turned into windows by the training data code and the live rows equal the run's rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data.fetch import Http
from nig_timenet_torch.data.sources import synthetic_market
from nig_timenet_torch.data_io.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import latest_report, live_model, live_run
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings
from tests.data.helpers import COINS, tiny_settings

pytestmark = pytest.mark.slow        # rule E3: builds a dataset, trains one epoch, predicts live (~20 s)
H = 3_600_000
SMALL = {"d_model": 16, "num_layers": 1, "head_hidden": 16, "class_head_hidden": 8}


def _market() -> dict:
    """The synthetic market with every coin cut at the same last bar (a live API has one 'now' for all)."""
    market = synthetic_market(list(COINS), 40, 40)
    end = min(f.index[-1] for f in market.values())
    return {c: f[f.index <= end] for c, f in market.items()}


def _klines(frame: pd.DataFrame) -> list:
    rows = []
    for t, r in frame.iterrows():
        ms = int(t.timestamp() * 1000)
        rows.append([ms, str(r["open"]), str(r["high"]), str(r["low"]), str(r["close"]), str(r["volume"]),
                     ms + H - 1, "1000", 5, "1", "1", "0"])
    return rows


class FakeApi(Http):
    """The futures klines API over the synthetic market: pages from ``startTime``, never an open bar."""

    def __init__(self, market: dict):
        self.rows = {c: _klines(f) for c, f in market.items()}
        self.asked = []

    def get_json(self, url, params=None):
        self.asked.append(params["symbol"])
        if params["symbol"] not in self.rows:
            raise ConnectionError("unknown symbol")
        return [r for r in self.rows[params["symbol"]] if r[0] >= params["startTime"]][:params["limit"]]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A run trained for one epoch on an artifact the torch data pipeline built from the synthetic market."""
    root = tmp_path_factory.mktemp("live")
    market = synthetic_market(list(COINS), 40, 40)
    write_artifact(build_dataset(tiny_settings(), loader=lambda n: market[n]), root / "art.dataset",
                   source_commit="a" * 40)
    s = updated(RunSettings(run_dir=str(root / "run"), data=LoaderSettings(name=str(root / "art.dataset"))),
                {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "model.overrides": SMALL})
    run = steps.run_all(s, root / "out")
    return run, root


def test_export_then_load_from_the_folder_gives_the_runs_own_predictions(trained):
    run, _ = trained
    path = live_model.export_live_model(run)
    assert path.name == live_model.BUNDLE and json.loads(path.read_text())["coins"] == list(COINS)
    lm = live_model.load_live_model(run.settings.run_dir)
    assert lm.model.training is False and lm.targets == run.targets and lm.identity == run.ident
    x, _ = steps.flat_run.flat_inputs(run.dataset, run.idx["val"][:30], False, False, ())
    with torch.no_grad():
        a, b = run.trainer.model.eval()(x), lm.model(x)
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_a_run_trained_before_the_export_existed_gets_the_same_json_from_its_folder_and_artifact(trained, tmp_path):
    run, root = trained
    folder = Path(run.settings.run_dir)
    from_run = json.loads(live_model.export_live_model(run).read_text())
    steps.write_summary(run)                                        # the run's own baseline_summary.json (the old way)
    (run.out / "baseline_summary.json").replace(folder / "baseline_summary.json")
    (folder / live_model.BUNDLE).unlink()
    path = live_model.export_from_folder(folder, root / "art.dataset", overrides=SMALL)
    from_folder = json.loads(path.read_text())
    assert from_folder == from_run and live_model.load_live_model(folder).targets == run.targets
    live_model.main([str(folder), "--artifact", str(root / "art.dataset"), "--overrides", json.dumps(SMALL)])
    with pytest.raises(ValueError, match="does not fit"):                    # the default model is not the trained one
        live_model.export_from_folder(folder, root / "art.dataset", overrides={})
    assert json.loads((folder / live_model.BUNDLE).read_text()) == from_run  # the failed export removed nothing else
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        live_model.export_from_folder(tmp_path, root / "art.dataset")


def test_loading_refuses_a_folder_without_the_json_the_weights_or_a_fitting_model(trained, tmp_path):
    run, _ = trained
    live_model.export_live_model(run)
    with pytest.raises(FileNotFoundError, match="export_live_model"):
        live_model.load_live_model(tmp_path)
    spec = json.loads((Path(run.settings.run_dir) / live_model.BUNDLE).read_text())
    (tmp_path / live_model.BUNDLE).write_text(json.dumps(spec))
    with pytest.raises(FileNotFoundError, match="best weights"):
        live_model.load_live_model(tmp_path)
    (tmp_path / "best.pt").write_bytes((Path(run.settings.run_dir) / "best.pt").read_bytes())
    live_model.load_live_model(tmp_path)                                     # the two files are enough
    (tmp_path / live_model.BUNDLE).write_text(json.dumps({**spec, "overrides": {**spec["overrides"], "d_model": 24}}))
    with pytest.raises(ValueError, match="does not fit the architecture"):
        live_model.load_live_model(tmp_path)
    (tmp_path / live_model.BUNDLE).write_text(json.dumps({**spec, "schema": "other/9"}))
    with pytest.raises(ValueError, match="schema"):
        live_model.load_live_model(tmp_path)
    (tmp_path / live_model.BUNDLE).write_text(json.dumps({**spec, "coin_channel": True}))
    with pytest.raises(NotImplementedError, match="coin-channel"):
        live_model.load_live_model(tmp_path)


def test_live_run_rows_equal_the_runs_rows_and_the_newest_row_is_live(trained, tmp_path):
    run, _ = trained
    live_model.export_live_model(run)
    market = _market()
    now = market[COINS[0]].index[-1] + pd.Timedelta("1h")                  # the last bar just closed
    lines = []
    res = live_run.live_run(run.settings.run_dir, list(COINS), last=300, bars=1000, http=FakeApi(market), now=now,
                            out_dir=tmp_path / "out", charts=True, echo=lines.append)
    tbl = res["table"]
    assert res["fetched"] == 4 and not res["skipped"] and set(tbl["coin"]) == set(COINS)
    newest = tbl[tbl["end_time"] == tbl["end_time"].max()]
    assert len(newest) == 4 and newest["y_ret"].isna().all() and newest["mu"].notna().all()  # live, a prediction
    assert tbl[tbl["end_time"] < tbl["end_time"].max()]["y_ret"].notna().all()               # horizon 1: others judged
    assert (tbl["end_time"].diff().dropna().dt.total_seconds() >= 0).all()                   # time major
    # the live row at a bar of the run's grid is the run's row of that bar (same windows, same model): the run's
    # battery table over the same stage gives the decoded return to compare with
    from nig_timenet_torch.workflow.battery_run import stage_inputs
    ref = pd.concat([stage_inputs(run, st).battery_table() for st in ("val", "test")])
    merged = tbl.merge(ref, on=["coin", "end_time"], suffixes=("", "_run"))
    assert len(merged) >= 5
    assert np.allclose(merged["mu"], merged["mu_run"], atol=1e-5) and np.allclose(merged["sigma"], merged["sigma_run"],
                                                                                   atol=1e-5)
    assert np.allclose(merged["p_dir"], merged["p_dir_run"], atol=1e-5)
    out = tmp_path / "out"
    assert (out / latest_report.LATEST_CSV).is_file() and (out / "live_run.json").is_file()
    assert len(list((out / "figures").glob("*.png"))) == 4 and any("live" in ln for ln in lines)


def test_a_coin_the_api_cannot_serve_or_that_is_stale_is_reported_not_predicted(trained, tmp_path):
    run, _ = trained
    live_model.export_live_model(run)
    market = _market()
    market["C1USDT"] = market["C1USDT"].iloc[:-30]                           # stopped trading 30 hours ago
    del market["C3USDT"]                                                     # the API does not know it
    now = market[COINS[0]].index[-1] + pd.Timedelta("1h")
    res = live_run.live_run(run.settings.run_dir, list(COINS), last=3, bars=1000, http=FakeApi(market), now=now,
                            charts=False, echo=lambda s: None)
    assert "ConnectionError" in res["skipped"]["C3USDT"] and "stale" in res["skipped"]["C1USDT"]
    assert set(res["table"]["coin"]) == {"C0USDT", "C2USDT"}
    with pytest.raises(RuntimeError, match="no candles fetched"):
        live_run.live_run(run.settings.run_dir, ["C3USDT"], last=3, http=FakeApi(market), now=now, echo=print)


def test_a_model_folder_without_data_settings_asks_for_them(trained, tmp_path):
    run, _ = trained
    live_model.export_live_model(run)
    spec = json.loads((Path(run.settings.run_dir) / live_model.BUNDLE).read_text())
    (tmp_path / live_model.BUNDLE).write_text(json.dumps({**spec, "data_settings": None}))
    (tmp_path / "best.pt").write_bytes((Path(run.settings.run_dir) / "best.pt").read_bytes())
    with pytest.raises(ValueError, match="data_settings"):
        live_run.live_run(tmp_path, list(COINS), http=FakeApi(_market()), now="2026-01-01", echo=print)
