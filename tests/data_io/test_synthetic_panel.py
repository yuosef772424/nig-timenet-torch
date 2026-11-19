"""E4 (data_io/synthetic_panel.py): the panel environment's known answers, its shape, the contract round trip."""

import numpy as np
import pytest

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.synthetic import ORACLE_PREFIX, oracle_keys
from nig_timenet_torch.data_io.synthetic_panel import BETA_FLOOR, PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.settings import SplitSettings

SHA = "0" * 40
SMALL = PanelSyntheticSettings(n_coins=24, n_bars=1600, specific_ic=0.2, seed=3)


@pytest.fixture(scope="module")
def e4():
    return build_panel_synthetic(SMALL)


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_every_timestamp_holds_every_coin_and_the_oracle_columns_are_there(e4):
    ts = np.asarray(e4["last_candles"])[:, TS_COL].astype(np.int64)
    stamps, counts = np.unique(ts, return_counts=True)
    assert len(stamps) == SMALL.n_bars // SMALL.horizon and set(counts) == {SMALL.n_coins}
    for key in ("beta", "market", "specific_sd", "regime", "signal"):
        assert len(e4[f"{ORACLE_PREFIX}{key}"]) == len(ts)
    beta = e4[f"{ORACLE_PREFIX}beta"].reshape(SMALL.n_coins, -1)
    assert (beta == beta[:, :1]).all() and beta.min() >= BETA_FLOOR and beta[:, 0].std() > 0.1   # one beta per coin
    market = e4[f"{ORACLE_PREFIX}market"].reshape(SMALL.n_coins, -1)
    assert (market == market[:1]).all()                                           # one market move per timestamp
    assert e4["environment"]["name"] == "E4" and e4["synthetic"]["generator"].endswith("synthetic_panel")


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_the_known_answers_are_the_planted_ones(e4):
    p = e4["synthetic"]["panel"]
    n = len(e4["last_candles"])
    assert abs(p["specific_ic_measured"] - SMALL.specific_ic) < 4 / np.sqrt(n)     # exact up to sampling
    assert 0.45 < p["market_share_of_close_variance"] < 0.75                        # the market is most of the move
    own, rel, oracle = (p["readers"][k] for k in ("own_move", "own_minus_timestamp_mean", "oracle_signal"))
    # a ranking inside the timestamp does not see the timestamp mean; a pooled loss does: the ladder a model climbs
    assert own["rank_ic_by_timestamp"] == pytest.approx(rel["rank_ic_by_timestamp"])
    assert own["pearson_pooled"] < rel["pearson_pooled"] and own["pearson_with_signal"] < rel["pearson_with_signal"]
    assert oracle["rank_ic_by_timestamp"] >= own["rank_ic_by_timestamp"] > 0.1
    # the market move has no predictable part: the oracle mean is the specific drift only
    y = np.asarray(e4["y_close_reg"], np.float64)
    mean = np.asarray(e4[oracle_keys("close")[0]], np.float64)
    resid = y - mean
    assert abs(np.corrcoef(resid, mean)[0, 1]) < 4 / np.sqrt(n)
    assert abs(np.mean(resid / np.asarray(e4[oracle_keys("close")[1]], np.float64))) < 0.05


def test_e4_is_a_contract_artifact_a_cross_section_run_can_split(e4, tmp_path):
    write_artifact(e4, tmp_path / "e4.dataset", source_commit=SHA)
    ds = load_dataset(LoaderSettings(directory=str(tmp_path), name="e4.dataset", local_dir="")).dataset
    run_common.check_decode(ds, run_common.price_targets(ds))
    parts, _ = run_common.make_plan(ds, SplitSettings(cross_section=True))
    assert all(len(p) and len(p) % SMALL.n_coins == 0 for p in parts)               # whole timestamps on each side
    assert ds["synthetic"]["panel"]["specific_ic_planted"] == SMALL.specific_ic


def test_settings_refuse_what_cannot_be_simulated():
    for bad in ({"n_coins": 1}, {"specific_ic": 1.0}, {"n_bars": 1601}, {"vol_persistence": 1.0},
                {"market_sigma": 0.0}, {"label_codec": "nope"}, {"signal_bars": 64}):
        with pytest.raises(ValueError):
            PanelSyntheticSettings(**bad)
