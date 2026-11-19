"""Permutation importance (doctor/feature_importance.py) against a known answer: a model trained on data where the label
depends on feature f0 (with a near-copy f1), on a market-wide feature f2, on the last bars only, and where f3's link to
the label reverses between the training and the validation rows; f4 / f5 are noise."""

import numpy as np
import pytest
import torch
from torch import nn

from nig_timenet_torch.doctor.feature_importance import (
    PERMUTERS,
    ImportanceSettings,
    correlation_clusters,
    feature_importance,
    time_blocks,
    write_importance,
)
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer

pytestmark = pytest.mark.slow   # rule E3 (the fast suite reached 312.6 s on CI); runs in CI's slow step

T, F, COINS = 6, 6, 16
NAMES = [f"f{j}" for j in range(F)]
CFG = {"targets": {"y": {"true_key": "y", "task_type": "regression", "output_keys": {"pred": "pred"}}}, "loss": {}}


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(T * F, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return {"pred": self.net(x).squeeze(-1)}


def _data(n_times, seed, spurious):
    g = torch.Generator().manual_seed(seed)
    n = n_times * COINS
    x = torch.randn(n, T, F, generator=g)
    times = np.repeat(np.arange(n_times), COINS)
    x[:, :, 2] = torch.randn(n_times, T, generator=g)[torch.as_tensor(times)]   # market-wide: same for every coin
    x[:, :, 1] = x[:, :, 0] + 0.05 * torch.randn(n, T, generator=g)              # a near-copy of f0
    y = x[:, -1, 0] + 0.7 * x[:, -1, 2] + 0.5 * torch.randn(n, generator=g) + spurious * x[:, -1, 3]
    return x, {"y": y}, times


@pytest.fixture(scope="module")
def trained():
    x, y, _ = _data(400, 0, spurious=0.6)        # f3's link to the label reverses between training and validation
    xv, yv, tv = _data(150, 1, spurious=-0.6)
    torch.manual_seed(0)
    t = Trainer(MLP(), CFG, optimizer=lambda ps: torch.optim.Adam(ps, lr=3e-3))
    keep_last = EarlyStopSettings(restore_best_weights=False)
    t.fit(x, y, xv, yv, FitSettings(epochs=15, batch_size=128, early_stop=keep_last))
    return t, xv, yv, tv


def _by_group(table):
    return table.set_index("group")


def test_global_importance_finds_the_planted_answer(trained):
    t, xv, yv, tv = trained
    table, summary = feature_importance(t, xv, yv, tv, NAMES, ImportanceSettings(repeats=3, bar_blocks=3))
    g = _by_group(table)
    assert set(g.index[g["verdict"] == "uses"]) == {"f0", "f1", "f2", "f0+f1", "bars[-2:]"}
    assert g.loc["f3", "verdict"] == "hurts"                     # reversed link: the model does better without it
    assert (g.loc[["f4", "f5", "bars[-6:-4]", "bars[-4:-2]"], "verdict"] == "none").all()
    # the twin: each of f0 / f1 alone looks smaller than the pair permuted together
    assert g.loc["f0+f1", "delta_loss"] > 2 * max(g.loc["f0", "delta_loss"], g.loc["f1", "delta_loss"])
    assert g.loc["f0+f1", "ic_y"] < -0.3 and summary["split_half_rho"] > 0.5
    assert summary["uses"] == 5 and summary["ci_level"] == pytest.approx(1 - 0.05 / len(table))


def test_within_time_importance_ignores_the_market_wide_feature(trained):
    t, xv, yv, tv = trained
    table, _ = feature_importance(t, xv, yv, tv, NAMES, ImportanceSettings(repeats=2, permutation="within_time"))
    g = _by_group(table)
    assert g.loc["f2", "delta_loss"] == 0.0 and g.loc["f2", "verdict"] == "none"
    assert g.loc["f0+f1", "verdict"] == "uses"


def test_permuters_and_blocks():
    rng = np.random.default_rng(0)
    times = np.repeat(np.arange(10), 7)[rng.permutation(70)]
    perm = PERMUTERS["within_time"](70, times, rng)
    assert sorted(perm) == list(range(70)) and (times[perm] == times).all() and (perm != np.arange(70)).any()
    assert sorted(PERMUTERS["global"](70, None, rng)) == list(range(70))
    with pytest.raises(ValueError, match="times"):
        PERMUTERS["within_time"](70, None, rng)
    b = time_blocks(times, 5)
    assert len(np.unique(b)) == 5 and all(len(np.unique(b[times == t])) == 1 for t in range(10))


def test_correlation_clusters_link_transitively():
    g = np.random.default_rng(1)
    x = g.normal(size=(2000, 2, 5))
    x[:, :, 1] = x[:, :, 0] + 0.01 * g.normal(size=(2000, 2))
    x[:, :, 2] = x[:, :, 1] + 0.01 * g.normal(size=(2000, 2))
    assert correlation_clusters(x, 0.95) == [[0, 1, 2]]
    assert correlation_clusters(x, 1.0 + 1e-9) == []


def test_settings_guards_and_writer(trained, tmp_path):
    with pytest.raises(ValueError, match="permutation"):
        ImportanceSettings(permutation="nope")
    with pytest.raises(ValueError):
        ImportanceSettings(n_blocks=2)
    t, xv, yv, tv = trained
    with pytest.raises(ValueError, match="names"):
        feature_importance(t, xv, yv, tv, NAMES[:3])
    table, summary = feature_importance(t, xv[:640], {"y": yv["y"][:640]}, tv[:640], NAMES,
                                        ImportanceSettings(repeats=1, n_blocks=8, cluster_threshold=1.0),
                                        groups={"f0_f2": [0, 2]})
    assert "custom" in set(table["kind"]) and "cluster" not in set(table["kind"])
    write_importance(tmp_path, table, summary)
    md = (tmp_path / "feature_importance.md").read_text()
    assert "split-half" in md and "f0_f2" in md and (tmp_path / "feature_importance.csv").exists()
