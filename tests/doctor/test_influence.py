"""The influence entry (doctor/influence.py) end to end on a small synthetic E1: the monitored training, the batch
reports by coin and by month, and the feature importance files."""

import json

from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.doctor import influence


def test_main_writes_every_report(tmp_path):
    art = tmp_path / "e1"
    write_synthetic(art, SyntheticSettings(n_coins=4, n_bars=800), overwrite=True)
    out = tmp_path / "out"
    assert influence.main([str(art), "--out", str(out), "--epochs", "1", "--batch-size", "128", "--probe-rows", "128",
                           "--first-order-every", "2", "--n-perm", "50", "--repeats", "1", "--within-time",
                           "--group-by", "coin", "--group-every", "4", "--group-rows", "32"]) == 0
    bi = out / "batch_influence"
    rep = json.loads((bi / "batch_influence_report.json").read_text())
    assert rep["n_steps"] > 0 and rep["first_order"]["n_steps"] > 0 and "train" in rep["other_probes"]
    assert len(rep["groups"]) == 4                                   # one row per coin of the training rows
    assert len(rep["alignment"]) == 4 and (bi / "group_alignment.csv").exists()   # group-level TracIn per coin
    assert (bi / "by_month" / "batch_influence_report.md").exists() and (bi / "batch_rows.npz").exists()
    fi = (out / "feature_importance.md").read_text()
    assert "| close | feature |" in fi and "bars[" in fi
    assert (out / "feature_importance_within_time.csv").exists()
