"""P37: rolling seam geometry only — train_fn / predict_fn are stubs; nothing is trained."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.split_plan import check_split_purity
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.rolling import rolling_folds, run_rolling_study
from nig_timenet_torch.workflow.settings import RollingSettings, RunSettings
from tests.contract_doc import DATASET_DIR
from tests.data_io.test_split_plan import fixture_plan

PARENT = RunIdentity(schema_version=1, settings_hash="a" * 64, seed=0, data_fingerprint="fixture", code_sha="0" * 40)


def settings(**over):
    base = dict(train_span="20D", test_span="7D", step="10D", embargo="1D")
    return RollingSettings(**{**base, **over})


def test_rolling_settings_default_is_none_equivalent_to_single_split():
    assert RunSettings().rolling is None and from_dict(RunSettings, {}).rolling is None
    # the single-split path (P25) is untouched: same plan as before the seam existed
    a, b = fixture_plan(), fixture_plan()
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("train_idx", "val_idx", "test_idx"))
    # a single-split RunIdentity's JSON / fingerprint does not carry the new rolling fields
    assert "fold" not in PARENT.to_json() and "data_cut" not in PARENT.to_json()
    assert RunIdentity.from_json(PARENT.to_json()) == PARENT
    with pytest.raises(TypeError):
        RollingSettings()  # spans have no invented defaults


@pytest.mark.parametrize("mode", ["expanding", "sliding"])
def test_rolling_folds_produces_non_overlapping_purged_folds(mode):
    d = npy_dir.load(DATASET_DIR)
    folds = rolling_folds(d, settings(mode=mode, max_folds=3))
    assert 2 <= len(folds) <= 3
    for f in folds:
        check_split_purity(f.plan)
        p = f.plan
        assert len(p.train_idx) and len(p.test_idx)
        assert p.label_end[p.train_idx].max() <= f.train_end.value
        assert p.feature_start[p.test_idx].min() > (f.train_end + pd.Timedelta("1D")).value
        assert p.label_end[p.test_idx].max() <= f.test_end.value
        if mode == "sliding":
            assert p.feature_start[p.train_idx].min() >= (f.train_end - pd.Timedelta("20D")).value
    for a, b in zip(folds, folds[1:], strict=False):  # monotone in time
        assert a.train_end < b.train_end and a.test_start < b.test_start
        assert a.plan.feature_start[a.plan.test_idx].min() < b.plan.feature_start[b.plan.test_idx].min()
    if mode == "expanding":
        assert all(len(a.plan.train_idx) < len(b.plan.train_idx) for a, b in zip(folds, folds[1:], strict=False))


def test_run_rolling_study_wiring_with_stubs():
    d = npy_dir.load(DATASET_DIR)
    calls = []
    run_rolling_study(d, settings(max_folds=3, retrain_every_k=2, seeds=(0, 1)),
                      lambda _d, fold, ident: calls.append(("train", fold.index, ident.seed)),
                      lambda *_: None, PARENT)
    assert calls == [("train", 0, 0), ("train", 0, 1), ("train", 2, 0), ("train", 2, 1)]
    seen = []
    run_rolling_study(d, settings(max_folds=2), lambda *_: None,
                      lambda _m, _d, fold, ident: seen.append(ident), PARENT)
    assert [i.fold for i in seen] == [0, 1] and all(i.parent_hashes == ("a" * 64,) for i in seen)
    assert len({i.fingerprint() for i in seen}) == 2 and dataclasses.replace(seen[0], fold=None).data_cut
