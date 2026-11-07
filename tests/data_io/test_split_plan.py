"""P25: SplitPlan purity on the P01 fixture (cut dates: the dataset's own split_dates; embargo window+horizon)."""

import dataclasses

import numpy as np
import pytest

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.data_io.split_plan import LeakageError, check_split_purity
from tests.data_io.helpers import fixture_plan


def test_purity_train_before_val_before_test():
    p = fixture_plan()
    assert min(len(p.train_idx), len(p.val_idx), len(p.test_idx)) > 0
    assert p.label_end[p.train_idx].max() < p.feature_start[p.val_idx].min()
    assert p.label_end[p.val_idx].max() < p.feature_start[p.test_idx].min()
    assert len(set(p.train_idx) | set(p.val_idx) | set(p.test_idx)) == len(p.train_idx) + len(p.val_idx) + len(
        p.test_idx)


def test_leaky_splitter_is_caught():
    p = fixture_plan()
    leaky_val = np.concatenate([p.train_idx[-5:], p.val_idx])  # val overlapping train's label_end
    leaky = dataclasses.replace(p, val_idx=leaky_val)
    with pytest.raises(LeakageError, match="train -> val"):
        check_split_purity(leaky)


def test_calibration_role_disjoint_from_fit():
    p = fixture_plan()
    assert not set(p.role_of[Role.CALIBRATION]) & set(p.role_of[Role.FIT])
    assert np.array_equal(p.calibration_idx, p.val_idx)
    bad = dataclasses.replace(p, role_of={**p.role_of, Role.CALIBRATION: p.train_idx[:3]})
    with pytest.raises(LeakageError, match="CALIBRATION"):
        check_split_purity(bad)
