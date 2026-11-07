"""The native pipeline reproduces crypto-signal-prediction's datasets exactly (same candles, same settings).

The fixtures tests/fixtures/tf_parity_{1tf,2tf}.npz were written by the reference pipeline
(tests/fixtures/make_tf_parity.py) on 4 seeded coins, 40 days, window 16 (+ 4h window 8, closed mode, float16)."""

from pathlib import Path

import numpy as np
import pytest

from nig_timenet_torch.data.build import build_dataset
from tests.data.helpers import tiny_settings

FIX = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.parametrize("two_tf,name", [(False, "tf_parity_1tf.npz"), (True, "tf_parity_2tf.npz")])
def test_dataset_equals_reference(two_tf, name):
    s = tiny_settings(two_tf, **({"runtime.x_storage_dtype": "float16"} if two_tf else {}))
    ds = build_dataset(s)
    ref = np.load(FIX / name)
    keys = [k for k in ref.files if k not in ("feature_order", "bounds")]
    assert [str(f) for f in ref["feature_order"]] == ds["feature_order"]
    assert [[b["start"], b["end"]] for b in ds["asset_bounds"]] == ref["bounds"].tolist()
    for k in keys:
        assert ds[k].dtype == ref[k].dtype, k
        np.testing.assert_array_equal(ds[k], ref[k], err_msg=k)
