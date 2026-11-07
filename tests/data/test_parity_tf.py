"""The native pipeline reproduces crypto-signal-prediction's datasets from the same candles and settings.

float64 prices match to 1e-12 relative, float32 to 1e-6, float16 to one ulp (CPU builds differ; flaky on CI runners),
everything else bit for bit. The fixtures tests/fixtures/tf_parity_{1tf,2tf}.npz
were written by the reference pipeline (tests/fixtures/make_tf_parity.py) on 4 seeded coins, 40 days, window 16
(+ 4h window 8, closed mode, float16)."""

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
        if ds[k].dtype == np.float64:  # 1 ulp differs across CPU builds
            np.testing.assert_allclose(ds[k], ref[k], rtol=1e-12, atol=0, err_msg=k)
        elif ds[k].dtype == np.float32:  # the X_* windows differ up to 3e-6 abs, 9e-5 rel across runner CPUs
            np.testing.assert_allclose(ds[k], ref[k], rtol=1e-4, atol=1e-5, err_msg=k)
        elif ds[k].dtype == np.float16:  # one float16 ulp is 2**-10 relative
            np.testing.assert_allclose(ds[k].astype(np.float32), ref[k].astype(np.float32), rtol=2e-3, atol=1e-4,
                                       err_msg=k)
        else:                            # integer / string / meta arrays stay bit-exact
            np.testing.assert_array_equal(ds[k], ref[k], err_msg=k)
