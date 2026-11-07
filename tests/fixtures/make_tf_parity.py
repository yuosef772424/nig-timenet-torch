"""Regenerates tests/fixtures/tf_parity_{1tf,2tf}.npz with the REFERENCE pipeline (crypto-signal-prediction).

    cd <crypto-signal-prediction checkout>   (pandas-ta-classic==0.8.32, the reference's pinned version)
    python <this file> /path/to/out_dir

The reference is only read, never changed. Not a test: pytest does not collect it.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from tests import run_fixture as rf  # noqa: E402  (the reference's own test helpers)

STARTS = ["2025-01-01", "2025-01-01 03:00", "2025-01-01 05:00", "2025-01-01 11:00"]


def build(ns, two_tf, out):
    ns["reset_config"]()
    over = ns["HOURLY_4H_OVERRIDES"] if two_tf else ns["HOURLY_W32_S8_OVERRIDES"]
    rf._quiet(ns["apply_hourly_preset"], over)
    ns["update_config"]({"tf_order": ["1h", "4h"] if two_tf else ["1h"],
                         "window_sizes": {"1h": 16, "4h": 8} if two_tf else {"1h": 16},
                         "phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False},
                         "funding_rate": {"enabled": False}, "open_interest": {"enabled": False},
                         "market_context": {"reference_symbol": "C0USDT"}})
    rf._quiet(ns["refresh_features"])

    def loader(fid, name):
        i = int(name[1])
        return rf.ohlcv(40, 40 + i, STARTS[i])

    ds = rf._quiet(ns["build_dataset"], [{"name": f"C{i}USDT"} for i in range(4)], load_asset_fn=loader,
                   resample_fn=ns["make_resample_fn"](ns["CONFIG"]), max_workers=1, config=ns["CONFIG"])
    ns["reset_config"]()
    arrays = {k: v for k, v in ds.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out, **arrays, feature_order=np.array(ds["feature_order"]),
                        bounds=np.array([[b["start"], b["end"]] for b in ds["asset_bounds"]]))


if __name__ == "__main__":
    namespace = rf._nbload.load_pipeline()
    build(namespace, False, f"{sys.argv[1]}/tf_parity_1tf.npz")
    build(namespace, True, f"{sys.argv[1]}/tf_parity_2tf.npz")
