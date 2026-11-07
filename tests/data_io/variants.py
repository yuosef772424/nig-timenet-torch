"""Variants of the committed 6-coin fixture artifact that look like the REAL TF artifact (HF-15, rule B3 for H2).

The fixture is 1h with ``split_dates`` set; the real artifact is 4h with ``split_dates = None``. Three Colab runs (#21,
#22, #23) failed on such differences. ``tf_shaped_artifact`` rewrites a copy of the fixture the way the export tool
would have written the real one: ``X_1h`` -> ``X_4h``, ``timeframes=['4h']``, ``base_tf='4h'``, ``split_dates=None``,
and a fresh ``export`` block (computed from the data). ``meta`` changes other keys of the dataset's ``meta.pkl``;
a value of ``...`` removes the key (for example ``reg_target_scale=...``).
"""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from tests.contract_doc import DATASET_DIR
from tests.data_io.helpers import write_manifest

TF_META = {"timeframes": ["4h"], "base_tf": "4h", "window_sizes": {"4h": 16, "1D": 32}, "split_dates": None}


def tf_shaped_artifact(dest: Path, **meta) -> Path:
    """Copy the fixture to ``dest`` and make it TF-shaped (4h, no split dates) plus the ``meta`` changes."""
    shutil.copytree(DATASET_DIR, dest)
    (dest / "X_1h.npy").rename(dest / "X_4h.npy")
    man = json.loads((dest / "manifest.json").read_text())
    man["arrays"] = {("X_4h" if k == "X_1h" else k): v for k, v in man["arrays"].items()}
    man["key_order"] = ["X_4h" if k == "X_1h" else k for k in man["key_order"]]
    (dest / "manifest.json").write_text(json.dumps(man))
    with open(dest / "meta.pkl", "rb") as f:
        m = pickle.load(f)
    for k, v in {**TF_META, **meta}.items():
        if v is ...:
            m.pop(k, None)
        else:
            m[k] = v
    with open(dest / "meta.pkl", "wb") as f:
        pickle.dump(m, f)
    ts = np.load(dest / "last_candles.npy")[:, TS_COL].astype("int64").max()
    bar_end = pd.Timestamp(int(ts), tz="UTC") + pd.Timedelta(m["base_tf"])  # the last input bar has closed
    write_manifest(dest, max_bar_time=bar_end.isoformat())
    return dest
