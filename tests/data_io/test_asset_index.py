"""asset_ids / row_coins equal the loop builders they replaced (C-19), on the committed fixture and on gaps."""

import numpy as np

from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.asset_index import asset_ids, row_coins
from tests.data_io.helpers import DATASET_DIR


def _old_ids(dataset):
    ids = np.full(np.asarray(dataset["last_candles"]).shape[0], -1, dtype=np.int64)
    for i, b in enumerate(dataset["asset_bounds"]):
        ids[b["start"]:b["end"]] = i
    return ids


def _old_names(dataset):
    names = np.empty(np.asarray(dataset["last_candles"]).shape[0], dtype=object)
    for b in dataset["asset_bounds"]:
        names[b["start"]:b["end"]] = b["name"]
    return names


def test_equals_the_loop_builders_on_the_fixture():
    d = npy_dir.load(DATASET_DIR)
    n = np.asarray(d["last_candles"]).shape[0]
    for idx in (np.arange(n), np.arange(n)[::-1], np.array([0, n - 1, n // 2]), np.array([], dtype=np.int64)):
        assert (asset_ids(d, idx) == _old_ids(d)[idx]).all()
        assert list(row_coins(d, idx)) == list(_old_names(d)[idx])


def test_unsorted_bounds_and_gaps():
    d = {"asset_bounds": [{"name": "B", "start": 5, "end": 8}, {"name": "A", "start": 0, "end": 3}],
         "last_candles": np.zeros((10, 1))}
    idx = np.arange(10)
    assert list(asset_ids(d, idx)) == [1, 1, 1, -1, -1, 0, 0, 0, -1, -1]
    assert list(row_coins(d, idx)) == ["A", "A", "A", None, None, "B", "B", "B", None, None]
