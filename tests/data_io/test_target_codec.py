"""P26: TARGET_CODECS / ReturnCodec."""

import numpy as np
import pytest

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, own_anchor_and_future
from tests.contract_doc import DATASET_DIR


def synthetic_candles(seed: int, n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    anchor = np.exp(rng.normal(np.log(100.0), 2.0, n))  # prices across many orders of magnitude
    future = anchor * np.exp(rng.normal(0.0, 0.05, n))
    return np.stack([anchor, future], axis=1)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_return_codec_round_trip(seed, scale):
    codec = TARGET_CODECS["return"]()
    state = codec.fit({"reg_target_scale": scale})
    raw = synthetic_candles(seed)
    np.testing.assert_allclose(codec.decode(codec.encode(raw, state), state, anchors=raw[:, 0]), raw[:, 1],
                               rtol=1e-12, atol=0)


def test_encode_matches_fixture_targets():
    d = npy_dir.load(DATASET_DIR)
    codec = TARGET_CODECS["return"]()
    state = codec.fit(d)
    assert state == {"mode": "return", "version": 1, "scale": 100.0, "clip": None}
    for t in ("high", "low", "close"):
        y = codec.encode(own_anchor_and_future(d["last_candles"], t), state)
        np.testing.assert_allclose(y, d[f"y_{t}_reg"], rtol=0, atol=1e-6)  # float32-stored labels


def test_clip_and_state_checks():
    codec = TARGET_CODECS["return"]()
    state = codec.fit({"reg_target_scale": 1.0, "reg_target_clip": 0.01})
    assert codec.encode(np.array([[100.0, 200.0]]), state)[0] == pytest.approx(0.01)
    assert codec.eval_spec(state)["round_trip_exact"] is False
    with pytest.raises(ValueError):
        codec.encode(np.array([[1.0, 2.0]]), {"mode": "scaled"})


def test_unregistered_mode_raises():
    assert TARGET_CODECS.names() == ("return",)
    with pytest.raises(UnknownNameError):
        TARGET_CODECS["entry_range"]


def test_fit_requires_reg_target_scale():
    """No silent scale 1.0: an artifact without the contract key is refused (B-034 class)."""
    codec = TARGET_CODECS["return"]()
    with pytest.raises(ValueError, match="reg_target_scale"):
        codec.fit({"reg_target_mode": "return"})
    assert codec.fit({"reg_target_scale": 100})["scale"] == 100.0


def test_unknown_codec_mode_raises():
    with pytest.raises(UnknownNameError):
        TARGET_CODECS["volnorm"]
