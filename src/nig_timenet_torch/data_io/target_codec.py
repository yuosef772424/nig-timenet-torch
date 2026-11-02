"""TARGET_CODECS: the one place a regression target is encoded / decoded (C4; BUILD_PLAN.md P26). Only "return"
is registered (the oracle's and fixture's reg_target_mode).

"return" (crypto-signal-prediction data/windows.py + data/defaults.py reg_target_*): every head is compared with
its OWN last price — y_high = (future_high_max - last_high) / last_high, likewise low (future_low_min / last_low)
and close (future_close / last_close) — then y = clip(r, +-reg_target_clip) * reg_target_scale.

``encode(raw, state)`` takes ``raw`` of shape (N, 2) = [anchor, future price] (``own_anchor_and_future`` builds it
from last_candles); ``decode(pred, state, anchors)`` returns the future PRICE. ``state`` is self-contained: decode
reads nothing that fit did not put there.
"""

import numpy as np

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["TARGET_CODECS", "ReturnCodec", "own_anchor_and_future", "LAST_COLUMNS"]

LAST_COLUMNS = ("last_high", "last_low", "last_close", "timestamp", "future_close", "future_low_min",
                "future_high_max")
_OWN = {"high": ("last_high", "future_high_max"), "low": ("last_low", "future_low_min"),
        "close": ("last_close", "future_close")}

TARGET_CODECS = Registry("target_codec", Contract("target_codec", required_meta=("supports_relative",)))


def own_anchor_and_future(last_candles: np.ndarray, target: str) -> np.ndarray:
    """(N, 2) [own last price, own future price] for price target ``target`` (high / low / close)."""
    a, f = (LAST_COLUMNS.index(c) for c in _OWN[target])
    lc = np.asarray(last_candles, dtype=np.float64)
    return np.stack([lc[:, a], lc[:, f]], axis=1)


class ReturnCodec:
    name = "return"

    def fit(self, train: dict) -> dict:
        """Nothing is learned; the state records the dataset's scale/clip stamps (defaults 1.0 / None)."""
        clip = train.get("reg_target_clip")
        return {"mode": self.name, "version": 1, "scale": float(train.get("reg_target_scale", 1.0)),
                "clip": None if clip is None else float(clip)}

    def _check(self, state: dict) -> None:
        if state.get("mode") != self.name:
            raise ValueError(f"state is for codec {state.get('mode')!r}, not {self.name!r}")

    def encode(self, raw: np.ndarray, state: dict) -> np.ndarray:
        self._check(state)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("raw must be (N, 2) = [anchor, future price]")
        r = (raw[:, 1] - raw[:, 0]) / raw[:, 0]
        if state["clip"] is not None:
            r = np.clip(r, -state["clip"], state["clip"])
        return r * state["scale"]

    def decode(self, pred: np.ndarray, state: dict, anchors: np.ndarray) -> np.ndarray:
        self._check(state)
        return np.asarray(anchors, dtype=np.float64) * (1.0 + np.asarray(pred, dtype=np.float64) / state["scale"])

    def eval_spec(self, state: dict) -> dict:
        self._check(state)
        return {"mode": self.name, "scale": state["scale"], "clip": state["clip"], "decoded": "price",
                "anchor": "own last price of the head (last_high / last_low / last_close)",
                "round_trip_exact": state["clip"] is None}


TARGET_CODECS.register("return", ReturnCodec, version=1, supports_relative=True)
