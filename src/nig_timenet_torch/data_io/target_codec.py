"""PURPOSE: TARGET_CODECS, the one place a regression target is encoded from last_candles and decoded back to a price:
    "return" (own-anchor simple return) and "logret_close" (log return vs the last close, TF logret_* artifacts).
TAGS: target codec, return target, log return, reg target scale, decode to price, own anchor, anchor column
PITFALLS: decode needs the anchor the codec names (``anchor_column``: own last price for "return", last_close for
    "logret_close"); ``codec_name`` picks the codec from reg_target_mode AND price_norm_mode (a TF logret_* artifact
    says reg_target_mode "return" but its labels are log(future / last_close), B-051); fit refuses a dataset without
    reg_target_scale; LAST_COLUMNS lives in core.schema and is only re-exported here.

TARGET_CODECS: the one place a regression target is encoded / decoded (C4; BUILD_PLAN.md P26).

"return" (crypto-signal-prediction data/windows.py + data/defaults.py reg_target_*): every head is compared with
its OWN last price — y_high = (future_high_max - last_high) / last_high, likewise low (future_low_min / last_low)
and close (future_close / last_close) — then y = clip(r, +-reg_target_clip) * reg_target_scale.

``encode(raw, state)`` takes ``raw`` of shape (N, 2) = [anchor, future price] (``own_anchor_and_future`` builds it
from last_candles); ``decode(pred, state, anchors)`` returns the future PRICE. ``state`` is self-contained: decode
reads nothing that fit did not put there.
"""

import numpy as np

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.schema import LAST_COLUMNS

__all__ = ["TARGET_CODECS", "ReturnCodec", "LogCloseCodec", "own_anchor_and_future", "anchor_and_future",
           "anchor_column", "codec_name", "LOGRET_NORM_MODES", "LAST_COLUMNS"]

#: price_norm_mode values of the TF pipeline whose regression labels are log(future price / last close), whatever
#: reg_target_mode says (crypto-signal-prediction data/windows.py: ``nz.raw_reg_target(own_future, last_c)``)
LOGRET_NORM_MODES = ("logret", "logret_std")

_OWN = {"high": ("last_high", "future_high_max"), "low": ("last_low", "future_low_min"),
        "close": ("last_close", "future_close")}

TARGET_CODECS = Registry("target_codec", Contract("target_codec", required_meta=("supports_relative",)))


def own_anchor_and_future(last_candles: np.ndarray, target: str) -> np.ndarray:
    """(N, 2) [own last price, own future price] for price target ``target`` (high / low / close)."""
    a, f = (LAST_COLUMNS.index(c) for c in _OWN[target])
    lc = np.asarray(last_candles, dtype=np.float64)
    return np.stack([lc[:, a], lc[:, f]], axis=1)


def anchor_column(state: dict, target: str) -> str:
    """The last_candles column the codec of ``state`` decodes head ``target`` against."""
    return f"last_{target}" if TARGET_CODECS[state["mode"]].anchor == "own" else "last_close"


def anchor_and_future(last_candles: np.ndarray, target: str, state: dict) -> np.ndarray:
    """(N, 2) [the codec's anchor, own future price] for price target ``target``."""
    lc = np.asarray(last_candles, dtype=np.float64)
    return np.stack([lc[:, LAST_COLUMNS.index(anchor_column(state, target))],
                     lc[:, LAST_COLUMNS.index(_OWN[target][1])]], axis=1)


def codec_name(dataset: dict) -> str:
    """The codec of a dataset: its ``reg_target_mode``, except a "return" artifact built with a TF logret_*
    price_norm_mode, whose labels are log(future / last_close) -> "logret_close"."""
    mode = dataset["reg_target_mode"]
    if mode == ReturnCodec.name and dataset.get("price_norm_mode") in LOGRET_NORM_MODES:
        return LogCloseCodec.name
    return mode


class ReturnCodec:
    name = "return"
    anchor = "own"

    def fit(self, train: dict) -> dict:
        """Nothing is learned; the state records the dataset's scale (REQUIRED contract key: a missing one raises,
        there is no silent 1.0) and clip (optional, None) stamps."""
        if "reg_target_scale" not in train:
            raise ValueError("the dataset lacks 'reg_target_scale' (a DATA_ARTIFACT_CONTRACT key): the return codec "
                             "cannot decode predictions without it")
        clip = train.get("reg_target_clip")
        return {"mode": self.name, "version": 1, "scale": float(train["reg_target_scale"]),
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


class LogCloseCodec(ReturnCodec):
    """y = clip(log(future / last_close), +-clip) * scale for EVERY head (TF logret_* artifacts); the anchor of every
    head is last_close."""
    name = "logret_close"
    anchor = "close"

    def encode(self, raw: np.ndarray, state: dict) -> np.ndarray:
        self._check(state)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("raw must be (N, 2) = [last_close, future price]")
        r = np.log(raw[:, 1] / raw[:, 0])
        if state["clip"] is not None:
            r = np.clip(r, -state["clip"], state["clip"])
        return r * state["scale"]

    def decode(self, pred: np.ndarray, state: dict, anchors: np.ndarray) -> np.ndarray:
        self._check(state)
        return np.asarray(anchors, dtype=np.float64) * np.exp(np.asarray(pred, dtype=np.float64) / state["scale"])

    def eval_spec(self, state: dict) -> dict:
        return {**super().eval_spec(state), "anchor": "last_close (every head)", "space": "log"}


TARGET_CODECS.register("return", ReturnCodec, version=1, supports_relative=True)
TARGET_CODECS.register("logret_close", LogCloseCodec, version=1, supports_relative=True)
