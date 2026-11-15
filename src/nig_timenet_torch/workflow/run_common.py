"""PURPOSE: Steps every run shares: load the artifact, read its facts, split by time, pick the device, build the run
    identity. Small functions of the settings so a notebook can call them one by one.
TAGS: run steps, load data, split plan, run identity, device, artifact facts, smoke mode
PITFALLS: every data fact (price targets, codec, timeframe, split dates) is read from the artifact, never assumed; the
    split uses the artifact's split_dates unless the settings give both instants (the baseline's split_bounds);
    check_decode refuses an artifact whose labels the codec does not decode to the real future prices (B-051).
SOURCE: ported from feat/ar-stream aa0f3dd (workflow/run_common.py); changes: strict split settings without
    retargeting, smoke stride (max_rows), artifact facts moved here, run identity from the manifest.
"""

import difflib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.colab.identity import identity_for_run
from nig_timenet_torch.core.schema import TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io.artifact import norm_coin
from nig_timenet_torch.data_io.loading import LoadedData, load_dataset
from nig_timenet_torch.data_io.split_plan import purge_embargo_split, sample_times
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, codec_name
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings

__all__ = ["load_data", "price_targets", "codec_state", "check_decode", "input_key", "split_bounds", "make_plan",
           "stride_rows", "pick_device", "code_sha", "run_identity"]


def load_data(settings: RunSettings) -> LoadedData:
    """The artifact ``settings.data`` names; rule D1 (a contract artifact) is checked by the loader."""
    return load_dataset(settings.data)


def price_targets(dataset: dict) -> tuple[str, ...]:
    """The artifact's ``price_targets``: a non-empty subset of ``TARGET_COLUMNS``, else raise (nothing is assumed)."""
    if "price_targets" not in dataset:
        raise ValueError("the artifact lacks 'price_targets' (a DATA_ARTIFACT_CONTRACT key)")
    targets = tuple(dataset["price_targets"])
    unknown = [t for t in targets if t not in TARGET_COLUMNS]
    if not targets or unknown or len(set(targets)) != len(targets):
        raise ValueError(f"artifact price_targets {list(targets)} must be distinct names from {TARGET_COLUMNS}")
    return targets


def codec_state(dataset: dict) -> dict:
    """The fitted state of the codec registered under the artifact's ``reg_target_mode`` (unknown mode raises
    UnknownNameError; a missing ``reg_target_scale`` raises ValueError)."""
    if "reg_target_mode" not in dataset:
        raise ValueError("the artifact lacks 'reg_target_mode' (a DATA_ARTIFACT_CONTRACT key)")
    return TARGET_CODECS[codec_name(dataset)]().fit(dataset)


def check_decode(dataset: dict, targets, n: int = 2000, rtol: float = 1e-4) -> None:
    """Rule B3 guard (B-051): the codec of ``dataset`` must decode its stored regression labels back to the REAL
    future prices of last_candles (evenly strided ``n`` rows, unclipped labels only); raises with the worst head."""
    state = codec_state(dataset)
    codec = TARGET_CODECS[state["mode"]]()
    lc = np.asarray(dataset["last_candles"])
    rows = stride_rows(np.arange(lc.shape[0]), n)
    for t in targets:
        y = np.asarray(dataset[head_spec("nig_regression", t).label_key])[rows].astype(np.float64)
        raw = anchor_and_future(lc[rows], t, state)
        ok = np.abs(y / state["scale"]) < 0.999
        dec = codec.decode(y[ok], state, raw[ok, 0])
        err = float(np.max(np.abs(dec / raw[ok, 1] - 1.0))) if ok.any() else 0.0
        if err > rtol:
            raise ValueError(f"codec {state['mode']!r} does not decode the {t!r} labels to the future prices of "
                             f"last_candles (max relative error {err:.2e} > {rtol:g}); the artifact's target "
                             f"definition (reg_target_mode / price_norm_mode) is not the codec's")


def input_key(dataset: dict) -> str:
    """``X_<tf>`` of a single-timeframe artifact (the TF main dataset is 4h; the test fixture is 1h)."""
    tfs = list(dataset["timeframes"])
    if len(tfs) != 1:
        raise ValueError(f"the flat run handles a single-timeframe artifact, got {tfs}")
    return f"X_{tfs[0]}"


def split_bounds(dataset: dict, split: SplitSettings) -> tuple:
    """(train_end, val_end): the settings' instants, else the artifact's ``split_dates``, else the share quantiles of
    the window times (TF's ``raw_quantile`` method; the TF main dataset leaves ``split_dates`` None)."""
    if split.train_end:
        return split.train_end, split.val_end
    sd = dataset.get("split_dates")
    if sd:
        return sd["train_end"], sd["val_end"]
    ts = np.sort(np.asarray(dataset["last_candles"])[:, TS_COL].astype(np.int64))
    q = [ts[min(len(ts) - 1, int(len(ts) * f))] for f in (split.train_share, split.train_share + split.val_share)]
    return tuple(pd.Timestamp(int(v), tz="UTC") for v in q)


def stride_rows(idx: np.ndarray, max_rows: int | None) -> np.ndarray:
    """At most ``max_rows`` of ``idx``, evenly strided over its whole range (smoke mode); ``None`` keeps all."""
    if max_rows is None or len(idx) <= max_rows:
        return idx
    return idx[np.unique(np.linspace(0, len(idx) - 1, max_rows).astype(np.int64))]


def excluded_rows(dataset: dict, coins) -> np.ndarray:
    """Boolean mask over ALL rows: True for the rows of the coins named in ``coins`` (``split.exclude_coins``). A name
    that is not a coin of the artifact raises (with a did-you-mean); excluding every coin raises."""
    if not coins:
        return np.zeros(np.asarray(dataset["last_candles"]).shape[0], dtype=bool)
    keys = {norm_coin(b["name"]): b for b in dataset["asset_bounds"]}
    wanted = {norm_coin(c): c for c in coins}
    unknown = sorted(c for k, c in wanted.items() if k not in keys)
    if unknown:
        close = {c: difflib.get_close_matches(norm_coin(c), list(keys), n=1) for c in unknown}
        raise ValueError("split.exclude_coins: not coins of this artifact: "
                         + ", ".join(f"{c!r}" + (f" (did you mean {m[0]!r}?)" if m else "") for c, m in close.items()))
    if len(wanted) >= len(keys):
        raise ValueError("split.exclude_coins removes every coin of the artifact")
    mask = np.zeros(np.asarray(dataset["last_candles"]).shape[0], dtype=bool)
    for k in wanted:
        mask[keys[k]["start"]:keys[k]["end"]] = True
    return mask


def make_plan(dataset: dict, split: SplitSettings) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple]:
    """((train_idx, val_idx, test_idx) with the smoke stride applied, (train_end, val_end)); the split purges by label
    end plus the embargo."""
    fs, fe, le = sample_times(dataset)
    train_end, val_end = split_bounds(dataset, split)
    plan = purge_embargo_split(le, fs, train_end, train_end, val_end, val_end, pd.Timedelta(hours=split.embargo_hours),
                               feature_end=fe)
    drop = excluded_rows(dataset, split.exclude_coins)  # after the time split: the cut dates do not move
    idx = tuple(stride_rows(i[~drop[i]], split.max_rows) for i in (plan.train_idx, plan.val_idx, plan.test_idx))
    return idx, (train_end, val_end)


def pick_device(settings: RunSettings) -> torch.device:
    d = settings.train.device
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d)


def code_sha(settings: RunSettings) -> str:
    """The commit recorded in the run identity: the settings' ``code_sha``, else the package checkout's git HEAD."""
    if settings.code_sha:
        return settings.code_sha
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_identity(trainer, fit_settings, model_settings, art, settings: RunSettings) -> RunIdentity:
    """The identity of the run the built ``trainer`` is about to fit (colab/identity.py is the one builder); ``art`` is
    the loaded artifact (its manifest gives the data fingerprint)."""
    return identity_for_run(trainer, fit_settings, model_settings, art.manifest, code_sha(settings))
