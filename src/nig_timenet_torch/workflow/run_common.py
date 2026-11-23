"""PURPOSE: Steps every run shares: load the artifact, read its facts, split by time, pick the device, build the run
    identity (with the run context of D-042). Small functions of the settings so a notebook can call them one by one.
TAGS: run steps, load data, split plan, run identity, device, artifact facts, smoke mode
PITFALLS: every data fact (price targets, codec, timeframe, split dates) is read from the artifact, never assumed; the
    split uses the artifact's split_dates unless the settings give both instants (the baseline's split_bounds);
    check_decode refuses an artifact whose labels the codec does not decode to the real future prices (B-051); a new
    RunSettings field that changes what is learned and reaches neither FitSettings, the trainer config nor the
    optimizer belongs in IDENTITY_SELECTION / IDENTITY_CONFIGURATION (tests/workflow/test_run_identity_context.py).
SOURCE: ported from feat/ar-stream aa0f3dd (workflow/run_common.py); changes: strict split settings, retargeting of
    the split rows through data.targets.retarget_splits (``retarget_art``, split.target_mode), smoke stride
    (max_rows), artifact facts moved here, run identity from the manifest.
"""

import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.colab.identity import identity_for_run
from nig_timenet_torch.core.registry import unknown_names_text
from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data.targets import retarget_splits, row_anchors  # also registers the "target_raw" codec
from nig_timenet_torch.data_io.artifact import norm_coin
from nig_timenet_torch.data_io.asset_index import asset_ids
from nig_timenet_torch.data_io.loading import LoadedData, load_dataset
from nig_timenet_torch.data_io.split_plan import purge_embargo_split, sample_times
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_column, codec_name, future_column
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings
from nig_timenet_torch.workflow.settings_roles import RESUMABLE_TRAIN_FIELDS

__all__ = ["load_data", "price_targets", "codec_state", "decode_anchors", "check_decode", "retarget_art", "input_key",
           "split_bounds", "make_plan", "stride_rows", "select_rows", "pick_device", "code_sha", "run_identity",
           "identity_context", "IDENTITY_SELECTION", "IDENTITY_CONFIGURATION", "RESUMABLE_TRAIN_FIELDS", "json_default",
           "write_json"]


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


def decode_anchors(dataset: dict, idx: np.ndarray, targets, state: dict, *, truth: bool = False) -> dict:
    """{target: what ``codec.decode(pred, state, anchors[target])`` needs} for the rows ``idx``: the codec's anchor
    column (an array of prices) or, for the target-mode codec, ``data.targets.row_anchors`` ((n, 2); ``truth`` sets
    how the sign / centre the labels dropped are filled, see there)."""
    if getattr(TARGET_CODECS[state["mode"]], "row_anchors", False):
        return row_anchors(dataset, idx, targets, truth=truth)
    lc = np.asarray(dataset["last_candles"])
    return {t: lc[idx, LAST_COLUMN_INDEX[anchor_column(state, t)]] for t in targets}


def check_decode(dataset: dict, targets, n: int = 2000, rtol: float = 1e-4) -> None:
    """Rule B3 guard (B-051): the codec of ``dataset`` must decode its stored regression labels back to the REAL
    future prices of last_candles (evenly strided ``n`` rows, unclipped labels only); raises with the worst head."""
    state = codec_state(dataset)
    codec = TARGET_CODECS[state["mode"]]()
    lc = np.asarray(dataset["last_candles"])
    rows = stride_rows(np.arange(lc.shape[0]), n)
    anchors = decode_anchors(dataset, rows, targets, state, truth=True)
    for t in targets:
        y = np.asarray(dataset[head_spec("nig_regression", t).label_key])[rows].astype(np.float64)
        future = lc[rows, LAST_COLUMN_INDEX[future_column(t)]].astype(np.float64)
        scale = state["scales"][t] if "scales" in state else state["scale"]
        ok = np.abs(y / scale) < 0.999 * (state["clip"] or 1.0)
        dec = codec.decode(y[ok], state, anchors[t][ok])
        err = float(np.max(np.abs(dec / future[ok] - 1.0))) if ok.any() else 0.0
        if err > rtol:
            raise ValueError(f"codec {state['mode']!r} does not decode the {t!r} labels to the future prices of "
                             f"last_candles (max relative error {err:.2e} > {rtol:g}); the artifact's target "
                             f"definition (reg_target_mode / price_norm_mode) is not the codec's")


def retarget_art(art: LoadedData, split: SplitSettings, targets, parts: tuple) -> LoadedData:
    """``art`` with the regression and class labels of the rows ``parts`` = (train, val, test) rebuilt in the
    ``split.target_mode`` (``data.targets.retarget_splits``: statistics fitted on the TRAIN rows, the stamps and the
    '+relative' centres stored); the new labels must decode to the real future prices (``check_decode``). A split with
    no ``target_mode`` returns ``art`` itself, so the default run is untouched."""
    if split.target_mode is None:
        return art
    ds, report = retarget_splits(art.dataset, parts, split.target_mode, targets=targets, clip=split.target_clip,
                                 close_reg=split.entry_close_reg)
    check_decode(ds, targets)
    shown = {k: report[k] for k in ("mode", "clip", "scales", "thresholds", "share_small_groups") if k in report}
    print(f"target mode {split.target_mode!r} (fitted on the train rows): {shown}", flush=True)
    return dataclasses.replace(art, dataset=ds)


def input_key(dataset: dict) -> str:
    """``X_<tf>`` of a single-timeframe artifact (the TF main dataset is 4h; the test fixture is 1h)."""
    tfs = list(dataset["timeframes"])
    if len(tfs) != 1:
        raise ValueError(f"the flat run handles a single-timeframe artifact, got {tfs}")
    return f"X_{tfs[0]}"


def input_keys(dataset: dict, timeframes=()) -> list[str]:
    """The ``X_<tf>`` keys a run reads: ``()`` = ``[input_key(dataset)]`` (one-frame artifact, as before); else one
    key per named frame, in that order, all of the artifact and all with the same window length (they are put side
    by side on the channel axis, owner task 2026-11-18)."""
    if not timeframes:
        return [input_key(dataset)]
    missing = [tf for tf in timeframes if tf not in dataset["timeframes"]]
    if missing:
        raise ValueError(f"input_timeframes {list(timeframes)}: the artifact has no {missing} "
                         f"(it holds {list(dataset['timeframes'])})")
    keys = [f"X_{tf}" for tf in timeframes]
    lengths = {k: int(np.asarray(dataset[k]).shape[1]) for k in keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"input_timeframes need windows of one length to stand side by side, got {lengths}")
    return keys


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


def stride_groups(gid: np.ndarray, max_rows: int | None) -> np.ndarray:
    """Positions of at most about ``max_rows`` rows made of WHOLE groups (``gid``: the rows' group ids, e.g. the
    timestamp ids of a cross-section run), the groups evenly strided over their range; ``None`` keeps all."""
    pos = np.arange(len(gid))
    if max_rows is None or len(gid) <= max_rows:
        return pos
    uniq = np.unique(gid)
    keep = max(1, int(round(len(uniq) * max_rows / len(gid))))
    chosen = uniq[np.unique(np.linspace(0, len(uniq) - 1, keep).astype(np.int64))]
    return pos[np.isin(gid, chosen)]


def excluded_ids(dataset: dict, coins) -> np.ndarray:
    """The asset ids (positions in ``asset_bounds``) of the coins named in ``coins`` (``split.exclude_coins``). A name
    that is not a coin of the artifact raises (with a did-you-mean); excluding every coin raises."""
    if not coins:
        return np.zeros(0, dtype=np.int64)
    keys = {norm_coin(b["name"]): i for i, b in enumerate(dataset["asset_bounds"])}
    wanted = {norm_coin(c): c for c in coins}
    unknown = sorted(c for k, c in wanted.items() if k not in keys)
    if unknown:
        raise ValueError("split.exclude_coins: not coins of this artifact: "
                         + unknown_names_text(unknown, keys, match=norm_coin))
    if len(wanted) >= len(keys):
        raise ValueError("split.exclude_coins removes every coin of the artifact")
    return np.array(sorted(keys[k] for k in wanted), dtype=np.int64)


def make_plan(dataset: dict, split: SplitSettings) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple]:
    """((train_idx, val_idx, test_idx) with the smoke stride applied, (train_end, val_end)); the split purges by label
    end plus the embargo."""
    fs, fe, le = sample_times(dataset)
    train_end, val_end = split_bounds(dataset, split)
    plan = purge_embargo_split(le, fs, train_end, train_end, val_end, val_end, pd.Timedelta(hours=split.embargo_hours),
                               feature_end=fe)
    return select_rows(dataset, split, (plan.train_idx, plan.val_idx, plan.test_idx)), (train_end, val_end)


def select_rows(dataset: dict, split: SplitSettings, parts: tuple) -> tuple:
    """The (train, val, test) rows a run uses out of the time split ``parts``: ``exclude_coins`` (after the time split,
    the cut dates do not move), ``top_coins`` (ranked on the TRAIN rows), then the smoke stride / cross-section order.
    A rolling fold (workflow/rolling_run.py) goes through the same function."""
    drop = excluded_ids(dataset, split.exclude_coins)
    parts = top_coin_rows(dataset, tuple(i[~np.isin(asset_ids(dataset, i), drop)] for i in parts), split.top_coins)
    pick = (lambda i: cross_section_rows(dataset, i, split.max_rows)) if split.cross_section else (
        lambda i: stride_rows(i, split.max_rows))
    return tuple(pick(i) for i in parts)


def top_coin_rows(dataset: dict, parts: tuple, top: int | None) -> tuple:
    """(train, val, test) rows of the ``top`` coins with the most TRAIN rows (ties: artifact order); None = as given.
    The ranking reads the train rows only, so nothing of val / test chooses the universe."""
    if top is None:
        return parts
    counts = np.bincount(asset_ids(dataset, parts[0]), minlength=len(dataset["asset_bounds"]))
    keep = np.argsort(-counts, kind="stable")[:top]
    return tuple(np.asarray(p)[np.isin(asset_ids(dataset, p), keep)] for p in parts)


def cross_section_rows(dataset: dict, idx: np.ndarray, max_rows: int | None) -> np.ndarray:
    """``idx`` in time-major order (all coins of a timestamp together, a stable sort) and, with ``max_rows``, only as
    many WHOLE timestamps, evenly strided over the dates, as hold about ``max_rows`` rows."""
    idx = np.asarray(idx)
    times = np.asarray(dataset["last_candles"])[idx, TS_COL].astype(np.int64)
    order = np.argsort(times, kind="stable")
    idx, times = idx[order], times[order]
    if max_rows is None or len(idx) <= max_rows:
        return idx
    uniq = np.unique(times)
    keep = max(1, int(round(len(uniq) * max_rows / len(idx))))
    chosen = uniq[np.unique(np.linspace(0, len(uniq) - 1, keep).astype(np.int64))]
    return idx[np.isin(times, chosen)]


def write_json(path, obj) -> None:
    """``obj`` as indented JSON (numpy numbers as Python ones through ``default=float``) in ``path``, UTF-8: the one
    writer of the run, study and report json files (C-18)."""
    Path(path).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def json_default(v):
    """``json.dumps(default=...)`` for numpy scalars (a numpy bool or number becomes the Python one; else ``str``)."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


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


# RESUMABLE_TRAIN_FIELDS (TrainSettings fields a resume may change: neither the run identity nor a rolling study hash
# holds them; tests/workflow/test_run_identity_context.py) comes from the one role table, workflow/settings_roles.py
#: D-042 (HB2, audit 2026-11b C-03): the settings that choose WHICH rows and inputs a run sees (every split field, the
#: input frames). Part of the run identity, not of the gate configuration key (gates run on their own synthetic data).
IDENTITY_SELECTION = {"split": tuple(f.name for f in dataclasses.fields(SplitSettings)),
                      "data": ("input_timeframes",)}
#: D-042 (C-04, C-05): training settings that reach neither ``FitSettings`` nor the trainer config nor the optimizer:
#: how the best epoch is chosen and the staged plan. Part of the identity AND of the gate configuration key.
IDENTITY_CONFIGURATION = {"train": ("selection_metric", "staged")}


def _plain(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return list(value) if isinstance(value, tuple) else value


def identity_context(settings: RunSettings, *, selection: bool = True) -> dict:
    """The run context hashed into the run identity (``build_run_identity(run_context=...)``, D-042):
    ``{section: {field: value}}`` of the ``IDENTITY_SELECTION`` (when ``selection``) and ``IDENTITY_CONFIGURATION``
    fields that differ from their defaults; a field at its default and an empty section are left out, so a run on the
    defaults keeps the identity it had before D-042 and its checkpoints resume (B-091)."""
    parts = {**(IDENTITY_SELECTION if selection else {}), **IDENTITY_CONFIGURATION}
    out = {}
    for section, names in parts.items():
        obj = getattr(settings, section)
        default = type(obj)()
        kept = {n: _plain(getattr(obj, n)) for n in names if getattr(obj, n) != getattr(default, n)}
        if kept:
            out[section] = kept
    return out


def run_identity(trainer, fit_settings, model_settings, art, settings: RunSettings) -> RunIdentity:
    """The identity of the run the built ``trainer`` is about to fit (colab/identity.py is the one builder); ``art`` is
    the loaded artifact (its manifest gives the data fingerprint), ``settings`` gives the run context
    (``identity_context``: split, input frames, selection metric, staged plan where not at their defaults)."""
    return identity_for_run(trainer, fit_settings, model_settings, art.manifest, code_sha(settings),
                            run_context=identity_context(settings))
