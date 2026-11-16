"""PURPOSE: Test environments of MODEL_DEV_PROTOCOL section 3: E2 = one global label permutation, E3 = a
    training-era slice of the real artifact, E3p = its panel form (whole timestamps); nig-envs writes E1 .. E4.
TAGS: e2 environment, e3 environment, shuffled labels, label permutation, real slice, panel slice, whole timestamps,
    training era, purged split, leakage canary, gates
PITFALLS: E2 permutes the regression AND class labels with one permutation and rewrites the future columns of
    last_candles from the permuted labels (so the codec decodes them, check_decode); the oracle_* columns are dropped
    (they describe the source rows, not the permuted labels). E3 needs the full run's train end from the caller
    (workflow.run_common.split_bounds); its rows end (label_end) at or before it, the full run's train membership.
    E3p keeps a timestamp only with EVERY chosen coin in it, so a coin with one missing bar is not chosen at all.
    ``main`` is the ``nig-envs`` console script (pyproject.toml), which is why nothing in the package imports it.

Both return a NEW dataset dict (row-aligned arrays sliced or copied, everything else shared) with an ``environment``
meta block; write it as a contract artifact with ``data.artifact_writer.write_artifact`` and read it back with
``data_io.load_dataset`` (rule D1), so the gates use the same code path as a real run.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.data_io.synthetic import ORACLE_PREFIX
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, codec_name, future_column
from nig_timenet_torch.model.heads import head_spec

__all__ = ["label_keys", "input_keys", "permute_labels", "real_slice", "panel_slice", "main"]

def label_keys(dataset: dict) -> list[str]:
    """The regression and class label keys of the dataset's price targets (from the head registry)."""
    return [head_spec(kind, t).label_key for t in dataset["price_targets"]
            for kind in ("nig_regression", "binary_classification")]


def input_keys(dataset: dict) -> list[str]:
    """What a model may read: the ``X_<tf>`` windows and ``base_params``."""
    return [f"X_{tf}" for tf in dataset["timeframes"]] + (["base_params"] if "base_params" in dataset else [])


def _rows(dataset: dict) -> int:
    return int(np.asarray(dataset["last_candles"]).shape[0])


def permute_labels(dataset: dict, *, seed: int = 0) -> dict:
    """E2: every label row moved by ONE global permutation, ``np.random.default_rng(seed).permutation(rows)`` (across
    coins and time); inputs and the input columns of
    last_candles untouched, its future columns rewritten as the codec's decode of the permuted labels."""
    n = _rows(dataset)
    perm = np.random.default_rng(seed).permutation(n)
    out = {k: v for k, v in dataset.items() if not k.startswith(ORACLE_PREFIX)}
    for k in label_keys(dataset):
        out[k] = np.asarray(dataset[k])[perm]
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    codec = TARGET_CODECS[state["mode"]]()
    last = np.array(dataset["last_candles"], dtype=np.float64, copy=True)
    for t in dataset["price_targets"]:
        anchor = anchor_and_future(last, t, state)[:, 0]
        y = np.asarray(out[head_spec("nig_regression", t).label_key], np.float64)
        last[:, LAST_COLUMNS.index(future_column(t))] = codec.decode(y, state, anchor)
    out["last_candles"] = last
    out["environment"] = {"name": "E2", "seed": int(seed), "permuted": label_keys(dataset),
                          "rewritten": [future_column(t) for t in dataset["price_targets"]],
                          "dropped": sorted(k for k in dataset if k.startswith(ORACLE_PREFIX)),
                          "base": dataset.get("environment", {}).get("name", "")}
    return out


def _iso(ns: int) -> str:
    return pd.Timestamp(int(ns), tz="UTC").isoformat()


def real_slice(dataset: dict, *, train_end, n_coins: int = 20, months: int = 6, min_coverage: float = 0.95,
               train_share: float = 0.70, val_share: float = 0.15) -> dict:
    """E3: the rows of ``n_coins`` coins whose window and label lie in the ``months`` months that end at ``train_end``
    (the FULL run's train end), with its own ``split_dates`` at the ``train_share`` / ``train_share + val_share``
    quantiles of the slice's window times (the run purges by label end). Coins: those with >= ``min_coverage`` of the
    best-covered coin's rows in the period, ranked by their median last-bar range log(high / low), taken at evenly
    spaced ranks (calm to violent). Raises if fewer coins qualify or if any row would end after ``train_end``."""
    if n_coins < 1 or months < 1 or not 0 < min_coverage <= 1 or not 0 < train_share < train_share + val_share < 1:
        raise ValueError("need n_coins >= 1, months >= 1, 0 < min_coverage <= 1, 0 < train_share < "
                         "train_share + val_share < 1")
    end = pd.Timestamp(train_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    start = end - pd.DateOffset(months=months)
    fs, _, le = sample_times(dataset)
    in_period = (fs >= start.value) & (le <= end.value)
    last = np.asarray(dataset["last_candles"])
    bounds = dataset["asset_bounds"]
    counts = np.array([in_period[b["start"]:b["end"]].sum() for b in bounds])
    eligible = np.flatnonzero(counts >= min_coverage * max(int(counts.max()), 1))
    if len(eligible) < n_coins:
        raise ValueError(f"only {len(eligible)} coins cover {min_coverage:.0%} of the period {start.isoformat()} .. "
                         f"{end.isoformat()}, {n_coins} requested")
    hi, lo = LAST_COLUMNS.index("last_high"), LAST_COLUMNS.index("last_low")
    vol = []
    for i in eligible:
        rows = bounds[i]["start"] + np.flatnonzero(in_period[bounds[i]["start"]:bounds[i]["end"]])
        vol.append(float(np.median(np.log(last[rows, hi] / last[rows, lo]))))
    ranked = eligible[np.argsort(vol, kind="stable")]
    chosen = sorted(ranked[np.unique(np.round(np.linspace(0, len(ranked) - 1, n_coins)).astype(int))])
    rows, new_bounds = [], []
    for i in chosen:
        b = bounds[i]
        r = b["start"] + np.flatnonzero(in_period[b["start"]:b["end"]])
        new_bounds.append({**b, "start": sum(map(len, rows)), "end": sum(map(len, rows)) + len(r)})
        rows.append(r)
    idx = np.concatenate(rows)
    if (le[idx] > end.value).any():                       # the guard of the protocol: never past the train end
        raise AssertionError("an E3 row ends after the full run's train end")
    n = _rows(dataset)
    out = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v)
           for k, v in dataset.items()}
    ts = np.sort(last[idx, TS_COL].astype(np.int64))
    q = [ts[min(len(ts) - 1, int(len(ts) * f))] for f in (train_share, train_share + val_share)]
    out["asset_bounds"] = new_bounds
    out["split_dates"] = {"train_end": _iso(q[0]), "val_end": _iso(q[1])}
    out["environment"] = {"name": "E3", "full_train_end": end.isoformat(), "period": [start.isoformat(),
                                                                                     end.isoformat()],
                          "months": int(months), "coins": [b["name"] for b in new_bounds], "rows": int(len(idx)),
                          "max_label_end": _iso(int(le[idx].max())), "split_shares": [train_share, val_share],
                          "selection": f"coverage >= {min_coverage}, evenly spaced ranks of the median bar range"}
    return out


def panel_slice(dataset: dict, *, train_end, n_coins: int = 50, max_rows: int = 60_000, train_share: float = 0.70,
                val_share: float = 0.15) -> dict:
    """E3p: a PANEL slice for multi-coin CPU looks. The period is the last ``max_rows // n_coins`` timestamps whose
    label ends at or before ``train_end`` (the FULL run's train end, so the full run's validation and test rows stay
    unseen); the coins are the ``n_coins`` with the most rows before ``train_end`` (the longest history; the artifact
    holds no volume) among those present at EVERY timestamp of the period. Every timestamp of the slice therefore
    holds every coin (whole cross-sections, ``SplitSettings.cross_section``). Own ``split_dates`` at the
    ``train_share`` / ``train_share + val_share`` quantiles of the timestamps. Raises if fewer coins qualify."""
    if n_coins < 2 or max_rows < 2 * n_coins or not 0 < train_share < train_share + val_share < 1:
        raise ValueError("need n_coins >= 2, max_rows >= 2 * n_coins, 0 < train_share < train_share + val_share < 1")
    end = pd.Timestamp(train_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    _, _, le = sample_times(dataset)
    last = np.asarray(dataset["last_candles"])
    ts = last[:, TS_COL].astype(np.int64)
    inside = le <= end.value
    stamps = np.unique(ts[inside])[-(max_rows // n_coins):]
    in_period = inside & (ts >= stamps[0])
    bounds = dataset["asset_bounds"]
    full = [i for i, b in enumerate(bounds) if int(in_period[b["start"]:b["end"]].sum()) == len(stamps)]
    if len(full) < n_coins:
        raise ValueError(f"only {len(full)} coins are present at all {len(stamps)} timestamps of the period "
                         f"{_iso(int(stamps[0]))} .. {end.isoformat()}, {n_coins} requested")
    history = [int(inside[bounds[i]["start"]:bounds[i]["end"]].sum()) for i in full]
    chosen = sorted(full[j] for j in np.argsort(-np.asarray(history), kind="stable")[:n_coins])
    rows, new_bounds = [], []
    for i in chosen:
        b = bounds[i]
        r = b["start"] + np.flatnonzero(in_period[b["start"]:b["end"]])
        new_bounds.append({**b, "start": len(rows) * len(stamps), "end": (len(rows) + 1) * len(stamps)})
        rows.append(r)
    idx = np.concatenate(rows)
    n = _rows(dataset)
    out = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v)
           for k, v in dataset.items()}
    q = [stamps[min(len(stamps) - 1, int(len(stamps) * f))] for f in (train_share, train_share + val_share)]
    out["asset_bounds"] = new_bounds
    out["split_dates"] = {"train_end": _iso(q[0]), "val_end": _iso(q[1])}
    out["environment"] = {"name": "E3p", "full_train_end": end.isoformat(),
                          "period": [_iso(int(stamps[0])), _iso(int(le[idx].max()))], "timestamps": int(len(stamps)),
                          "coins": [b["name"] for b in new_bounds], "rows": int(len(idx)),
                          "max_label_end": _iso(int(le[idx].max())), "split_shares": [train_share, val_share],
                          "selection": "present at every timestamp of the period, most rows before the train end"}
    return out


def main(argv: list[str] | None = None) -> int:
    """``nig-envs e1 OUT [--settings JSON]`` | ``nig-envs e2 SOURCE OUT [--seed N]`` | ``nig-envs e3 SOURCE OUT
    [--train-end ISO]`` | ``nig-envs e3p SOURCE OUT [--train-end ISO] [--coins N] [--max-rows N]`` | ``nig-envs e4
    OUT [--settings JSON]``: write the environment as a contract artifact (SOURCE is read through ``load_dataset``;
    the train end of E3 / E3p defaults to the full run's, ``run_common.split_bounds`` with the default
    SplitSettings; E4 = ``data_io.synthetic_panel``, its settings are PanelSyntheticSettings fields)."""
    from nig_timenet_torch.core.config import from_dict
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
    from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
    from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, write_panel_synthetic

    ap = argparse.ArgumentParser(prog="nig-envs", description="Write a test environment (E1 / E2 / E3 / E3p / E4).")
    ap.add_argument("env", choices=("e1", "e2", "e3", "e3p", "e4"))
    ap.add_argument("paths", nargs="+", help="e1 / e4: OUT; e2 / e3 / e3p: SOURCE OUT")
    ap.add_argument("--settings", default="{}", help="e1: SyntheticSettings fields as JSON; e4: PanelSyntheticSettings")
    ap.add_argument("--seed", type=int, default=0, help="e2: permutation seed")
    ap.add_argument("--train-end", default="", help="e3 / e3p: the full run's train end (default: its split_bounds)")
    ap.add_argument("--coins", type=int, default=50, help="e3p: coins of the panel slice")
    ap.add_argument("--max-rows", type=int, default=60_000, help="e3p: rows of the panel slice (whole timestamps)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    synthetic = {"e1": (SyntheticSettings, write_synthetic), "e4": (PanelSyntheticSettings, write_panel_synthetic)}
    if len(a.paths) != (1 if a.env in synthetic else 2):
        ap.error(f"{a.env} takes {'OUT' if a.env in synthetic else 'SOURCE OUT'}")
    if a.env in synthetic:
        settings_type, write = synthetic[a.env]
        man = write(a.paths[0], from_dict(settings_type, json.loads(a.settings)), overwrite=a.overwrite)
    else:
        src = load_dataset(LoaderSettings(name=a.paths[0], local_dir=""))
        ds = src.dataset
        if a.env == "e2":
            env = permute_labels(ds, seed=a.seed)
        else:
            from nig_timenet_torch.workflow.run_common import split_bounds
            from nig_timenet_torch.workflow.settings import SplitSettings
            train_end = a.train_end or split_bounds(ds, SplitSettings())[0]
            env = (real_slice(ds, train_end=train_end) if a.env == "e3"
                   else panel_slice(ds, train_end=train_end, n_coins=a.coins, max_rows=a.max_rows))
        env["environment"]["source_fingerprint"] = src.manifest.data_fingerprint
        man = write_artifact(env, a.paths[1], source_commit=src.manifest.source_commit, overwrite=a.overwrite,
                             producer=f"nig_timenet_torch.data_io.environments:{a.env}")
    print(f"wrote {a.env}: {man['export']['rows']} rows, {len(man['export']['coins'])} coins, "
          f"fingerprint {man['export']['data_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
