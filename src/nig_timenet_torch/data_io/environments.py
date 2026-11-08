"""PURPOSE: Test environments of MODEL_DEV_PROTOCOL section 3: E2 = one global label permutation (inputs untouched),
    E3 = a training-era slice of the real artifact (coins x months, own purged split); nig-envs writes E1 / E2 / E3.
TAGS: e2 environment, e3 environment, shuffled labels, label permutation, real slice, training era, purged split,
    leakage canary, gates
PITFALLS: E2 permutes the regression AND class labels with one permutation and rewrites the future columns of
    last_candles from the permuted labels (so the codec decodes them, check_decode); the oracle_* columns are dropped
    (they describe the source rows, not the permuted labels). E3 needs the full run's train end from the caller
    (workflow.run_common.split_bounds); its rows end (label_end) at or before it, the full run's train membership.
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

__all__ = ["label_keys", "input_keys", "permute_labels", "real_slice", "main"]

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


def main(argv: list[str] | None = None) -> int:
    """``nig-envs e1 OUT [--settings JSON]`` | ``nig-envs e2 SOURCE OUT [--seed N]`` | ``nig-envs e3 SOURCE OUT
    [--train-end ISO]``: write the environment as a contract artifact (SOURCE is read through ``load_dataset``; E3's
    train end defaults to the full run's, ``run_common.split_bounds`` with the default SplitSettings)."""
    from nig_timenet_torch.core.config import from_dict
    from nig_timenet_torch.data.artifact_writer import write_artifact
    from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
    from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic

    ap = argparse.ArgumentParser(prog="nig-envs", description="Write a gate test environment (E1 / E2 / E3).")
    ap.add_argument("env", choices=("e1", "e2", "e3"))
    ap.add_argument("paths", nargs="+", help="e1: OUT; e2 / e3: SOURCE OUT")
    ap.add_argument("--settings", default="{}", help="e1: SyntheticSettings fields as JSON")
    ap.add_argument("--seed", type=int, default=0, help="e2: permutation seed")
    ap.add_argument("--train-end", default="", help="e3: the full run's train end (default: its split_bounds)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if len(a.paths) != (1 if a.env == "e1" else 2):
        ap.error(f"{a.env} takes {'OUT' if a.env == 'e1' else 'SOURCE OUT'}")
    if a.env == "e1":
        man = write_synthetic(a.paths[0], from_dict(SyntheticSettings, json.loads(a.settings)), overwrite=a.overwrite)
    else:
        src = load_dataset(LoaderSettings(name=a.paths[0], local_dir=""))
        ds = src.dataset
        if a.env == "e2":
            env = permute_labels(ds, seed=a.seed)
        else:
            from nig_timenet_torch.workflow.run_common import split_bounds
            from nig_timenet_torch.workflow.settings import SplitSettings
            env = real_slice(ds, train_end=a.train_end or split_bounds(ds, SplitSettings())[0])
        env["environment"]["source_fingerprint"] = src.manifest.data_fingerprint
        man = write_artifact(env, a.paths[1], source_commit=src.manifest.source_commit, overwrite=a.overwrite,
                             producer=f"nig_timenet_torch.data_io.environments:{a.env}")
    print(f"wrote {a.env}: {man['export']['rows']} rows, {len(man['export']['coins'])} coins, "
          f"fingerprint {man['export']['data_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
