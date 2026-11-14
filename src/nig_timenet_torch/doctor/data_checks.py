"""PURPOSE: Card C: is the data sound? Label / window alignment, codec round trip, unit scales, split purity and shift,
    the information ceiling of the inputs, duplicates / effective sample size, coverage counts.
TAGS: doctor, card c, data checks, leakage, label alignment, codec round trip, split purity, distribution shift,
    input information ceiling, near duplicate windows, effective sample size, flat bars, clipped labels
PITFALLS: the alignment check needs an artifact whose inputs are the RAW log-move channels (high / low / close vs the
    previous close; ``price_norm_state`` "unfitted"): other artifacts report it as not applicable; the ceiling is a
    strong SIMPLE model (ridge and gradient boosting on HAR-style features), an upper-ish reference, not a proof that
    nothing better exists; the test split is read here only for composition, shift and the ceiling's own score.

``data_checks(ctx) -> list[Check]``; the pieces (``alignment``, ``ceiling``, ``overlap`` ...) work on a dataset dict
and row indices, so a script can call them on any artifact.
"""

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.schema import TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.doctor.common import (
    Check,
    DoctorContext,
    judge,
    predict_outputs,
    reg_key,
    ridge_oof,
    simple_features,
    stride,
)
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.layers import InstanceNorm
from nig_timenet_torch.stats.metrics import rank_ic, spearman
from nig_timenet_torch.workflow import run_common

__all__ = ["data_checks", "alignment", "ceiling", "overlap", "coverage", "shift", "batch_independence",
           "RAW_CHANNELS"]

RAW_CHANNELS = TARGET_COLUMNS
ALIGN_TOL = 2e-5            # float32 labels and sums of float32 log moves
SHIFT_BAND = (0.67, 1.5)    # sd ratio of a split against train, either way
CEILING_ROWS = 60_000       # train rows the simple models are fitted on


def _coin_of(dataset: dict, n: int) -> np.ndarray:
    coin = np.empty(n, dtype=np.int64)
    for i, b in enumerate(dataset["asset_bounds"]):
        coin[b["start"]:b["end"]] = i
    return coin


def _raw_logmoves(dataset: dict) -> bool:
    return (list(dataset["feature_order"]) == list(RAW_CHANNELS) and len(dataset["timeframes"]) == 1
            and dataset.get("price_norm_state") == "unfitted")


def alignment(dataset: dict, *, max_pairs: int = 20000) -> dict | None:
    """For pairs (row i, a later row j of the same coin, k bars later, horizon <= k <= window): (a) the bars the two
    windows share must be identical (a window is a pure slice of the coin's bars: no per-window state, no future);
    (b) the labels of row i must equal what the bars AFTER i say, read from window j: close = sum of the next h
    close moves, high / low = the max / min of the running close + that bar's high / low move. Returns the largest
    absolute errors (None when the inputs are not the raw log-move channels)."""
    if not _raw_logmoves(dataset):
        return None
    x = np.asarray(dataset[run_common.input_key(dataset)])
    lc = np.asarray(dataset["last_candles"])
    n, t = x.shape[0], x.shape[1]
    h = int(dataset["forecast_horizon"])
    bar = int(pd.Timedelta(dataset["base_tf"]).value)
    ts = lc[:, TS_COL].astype(np.int64)
    coin = _coin_of(dataset, n)
    out = {"pairs": 0, "window_err": 0.0}
    labels = {tg: np.asarray(dataset[reg_key(tg)]) for tg in dataset["price_targets"]}
    clip = dataset.get("reg_target_clip") or np.inf
    scale = float(dataset["reg_target_scale"])
    err = dict.fromkeys(labels, 0.0)
    for gap in range(1, 4):                                      # the next rows of the same coin
        i = stride(np.arange(n - gap), max_pairs)
        j = i + gap
        k = (ts[j] - ts[i]) // bar
        ok = (coin[i] == coin[j]) & (k >= h) & (k <= t - 1) & ((ts[j] - ts[i]) % bar == 0)
        for kk in np.unique(k[ok]):
            sel = ok & (k == kk)
            a, b = i[sel], j[sel]
            kk = int(kk)
            xi, xj = np.asarray(x[a], dtype=np.float64), np.asarray(x[b], dtype=np.float64)
            out["window_err"] = max(out["window_err"], float(np.abs(xi[:, kk:] - xj[:, :t - kk]).max()))
            fut = xj[:, t - kk:t - kk + h]                        # the h bars after row i: (pairs, h, 3)
            before = np.concatenate([np.zeros((len(a), 1)), np.cumsum(fut[:, :, 2], axis=1)[:, :-1]], axis=1)
            want = {"close": fut[:, :, 2].sum(axis=1), "high": (before + fut[:, :, 0]).max(axis=1),
                    "low": (before + fut[:, :, 1]).min(axis=1)}
            for tg, y in labels.items():
                got = np.asarray(y[a], dtype=np.float64) / scale
                free = np.abs(got) < 0.999 * clip                 # a clipped label cannot be rebuilt
                if free.any():
                    err[tg] = max(err[tg], float(np.abs(got[free] - want[tg][free]).max()))
            out["pairs"] += int(sel.sum())
        if out["pairs"] >= max_pairs // 2:
            break
    out["label_err"] = err
    return out


@torch.no_grad()
def batch_independence(model, x: torch.Tensor, rows: int = 64) -> float:
    """Largest change of the first ``rows`` rows' outputs (eval mode) when OTHER rows are added to the batch: an
    evaluated row must not depend on its batch mates (0 = independent)."""
    alone = predict_outputs(model, x[:rows])
    g = torch.Generator().manual_seed(0)
    mixed = torch.cat([x[:rows], x[rows:rows * 4] * 3.0 + torch.randn(x[rows:rows * 4].shape, generator=g)])
    both = predict_outputs(model, mixed)
    return max(float(np.abs(alone[k] - both[k][:rows]).max()) for k in alone)


def ceiling(dataset: dict, idx: dict, targets, *, boost: bool = True) -> dict:
    """What a strong SIMPLE model reads from the same windows: ridge (and, with scikit-learn, gradient boosting) on
    ``common.simple_features``, fitted on (strided) TRAIN rows, scored on VAL and TEST per head: pooled Spearman,
    rank IC by timestamp, R^2. A model below this lost information; a head whose ceiling is ~0 has nothing to learn
    from these inputs."""
    fo = list(dataset["feature_order"])
    xk = run_common.input_key(dataset)
    rows = {"train": stride(idx["train"], CEILING_ROWS), "val": stride(idx["val"], CEILING_ROWS),
            "test": stride(idx["test"], CEILING_ROWS)}
    feats = {s: simple_features(np.asarray(dataset[xk])[np.sort(r)], fo)[0] for s, r in rows.items()}
    ts = {s: np.asarray(dataset["last_candles"])[np.sort(r), TS_COL] for s, r in rows.items()}
    out: dict = {}
    for t in targets:
        y = {s: np.asarray(dataset[reg_key(t)])[np.sort(r)].astype(np.float64) for s, r in rows.items()}
        res = {}
        a = np.concatenate([feats["train"], feats["val"], feats["test"]])
        yy = np.concatenate([y["train"], y["val"], y["test"]])
        ntr, nva = len(y["train"]), len(y["val"])
        tr = np.arange(ntr)
        pred = ridge_oof(a, yy, [(tr, np.arange(ntr, len(yy)))])
        preds = {"ridge": {"val": pred[ntr:ntr + nva], "test": pred[ntr + nva:]}}
        if boost:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=0)
                m.fit(feats["train"], y["train"])
                preds["boost"] = {s: m.predict(feats[s]) for s in ("val", "test")}
            except ImportError:
                pass
        for name, p in preds.items():
            for s in ("val", "test"):
                sst = np.sum((y[s] - y[s].mean()) ** 2)
                res[f"{name}_{s}"] = {"ic": float(spearman(y[s], p[s])),
                                      "ic_by_ts": float(rank_ic(y[s], p[s], groups=ts[s])),
                                      "r2": float(1.0 - np.sum((y[s] - p[s]) ** 2) / sst) if sst > 0 else float("nan")}
        out[t] = res
    return out


def overlap(dataset: dict, idx_train: np.ndarray) -> dict:
    """How independent the training rows are: the share of a window shared with the next row of the coin, the share of
    a label's bars shared with the next row's label, the rows per coin that do not overlap at all, the mean pairwise
    correlation of same-timestamp close labels across coins and the effective number of independent coins it
    implies, and the share of exactly duplicated windows (strided sample)."""
    lc = np.asarray(dataset["last_candles"])
    ts = lc[:, TS_COL].astype(np.int64)
    n = len(ts)
    coin = _coin_of(dataset, n)
    bar = int(pd.Timedelta(dataset["base_tf"]).value)
    t = int(dataset["window_sizes"][dataset["base_tf"]])
    h = int(dataset["forecast_horizon"])
    same = coin[1:] == coin[:-1]
    step = float(np.median((ts[1:] - ts[:-1])[same] / bar)) if same.any() else float("nan")
    res = {"row_step_bars": step, "window_overlap": max(0.0, 1.0 - step / t), "label_overlap": max(0.0, 1.0 - step / h),
           "train_rows": int(len(idx_train))}
    y = np.asarray(dataset[reg_key("close")])[idx_train] if "close" in dataset["price_targets"] else None
    if y is not None:
        df = pd.DataFrame({"ts": ts[idx_train], "y": y})
        g = df.groupby("ts")["y"]
        cnt, mean = g.count(), g.mean()
        nbar = float(cnt.mean())
        var = float(df["y"].var())
        rho = (float(mean.var()) - var / nbar) / (var * (1.0 - 1.0 / nbar)) if nbar > 1 and var > 0 else float("nan")
        res.update(coins_per_timestamp=nbar, cross_coin_corr=rho,
                   effective_coins=nbar / (1.0 + (nbar - 1.0) * max(rho, 0.0)) if np.isfinite(rho) else float("nan"))
    x = np.asarray(dataset[run_common.input_key(dataset)])
    pick = np.sort(stride(idx_train, 100_000))
    flat = np.ascontiguousarray(np.asarray(x[pick]).reshape(len(pick), -1))
    keys = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1])))
    res["duplicate_window_share"] = float(1.0 - len(np.unique(keys)) / len(pick))
    return res


def coverage(dataset: dict, idx: np.ndarray, targets) -> dict:
    """Counts on the rows ``idx`` (strided): non-finite inputs / labels, flat bars (high move == low move), zero close
    moves, constant windows, clipped and exactly-zero labels."""
    x = np.asarray(dataset[run_common.input_key(dataset)])[np.sort(stride(idx, 100_000))].astype(np.float64)
    fo = list(dataset["feature_order"])
    res = {"nonfinite_inputs": int((~np.isfinite(x)).sum()), "constant_window_share": float((x.std(axis=1) == 0).all(
        axis=1).mean())}
    if "high" in fo and "low" in fo:
        res["flat_bar_share"] = float((x[:, :, fo.index("high")] == x[:, :, fo.index("low")]).mean())
    if "close" in fo:
        res["zero_close_move_share"] = float((x[:, :, fo.index("close")] == 0).mean())
    clip = dataset.get("reg_target_clip")
    scale = float(dataset["reg_target_scale"])
    for t in targets:
        y = np.asarray(dataset[reg_key(t)])[idx].astype(np.float64)
        res[f"{t}_nonfinite_labels"] = int((~np.isfinite(y)).sum())
        res[f"{t}_zero_label_share"] = float((y == 0).mean())
        res[f"{t}_clipped_label_share"] = float((np.abs(y / scale) >= 0.999 * clip).mean()) if clip else 0.0
    return res


def shift(dataset: dict, idx: dict, targets) -> dict:
    """Per split against TRAIN: sd ratio of every input channel and of every regression label, and the class balance."""
    xk = run_common.input_key(dataset)
    stats = {}
    for s, r in idx.items():
        r = np.sort(stride(r, 100_000))
        x = np.asarray(dataset[xk])[r].astype(np.float64)
        stats[s] = {"x_sd": x.reshape(-1, x.shape[-1]).std(axis=0),
                    "y_sd": {t: float(np.asarray(dataset[reg_key(t)])[r].std()) for t in targets},
                    "y_mean": {t: float(np.asarray(dataset[reg_key(t)])[r].mean()) for t in targets},
                    "up_share": {t: float(np.asarray(
                        dataset[head_spec("binary_classification", t).label_key])[r].mean())
                                 for t in targets}}
    out = {}
    for s in ("val", "test"):
        out[s] = {"x_sd_ratio": (stats[s]["x_sd"] / stats["train"]["x_sd"]).tolist(),
                  "y_sd_ratio": {t: stats[s]["y_sd"][t] / stats["train"]["y_sd"][t] for t in targets},
                  "y_mean_shift_sd": {t: (stats[s]["y_mean"][t] - stats["train"]["y_mean"][t])
                                      / stats["train"]["y_sd"][t]
                                      for t in targets},
                  "up_share": stats[s]["up_share"]}
    out["train"] = {"y_sd": stats["train"]["y_sd"], "y_mean": stats["train"]["y_mean"],
                    "x_sd": stats["train"]["x_sd"].tolist(), "up_share": stats["train"]["up_share"]}
    return out


def _unit_checks(ctx: DoctorContext, sh: dict) -> list[Check]:
    """Label and input scale against the model's unit-ful constants (the NIG floors, InstanceNorm eps)."""
    checks, model = [], ctx.model
    heads = model.head_modules() if hasattr(model, "head_modules") else {}
    for t, head in heads.items():
        s = head._s() if hasattr(head, "_s") else None
        s = 1.0 if s is None else float(s)
        sd = sh["train"]["y_sd"][t] / s
        floor_sd = float(np.sqrt(head.beta_min / max(head.alpha_min - 1.0, 1e-6)))
        v = judge(sd, fail_below=0.1, warn_below=0.3, warn_above=5.0, fail_above=20.0)
        checks.append(Check("C", "data/units", f"{t}: label sd in the head's units (label sd / head scale)", sd,
                            f"~1 (the head's floor sd is {floor_sd:.2f} in these units)", v,
                            "the NIG head's constants (beta floor, softplus offsets) are written for labels of sd ~1: "
                            "model override nig_scale='train_std' (or a codec scale)" if v != "pass" else ""))
    norms = [m for m in model.modules() if isinstance(m, InstanceNorm)]
    if norms:
        _, x, _ = ctx.xy("train", 20000)
        scaler = getattr(model, "input_scaler", None)
        if scaler is not None:
            with torch.no_grad():
                x = scaler(x.to(next(model.parameters()).device)).cpu()
        med = float(x.var(dim=1).median())
        ratio = med / norms[0].eps
        v = judge(ratio, fail_below=3.0, warn_below=30.0)
        checks.append(Check("C", "data/units", "median per-window input variance / InstanceNorm eps", ratio,
                            ">= 30 (eps must be negligible)", v,
                            "eps is the size of the window variance: the per-window normalisation is damped and its "
                            "log-sd statistic is floored; model override input_scaler='train_std'" if v != "pass"
                            else ""))
    return checks


def data_checks(ctx: DoctorContext) -> list[Check]:
    ds, r = ctx.dataset, ctx.run
    checks: list[Check] = []
    # C1 alignment / leakage
    al = alignment(ds)
    if al is None or not al["pairs"]:
        checks.append(Check("C", "data/alignment", "labels rebuilt from later windows", "not applicable", "", "info",
                            "needs raw log-move channels (high, low, close) and rows of one coin within a window"))
    else:
        checks.append(Check("C", "data/alignment", "shared bars of overlapping windows identical (max abs diff)",
                            al["window_err"], f"<= {ALIGN_TOL} on {al['pairs']} pairs",
                            "pass" if al["window_err"] <= ALIGN_TOL else "fail",
                            "" if al["window_err"] <= ALIGN_TOL else "a window is not a pure slice of the coin's bars "
                            "(per-window state, misaligned rows, or future bars inside a window)"))
        for t, e in al["label_err"].items():
            checks.append(Check("C", "data/alignment", f"{t} label == the next bars read from a later window (max err)",
                                e, f"<= {ALIGN_TOL}", "pass" if e <= ALIGN_TOL else "fail",
                                "" if e <= ALIGN_TOL else "the label is not the stated function of the bars after the "
                                "window: horizon, anchor or row alignment is off"))
    _, xv, _ = ctx.xy("val", 512)
    dep = batch_independence(ctx.model, xv)
    checks.append(Check("C", "data/alignment", "an evaluated row is independent of its batch mates (max output diff)",
                        dep, "<= 1e-5", "pass" if dep <= 1e-5 else "fail",
                        "" if dep <= 1e-5 else "eval-mode outputs depend on other rows of the batch (a normalisation "
                        "in batch mode, or a cross-row path): a leak at prediction time"))
    # C2 codec and units
    try:
        run_common.check_decode(ds, ctx.targets)
        checks.append(Check("C", "data/units", "codec decodes the stored labels to the future prices",
                            "ok", "", "pass"))
    except ValueError as e:
        checks.append(Check("C", "data/units", "codec decodes the stored labels to the future prices", "FAILS", "",
                            "fail", str(e)[:200]))
    if "close" in ctx.targets:
        y = np.asarray(ds[reg_key("close")])[r.idx["train"]]
        c = np.asarray(ds[head_spec("binary_classification", "close").label_key])[r.idx["train"]]
        bad = float(((y > 0) != (c > 0.5))[y != 0].mean())
        checks.append(Check("C", "data/units", "close class label == sign of the close regression label (mismatch)",
                            bad, "0", "pass" if bad < 1e-4 else "fail",
                            "" if bad < 1e-4 else "the two labels of one head disagree: they were built from different "
                            "bars or anchors"))
    sh = shift(ds, r.idx, ctx.targets)
    checks += _unit_checks(ctx, sh)
    # C3 split
    fs, _, le = sample_times(ds)
    for a, b in (("train", "val"), ("val", "test")):
        if len(r.idx[a]) and len(r.idx[b]):
            gap = (fs[r.idx[b]].min() - le[r.idx[a]].max()) / pd.Timedelta(ds["base_tf"]).value
            checks.append(Check("C", "data/split", f"{a} -> {b}: bars between the last label and the first window", gap,
                                ">= 0 (purged)", "pass" if gap >= 0 else "fail",
                                "" if gap >= 0 else "a later split starts before an earlier label ends: leakage"))
    coin = _coin_of(ds, len(fs))
    coins = {s: set(np.unique(coin[i]).tolist()) for s, i in r.idx.items()}
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    span = {s: (str(pd.Timestamp(int(ts[i].min()), tz="UTC").date()),
                str(pd.Timestamp(int(ts[i].max()), tz="UTC").date()))
            for s, i in r.idx.items() if len(i)}
    checks.append(Check("C", "data/split", "rows train / val / test (coins; coins unseen in train)",
                        " / ".join(str(len(r.idx[s])) for s in ("train", "val", "test")),
                        f"coins {len(coins['train'])} / {len(coins['val'])} / {len(coins['test'])}; unseen "
                        f"{len(coins['val'] - coins['train'])} / {len(coins['test'] - coins['train'])}", "info", "",
                        {"span": span}))
    # C4 shift
    for s in ("val", "test"):
        ratios = sh[s]["x_sd_ratio"] + list(sh[s]["y_sd_ratio"].values())
        worst_r = max(ratios, key=lambda v: abs(np.log(v)))
        ok = SHIFT_BAND[0] <= worst_r <= SHIFT_BAND[1]
        checks.append(Check("C", "data/shift", f"{s} vs train: worst sd ratio over inputs and labels", worst_r,
                            f"{SHIFT_BAND}", "pass" if ok else "warn",
                            "" if ok else "the split lives in another volatility regime than train: errors and "
                            "intervals measured on it mix skill with the regime change", sh[s]))
    # C5 ceiling
    ce = ceiling(ds, r.idx, ctx.targets, boost=True)
    ctx.notes["ceiling"] = ce
    for t, res in ce.items():
        best = max(res, key=lambda k: res[k]["ic_by_ts"] if k.endswith("_val") else -9)
        b = res[best]
        tb = res[best.replace("_val", "_test")]
        weak = b["ic_by_ts"] < 0.02
        checks.append(Check("C", "data/information", f"{t}: simple-model ceiling, rank IC by timestamp val / test "
                            f"({best.split('_')[0]})", f"{b['ic_by_ts']:.3f} / {tb['ic_by_ts']:.3f}",
                            f"pooled IC {b['ic']:.3f}, R2 {b['r2']:.3f}", "warn" if weak else "info",
                            "the inputs hold no readable signal for this label: no architecture or trainer change "
                            "fixes that, the data needs new inputs (or the target is unpredictable)" if weak else "",
                            res))
    # C6 overlap / memorisation risk
    ov = overlap(ds, r.idx["train"])
    n_par = sum(p.numel() for p in ctx.model.parameters())
    eff = ov["train_rows"] * (1.0 - ov["label_overlap"])
    if np.isfinite(ov.get("effective_coins", float("nan"))):
        eff *= ov["effective_coins"] / max(ov["coins_per_timestamp"], 1.0)
    v = judge(n_par / max(eff, 1.0), warn_above=1.0)
    checks.append(Check("C", "data/independence", "parameters per effectively independent training label",
                        n_par / max(eff, 1.0), f"< 1; {ov['train_rows']} rows = ~{eff:.0f} independent labels (label "
                        "overlap in time x coins moving together)", v,
                        "more parameters than independent labels: the model can memorise the training noise; shrink "
                        "it, regularise, or add independent data (more history, de-correlated targets)"
                        if v != "pass" else "", {**ov, "parameters": n_par, "effective_labels": eff}))
    checks.append(Check("C", "data/independence", "window overlap with the next row / label overlap / exact duplicates",
                        f"{ov['window_overlap']:.2f} / {ov['label_overlap']:.2f} / {ov['duplicate_window_share']:.4f}",
                        "0 = independent rows", "info", "", ov))
    # C7 coverage
    cov = coverage(ds, r.idx["train"], ctx.targets)
    bad = cov["nonfinite_inputs"] + sum(v for k, v in cov.items() if k.endswith("nonfinite_labels"))
    checks.append(Check("C", "data/coverage", "non-finite inputs or labels (train)", bad, "0",
                        "pass" if bad == 0 else "fail", "" if bad == 0 else "NaN / inf in the artifact", cov))
    clipped = max(v for k, v in cov.items() if k.endswith("clipped_label_share"))
    checks.append(Check("C", "data/coverage", "flat bars / zero close moves / constant windows / clipped labels",
                        f"{cov.get('flat_bar_share', float('nan')):.4f} / {cov.get('zero_close_move_share', 0):.4f} / "
                        f"{cov['constant_window_share']:.4f} / {clipped:.5f}", "shares of bars or rows", "info",
                        "", cov))
    # batch composition of a shuffled epoch
    bs = r.fit_settings.batch_size
    order = np.random.default_rng(0).permutation(len(r.idx["train"]))[:bs * 20].reshape(20, -1)
    per = [len(np.unique(coin[r.idx["train"][b]])) for b in order]
    checks.append(Check("C", "data/batches", "distinct coins per shuffled batch (mean of 20)", float(np.mean(per)),
                        f"{len(coins['train'])} coins, batch {bs}", "info"))
    return checks
