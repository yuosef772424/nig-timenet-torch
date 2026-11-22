"""M3 improve (PREREGISTERED.md, 2026-11-20): leakage check, model-free variants (H3 ridge, H4 beta) and the reading.

    python m3_improve.py leak RAW_DIR RAW_ART RES_ART
    python m3_improve.py variants RAW_DIR RAW_ART RES_ART RG_ART
    python m3_improve.py detail --base OUT_DIR [--base ...] [--arm NAME=OUT_DIR ...] --raw RAW_DIR --res-art RES_ART
    python m3_improve.py read --base OUT_DIR [--base ...] --arm NAME=OUT_DIR[:mean] [--arm ...] --res-art RES_ART
        [--own NAME ...]

RAW_DIR = the vision_slice raw folder (monthly csv), RAW_ART = the vision_slice artifact, RES_ART = its
market_residual artifact (default label), RG_ART = the same with ``--label cs_rank_gauss``. ``read`` joins every
arm's TEST (and VAL) rows to the base arm B's residual return (``y_ret`` of B's tables) by (coin, end_time); the
suffix ``:mean`` scores an arm by the lognormal mean (1 + mu) exp(s^2 / 2) - 1, s = sigma / (1 + mu) (H1); ``--own``
names arms whose own ``y_ret`` is their own residual (H4) and adds their IC / top decile on it.
A study script (rule G6): it trains nothing.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data import market_residual as mr
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data.vision_slice import load_frames
from nig_timenet_torch.stats.metrics import block_bootstrap_ci, group_rank_ics

STEP = pd.Timedelta("4h").value
FIRST_TEST, SPAN, FOLDS, EMBARGO = pd.Timestamp("2024-01-01"), pd.Timedelta("180D"), 4, pd.Timedelta("2D")
MIN_COINS = 5
TS = LAST_COLUMN_INDEX["timestamp"]


# ---------------------------------------------------------------------------------------------------------- shared
def _art(path):
    d = npy_dir.load(path, mmap=False)
    lc = np.asarray(d["last_candles"])
    coin = np.empty(len(lc), dtype=object)
    for b in d["asset_bounds"]:
        coin[int(b["start"]):int(b["end"])] = b["name"]
    return d, lc[:, TS].astype(np.int64), coin


def _grid(raw):
    frames = load_frames(Path(raw), _coins(raw), "4h")
    closes = {k: f.set_index(pd.to_datetime(f["open_time"], utc=True))["close"] for k, f in frames.items()}
    vols = {k: f.set_index(pd.to_datetime(f["open_time"], utc=True))["volume"] for k, f in frames.items()}
    return closes, vols


def _coins(raw):
    return sorted(p.name for p in Path(raw).iterdir() if p.is_dir())


def _on_grid(s: pd.Series, first: int, size: int) -> np.ndarray:
    ns = pd.DatetimeIndex(s.index).as_unit("ns").asi8
    out = np.full(size, np.nan)
    on = (ns - first) % STEP == 0
    out[(ns[on] - first) // STEP] = s.to_numpy(np.float64)[on]
    return out


def ic_terms(score, y, t, baseline=None):
    """(timestamps, per-timestamp rank IC) over timestamps with >= MIN_COINS rows (stats.metrics.group_rank_ics);
    with ``baseline`` the partial rank IC net of that score."""
    t = np.asarray(t)
    n = pd.Series(t).map(pd.Series(t).value_counts()).to_numpy()
    k = n >= MIN_COINS
    b = None if baseline is None else np.asarray(baseline, float)[k]
    return group_rank_ics(np.asarray(y, float)[k], np.asarray(score, float)[k], t[k], baseline=b)


def ic_mean(score, y, t) -> float:
    return float(np.mean(ic_terms(score, y, t)[1]))


def _ridge(xtr, ytr, xte, alpha=1.0):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (ytr - ytr.mean()))
    return b @ w


def _feats_res(x):
    """The residual feature set of 2026-11-19-arch-indicators/m3_local.py (copied: a study never imports a study)."""
    c = x[..., 2]
    return np.column_stack([c[:, -1], c[:, -6:].sum(1), c[:, -18:].sum(1), c.sum(1), x[:, -1, 0] - x[:, -1, 1],
                            x[:, -1, 3], x[:, -1, 4], x[:, -1, 6], x[:, -1, 7], x[:, -6:, 7].sum(1), x[:, -1, 9]])


# ------------------------------------------------------------------------------------------------------------ leak
def leak(raw: str, raw_art: str, res_art: str) -> None:
    """(1) Perturb every coin's closes AFTER a cut bar K: the market, beta and channels of bars <= K must not move.
    (2) Recompute the residual label and the last bar's residual close of 3000 random rows from the raw candles.
    (3) The coin's own weight in its market: beta with the market of the OTHER coins."""
    closes, vols = _grid(raw)
    first, mkt, moves = mr.market_panel(closes, STEP)
    size = len(mkt)
    rng = np.random.default_rng(0)
    coins = sorted(closes)
    worst = 0.0
    for k in rng.integers(size // 4, size - 50, 4):
        cut = first + int(k) * STEP
        pert = {c: s * np.where(pd.DatetimeIndex(s.index).as_unit("ns").asi8 > cut,
                                np.exp(0.05 * rng.standard_normal(len(s))), 1.0) for c, s in closes.items()}
        f2, mkt2, moves2 = mr.market_panel(pert, STEP)
        assert f2 == first
        same = np.allclose(mkt2[:k + 1], mkt[:k + 1], equal_nan=True)
        moved_after = np.nanmax(np.abs(mkt2[k + 1:] - mkt[k + 1:]))
        for c in coins[::6]:
            v = _on_grid(vols[c], first, size)
            ch1, b1, _ = mr._coin_series(moves[c], mkt, v)
            ch2, b2, _ = mr._coin_series(moves2[c], mkt2, v)
            d = np.nanmax(np.abs(np.nan_to_num(ch1[:k + 1]) - np.nan_to_num(ch2[:k + 1])))
            worst = max(worst, float(d), float(np.nanmax(np.abs(b1[:k + 1] - b2[:k + 1]))))
        print(f"cut bar {k}: market <= K unchanged {same}; market after K moved by up to {moved_after:.4f}")
    print(f"max change of channels / beta at bars <= K over the cuts and 10 coins: {worst:.3g}")

    dr, ts, coin = _art(raw_art)
    dm, ts_m, _ = _art(res_art)
    assert (ts == ts_m).all()
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mkt))])
    xr, xm = np.asarray(dr["X_4h"]), np.asarray(dm["X_4h"])
    yo, ym = np.asarray(dr["y_close_reg"], float), np.asarray(dm["y_close_reg"], float)
    ok = np.abs(xm).sum(axis=(1, 2)) > 0
    pick = rng.choice(np.flatnonzero(ok), 3000, replace=False)
    betas = {c: mr.rolling_beta(moves[c], mkt) for c in coins}
    err_lab, err_x, err_b = [], [], []
    for i in pick:
        p = (ts[i] - first) // STEP
        b = betas[coin[i]][p]
        b = 1.0 if not np.isfinite(b) else b
        err_lab.append(abs(ym[i] - np.clip(yo[i] - b * (cum[p + 3] - cum[p + 1]), -1, 1)))
        err_x.append(abs(xm[i, -1, 2] - (xr[i, -1, 2] - b * mkt[p])))
        err_b.append(abs(xm[i, -1, 9] - b))
    print(f"3000 rows: max |label - recomputed| {max(err_lab):.2e}; max |res_close(last bar) - recomputed| "
          f"{max(err_x):.2e}; max |beta channel - beta(bars <= last bar)| {max(err_b):.2e}")
    lab_on = np.isfinite(yo)
    print(f"rows {len(ts)}, rows with channels {int(ok.sum())}, rows WITHOUT channels {int((~ok).sum())} "
          f"(zero inputs; they keep a residual label and enter training / test)")

    diffs = []
    tot = np.zeros(size)
    cnt = np.zeros(size)
    for c in coins:
        m = moves[c]
        g = np.isfinite(m)
        tot[g] += np.clip(m[g], -mr.MOVE_CLIP, mr.MOVE_CLIP)
        cnt[g] += 1
    for c in coins:
        m = moves[c]
        g = np.isfinite(m)
        t2, c2 = tot.copy(), cnt.copy()
        t2[g] -= np.clip(m[g], -mr.MOVE_CLIP, mr.MOVE_CLIP)
        c2[g] -= 1
        with np.errstate(invalid="ignore"):
            ex = np.where(c2 >= mr.MARKET_MIN, t2 / np.maximum(c2, 1), np.nan)
        b_in, b_ex = betas[c], mr.rolling_beta(m, ex)
        k = np.isfinite(b_in) & np.isfinite(b_ex)
        diffs.append(np.median(b_in[k] - b_ex[k]))
    print(f"beta with the coin in its market minus without, median over bars, per coin: median {np.median(diffs):+.4f}"
          f", range {min(diffs):+.4f} .. {max(diffs):+.4f}; label rows {int(lab_on.sum())}")


# -------------------------------------------------------------------------------------------------------- variants
def _beta_window(r, m, bars, low):
    df = pd.DataFrame({"r": r, "m": m})
    df.loc[~(np.isfinite(df["r"]) & np.isfinite(df["m"])), :] = np.nan
    roll = df.rolling(bars, min_periods=low)
    b = (roll.cov()["r"].xs("m", level=1) / roll.var()["m"]).to_numpy()
    return np.clip(b, *mr.BETA_CLIP)


def _beta_down(r, m):
    """Downside beta: cov / var over the last BETA_BARS bars on the bars whose market move is < 0 (past only)."""
    rr, mm = r.copy(), m.copy()
    up = ~(m < 0)
    rr[up], mm[up] = np.nan, np.nan
    return _beta_window(rr, mm, mr.BETA_BARS, mr.BETA_MIN // 2)


def _beta_of(name, r, m):
    if name == "beta60":
        return _beta_window(r, m, 60, 30)
    if name == "down":
        return _beta_down(r, m)
    return mr.rolling_beta(r, m)                     # "base" and "ex_coin" (whose m is the market without the coin)


def _variant_rows(name, closes, moves, mkt, first, ts, coin, x_raw, y_own):
    """(label, features) of every artifact row under the beta / market variant ``name``."""
    size = len(mkt)
    tot, cnt = np.zeros(size), np.zeros(size)
    for m in moves.values():
        g = np.isfinite(m)
        tot[g] += np.clip(m[g], -mr.MOVE_CLIP, mr.MOVE_CLIP)
        cnt[g] += 1
    lab = np.full(len(ts), np.nan)
    feat = np.full((len(ts), 9), np.nan)
    for c in np.unique(coin):
        r = moves[c]
        mk = mkt
        if name == "ex_coin":
            g = np.isfinite(r)
            t2, c2 = tot.copy(), cnt.copy()
            t2[g] -= np.clip(r[g], -mr.MOVE_CLIP, mr.MOVE_CLIP)
            c2[g] -= 1
            with np.errstate(invalid="ignore"):
                mk = np.where(c2 >= mr.MARKET_MIN, t2 / np.maximum(c2, 1), np.nan)
        b = _beta_of(name, r, mk)
        b = np.where(np.isfinite(b), b, 1.0)
        cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mk))])
        rows = np.flatnonzero(coin == c)
        p = (ts[rows] - first) // STEP
        lab[rows] = np.clip(y_own[rows] - b[p] * (cum[p + 3] - cum[p + 1]), -1, 1)
        win = p[:, None] + np.arange(-31, 1)[None, :]
        mw, bw = np.nan_to_num(mk[win]), b[win]
        rc = x_raw[rows, :, 2] - bw * mw
        rng_last = x_raw[rows, -1, 0] - x_raw[rows, -1, 1]
        feat[rows] = np.column_stack([rc[:, -1], rc[:, -6:].sum(1), rc[:, -18:].sum(1), rc.sum(1), rng_last,
                                      mw[:, -1], mw[:, -6:].sum(1), b[p], rc.std(1)])
    return lab, feat, tot / np.maximum(cnt, 1)


def variants(raw: str, raw_art: str, res_art: str, rg_art: str) -> None:
    closes, _ = _grid(raw)
    first, mkt, moves = mr.market_panel(closes, STEP)
    dr, ts, coin = _art(raw_art)
    dm, _, _ = _art(res_art)
    dg, _, _ = _art(rg_art)
    t = pd.to_datetime(ts)
    xr, xm = np.asarray(dr["X_4h"], float), np.asarray(dm["X_4h"], float)
    y_own, y_b, y_rg = (np.asarray(d["y_close_reg"], float) for d in (dr, dm, dg))
    ok = np.abs(xm).sum(axis=(1, 2)) > 0
    folds = [(FIRST_TEST + k * SPAN, FIRST_TEST + (k + 1) * SPAN) for k in range(FOLDS)]
    test_all = ok & (t >= folds[0][0]) & (t < folds[-1][1])

    # H3: the rank-gauss label as a ridge target (art_mres features), scored against B's label
    fm = _feats_res(xm)
    rows = []
    for k, (lo, hi) in enumerate(folds):
        tr, te = ok & (t < lo - EMBARGO), ok & (t >= lo) & (t < hi)
        r = {"fold": k}
        for nm, yfit in (("ridge on B label", y_b), ("ridge on rank-gauss label", y_rg)):
            r[nm] = ic_mean(_ridge(fm[tr], yfit[tr], fm[te]), y_b[te], ts[te])
        rows.append(r)
    h3 = pd.DataFrame(rows)
    print("H3 model-free bound (IC by timestamp against B's label, art_mres feature set of m3_local.py)")
    print(h3.to_string(index=False, float_format="{:+.4f}".format))
    print(h3.drop(columns="fold").mean().to_string(float_format="{:+.4f}".format))

    # H4: beta / market variants
    span_mkt = None
    out = []
    labs = {}
    for name in ("base", "beta60", "down", "ex_coin"):
        lab, feat, _ = _variant_rows(name, closes, moves, mkt, first, ts, coin, xr, y_own)
        labs[name] = lab
        if name == "base":
            dev = np.nanmax(np.abs(lab[ok] - y_b[ok]))
            print(f"variant 'base' reproduces the artifact's residual label: max |diff| {dev:.2e}")
            cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mkt))])
            p = (ts - first) // STEP
            span_mkt = cum[p + 3] - cum[p + 1]
        df = pd.DataFrame({"t": ts[test_all], "lab": lab[test_all], "m": span_mkt[test_all]})
        g = df.groupby("t").agg(lab=("lab", "mean"), m=("m", "first"), n=("lab", "size"))
        g = g[g["n"] >= MIN_COINS]
        lo5, hi5 = g["m"].quantile(0.05), g["m"].quantile(0.95)
        rec = {"variant": name, "corr(mean residual, market span)": g["lab"].corr(g["m"]),
               "mean residual, bottom 5 % market spans": g.loc[g["m"] <= lo5, "lab"].mean(),
               "mean residual, top 5 % market spans": g.loc[g["m"] >= hi5, "lab"].mean(),
               "label var": float(np.nanvar(lab[test_all]))}
        ics_own, ics_b = [], []
        good = ok & np.isfinite(feat).all(axis=1) & np.isfinite(lab)
        for lo, hi in folds:
            tr, te = good & (t < lo - EMBARGO), good & (t >= lo) & (t < hi)
            pr = _ridge(feat[tr], lab[tr], feat[te])
            ics_own.append(ic_mean(pr, lab[te], ts[te]))
            ics_b.append(ic_mean(pr, y_b[te], ts[te]))
        rec["ridge IC vs own label"] = float(np.mean(ics_own))
        rec["ridge IC vs B label"] = float(np.mean(ics_b))
        rec["ridge IC own, per fold"] = " ".join(f"{v:+.3f}" for v in ics_own)
        out.append(rec)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("H4 model-free (TEST windows 2024-01 .. 2025-12; ridge on 9 residual features of each variant)")
        print(pd.DataFrame(out).to_string(index=False, float_format="{:+.5f}".format))
    base = out[0]["mean residual, bottom 5 % market spans"]
    for rec in out[1:]:
        shrink = 1 - abs(rec["mean residual, bottom 5 % market spans"]) / abs(base)
        gate = shrink >= 0.25 and rec["ridge IC vs own label"] >= out[0]["ridge IC vs B label"] - 0.005
        print(f"gate {rec['variant']}: bottom-5 % contamination shrinks by {shrink:+.0%}; passes: {gate}")


# ------------------------------------------------------------------------------------------------------------ read
def _folds(path: Path):
    for d in sorted(path.glob("fold_*_seed_*")):
        m = re.match(r"fold_(\d+)_seed_(\d+)", d.name)
        yield int(m.group(1)), int(m.group(2)), pd.read_parquet(d / "battery_table_val.parquet"), \
            pd.read_parquet(d / "battery_table_test.parquet")


def _mean_score(df):
    s = df["sigma"] / (1 + df["mu"])
    return (1 + df["mu"]) * np.exp(0.5 * s ** 2) - 1


def _trivial(res_art: str):
    """{(coin, end_time): (reversal, low volatility, beta)} from the residual artifact's windows."""
    d, ts, coin = _art(res_art)
    x = np.asarray(d["X_4h"], float)
    end = pd.to_datetime(ts) + pd.Timedelta("4h")
    f = pd.DataFrame({"coin": coin, "end_time": end.to_numpy(), "reversal": -x[:, -1, 2],
                      "low_vol": -x[:, :, 2].std(axis=1), "beta": x[:, -1, 9]})
    return f.set_index(["coin", "end_time"])


def _book(st, val, col, yref):
    """(top-decile trades, gross per trade, day mean, lo, hi) with each fold's VAL 90th percentile of ``col``."""
    q = val.groupby("fold")[col].quantile(0.9)
    lg = st[st[col] >= st["fold"].map(q)]
    days = lg.groupby(pd.to_datetime(lg["end_time"]).dt.floor("D"))[yref].mean()
    m, lo, hi = block_bootstrap_ci(days.to_numpy(), np.arange(len(days)), level=0.95, n_boot=2000)
    return len(lg), float(lg[yref].mean()), m, lo, hi


def _deciles(st, col, yref):
    st = st[st.groupby("end_time")[col].transform("size") >= 10]
    dec = st.groupby("end_time")[col].rank(pct=True, method="first").mul(10).apply(np.ceil).clip(1, 10)
    ex = st[yref] - st.groupby("end_time")[yref].transform("mean")
    return ex.groupby(dec).mean()


def _score_rows(name, st, val, col, yref, base_terms=None):
    keys, vals = ic_terms(st[col], st[yref], st["end_time"])
    n, gross, m, lo, hi = _book(st, val, col, yref)
    dec = _deciles(st, col, yref)
    fold_ic = [ic_mean(g[col], g[yref], g["end_time"]) for _, g in st.groupby("fold")]
    p_ic = float(np.nanmean(ic_terms(st[col], st[yref], st["end_time"], st["low_vol"])[1]))
    r = {"score": name, "IC": float(np.mean(vals)), "IC per fold": " ".join(f"{v:+.3f}" for v in fold_ic),
         "IC net of low_vol": p_ic,
         "top10 n": n, "top10 gross %": 100 * gross, "top10 day %": 100 * m, "CI lo": 100 * lo, "CI hi": 100 * hi,
         **{f"d{k} %": 100 * float(dec.get(k, np.nan)) for k in (1, 2, 5, 9, 10)}}
    if base_terms is not None:
        a = pd.Series(vals, index=pd.to_datetime(keys))
        b = base_terms.reindex(a.index)
        diff = (a - b).dropna()
        day = diff.groupby(diff.index.floor("D")).mean()
        dm, dlo, dhi = block_bootstrap_ci(day.to_numpy(), np.arange(len(day)), level=0.95, n_boot=2000)
        r.update({"dIC vs B": dm, "dIC lo": dlo, "dIC hi": dhi})
    return r, pd.Series(vals, index=pd.to_datetime(keys))


def read(base: str, arms: list[str], res_art: str, own: list[str]) -> None:
    ref = {}
    for f, s, va, te in (x for b in base for x in _folds(Path(b))):
        ref[(f, s)] = (va.set_index(["coin", "end_time"])["y_ret"], te.set_index(["coin", "end_time"])["y_ret"])
    triv = _trivial(res_art)
    specs = [("B", b, False) for b in base]
    for a in arms:
        name, path = a.split("=", 1)
        mean = path.endswith(":mean")
        specs.append((name, path[:-len(":mean")] if mean else path, mean))
    rows, bias, base_terms = [], [], {}
    names = list(dict.fromkeys(n for n, _, _ in specs))
    for name in names:
        per_seed: dict[int, list] = {}
        for f, s, va, te, mean in ((*x, m) for n, p, m in specs if n == name for x in _folds(Path(p))):
            yv, yt = ref[(f, s)]
            parts = []
            for df, yr in ((va, yv), (te, yt)):
                df = df.join(yr.rename("y_res"), on=["coin", "end_time"]).dropna(subset=["y_res"])
                df["score"] = _mean_score(df) if mean else df["mu"]
                df["fold"] = f
                parts.append(df.join(triv, on=["coin", "end_time"]))
            per_seed.setdefault(s, []).append(tuple(parts))
        for s, parts in sorted(per_seed.items()):
            val = pd.concat([p[0] for p in parts])
            st = pd.concat([p[1] for p in parts])
            r, terms = _score_rows(name, st, val, "score", "y_res", base_terms.get(s))
            if name == "B":
                base_terms[s] = terms
            rows.append({"seed": s, "rows": len(st), **r})
            if name in own:
                r2, _ = _score_rows(f"{name} on own residual", st, val, "score", "y_ret")
                rows.append({"seed": s, "rows": len(st), **r2})
            if name == "B":
                rng = np.random.default_rng(s)
                st["shuffled"] = st.groupby("end_time")["score"].transform(lambda v, g=rng: g.permutation(v.to_numpy()))
                val["shuffled"] = rng.permutation(val["score"].to_numpy())
                for c in ("shuffled", "reversal", "low_vol", "beta"):
                    rows.append({"seed": s, "rows": len(st), **_score_rows(f"trivial: {c}", st, val, c, "y_res")[0]})
                for k, g in st.groupby("fold"):
                    bias.append({"seed": s, "fold": k, "share mu>0": float((g["mu"] > 0).mean()),
                                 "share y_res>0": float((g["y_res"] > 0).mean()), "mean mu %": 100 * g["mu"].mean(),
                                 "mean y_res %": 100 * g["y_res"].mean()})
            bias.append({"seed": s, "fold": f"all ({name})", "share mu>0": float((st["mu"] > 0).mean()),
                         "share y_res>0": float((st["y_res"] > 0).mean()), "mean mu %": 100 * st["mu"].mean(),
                         "mean y_res %": 100 * st["y_res"].mean()})
    with pd.option_context("display.width", 400, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(bias).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------- detail
def _load_arm(paths, ref, triv):
    """{seed: (val, test)} of one arm joined to B's residual return and the trivial scores."""
    per_seed: dict[int, list] = {}
    for f, s, va, te in (x for p in paths for x in _folds(Path(p))):
        yv, yt = ref[(f, s)]
        parts = []
        for df, yr in ((va, yv), (te, yt)):
            df = df.join(yr.rename("y_res"), on=["coin", "end_time"]).dropna(subset=["y_res"])
            df["fold"] = f
            parts.append(df.join(triv, on=["coin", "end_time"]))
        per_seed.setdefault(s, []).append(parts)
    return {s: (pd.concat(p[0] for p in v), pd.concat(p[1] for p in v)) for s, v in sorted(per_seed.items())}


def detail(base: list[str], arms: list[str], raw: str, res_art: str) -> None:
    """The C7 reading of each arm (score = mu): IC and top-decile residual gross per group of rows, the worst
    top-decile rows, and three VAL-fitted selection rules (information: residual gross per trade, no cost)."""
    ref = {}
    for f, s, va, te in (x for b in base for x in _folds(Path(b))):
        ref[(f, s)] = (va.set_index(["coin", "end_time"])["y_ret"], te.set_index(["coin", "end_time"])["y_ret"])
    triv = _trivial(res_art)
    closes, _ = _grid(raw)
    first, mkt, _ = mr.market_panel(closes, STEP)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mkt))])
    specs: dict[str, list] = {"B": list(base)}
    for a in arms:
        n, p = a.split("=", 1)
        specs.setdefault(n, []).append(p)
    for name, paths in specs.items():
        for s, (val, st) in _load_arm(paths, ref, triv).items():
            et = pd.to_datetime(st["end_time"]).dt.tz_localize("UTC").astype("int64").to_numpy()
            p = (et - pd.Timedelta("4h").value - first) // STEP
            st["mkt_span"] = cum[p + 3] - cum[p + 1]
            q = val.groupby("fold")["mu"].quantile(0.9)
            st["long"] = st["mu"] >= st["fold"].map(q)
            groups = {
                "market span up / down": np.where(st["mkt_span"] > 0, "up", "down"),
                "market span |move| tercile": pd.qcut(st["mkt_span"].abs(), 3, labels=["calm", "mid", "wild"]),
                "coin volatility tercile (window sd)": pd.qcut(-st["low_vol"], 3, labels=["calm", "mid", "wild"]),
                "sigma tercile": pd.qcut(st["sigma"], 3, labels=["narrow", "mid", "wide"]),
                "year": pd.to_datetime(st["end_time"]).dt.year.to_numpy(),
            }
            out = []
            for gname, key in groups.items():
                for k, g in st.groupby(np.asarray(key)):
                    lg = g[g["long"]]
                    out.append({"group": gname, "value": k, "rows": len(g), "IC": ic_mean(g["mu"], g["y_res"],
                                g["end_time"]), "IC low_vol (trivial)": ic_mean(g["low_vol"], g["y_res"],
                                g["end_time"]), "top10 n": len(lg), "top10 gross %": 100 * lg["y_res"].mean(),
                                "group mean y_res %": 100 * g["y_res"].mean(),
                                "top10 hit": float((lg["y_res"] > 0).mean())})
            with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
                print(f"=== {name} seed {s}: error map")
                print(pd.DataFrame(out).to_string(index=False))
                lg = st[st["long"]].nsmallest(12, "y_res")
                print(f"=== {name} seed {s}: worst top-decile rows")
                print(lg[["coin", "end_time", "mu", "sigma", "y_res", "mkt_span", "beta", "low_vol"]].to_string(
                    index=False))
                rules = {"top 5 % mu": ("mu", 0.95, None), "top 10 % mu, sigma < VAL median": ("mu", 0.9, "lo_sig"),
                         "top 10 % mu / sigma": ("z", 0.9, None)}
                st["z"], val["z"] = st["mu"] / st["sigma"], val["mu"] / val["sigma"]
                sel = []
                for rn, (c, qq, extra) in rules.items():
                    thr = st["fold"].map(val.groupby("fold")[c].quantile(qq))
                    keep = st[c] >= thr
                    if extra:
                        keep &= st["sigma"] < st["fold"].map(val.groupby("fold")["sigma"].median())
                    k = st[keep]
                    sel.append({"rule (VAL-fitted)": rn, "share kept": float(keep.mean()),
                                "gross per trade %": 100 * k["y_res"].mean(),
                                "minus timestamp mean %": 100 * (k["y_res"] - st.groupby("end_time")["y_res"]
                                                                  .transform("mean")[keep]).mean()})
                print(f"=== {name} seed {s}: selection rules (3 tried)")
                print(pd.DataFrame(sel).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("leak")
    for k in ("raw", "raw_art", "res_art"):
        lk.add_argument(k)
    v = sub.add_parser("variants")
    for k in ("raw", "raw_art", "res_art", "rg_art"):
        v.add_argument(k)
    q = sub.add_parser("read")
    q.add_argument("--base", action="append", required=True)
    q.add_argument("--arm", action="append", default=[])
    q.add_argument("--res-art", required=True)
    q.add_argument("--own", action="append", default=[])
    d = sub.add_parser("detail")
    d.add_argument("--base", action="append", required=True)
    d.add_argument("--arm", action="append", default=[])
    d.add_argument("--raw", required=True)
    d.add_argument("--res-art", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "detail":
        detail(a.base, a.arm, a.raw, a.res_art)
    elif a.cmd == "leak":
        leak(a.raw, a.raw_art, a.res_art)
    elif a.cmd == "variants":
        variants(a.raw, a.raw_art, a.res_art, a.rg_art)
    else:
        read(a.base, a.arm, a.res_art, a.own)
    return 0


if __name__ == "__main__":
    sys.exit(main())
