"""Price-unit short rules R0 / H1a / H1b / H3 / H13 on saved battery tables (spec: PREREGISTERED_H1_H3.md).

Usage: python docs/audit/2026-11-19-short-side/h1_h3_rules.py <folder> <seeds e.g. 0,1>   (or add --selftest)
Per fold every threshold is fitted on that fold's VAL table only and applied to its TEST table; the TEST trades of
the 7 folds are pooled per seed. Rules change the SHORT side only. Loading: read_short_side.load.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_short_side import load  # noqa: E402

from nig_timenet_torch.eval.battery_lib import round_trip  # noqa: E402
from nig_timenet_torch.eval.battery_scorecard import day_profile  # noqa: E402

warnings.filterwarnings("ignore")
RT = round_trip(0.0008)
LEVEL = 1.0 - 0.05 / 5
RULES = ["R0", "H1a", "H1b", "H3", "H13"]
QS = (0.5, 0.6, 0.7, 0.8, 0.9)


def prep(df):
    """Side, confidence c, log-space sd s and the scored-row mask of a table."""
    mu, sig = df["mu"].to_numpy(), df["sigma"].to_numpy()
    side = np.where(df["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
    return side, np.abs(df["p_dir"].to_numpy() - 0.5), np.log(1 + sig / (1 + mu)), (df["dir_true"] != 0).to_numpy()


def fold_masks(val, test):
    """Per-rule boolean keep masks over TEST (only shorts matter), the universe mask of TEST and the chosen q."""
    sv, cv, _, scv = prep(val)
    st, ct, s_t, sct = prep(test)
    thr = np.quantile(np.abs(val["mu"].to_numpy()), 0.8)
    u_v = scv & (np.abs(val["mu"].to_numpy()) >= thr)
    u_t = sct & (np.abs(test["mu"].to_numpy()) >= thr)
    vs = u_v & (sv < 0)
    net_v = -val["y_ret"].to_numpy() - RT
    best_q, best_net, t_short = None, -np.inf, None
    for q in QS:
        t = np.quantile(cv[vs], q)
        sel = vs & (cv >= t)
        net = net_v[sel].mean() if sel.any() else -np.inf
        if net > best_net:
            best_q, best_net, t_short = q, net, t
    h1a = -np.log1p(test["mu"].to_numpy()) > s_t**2 / 2 + RT
    h1b = ct >= t_short
    h3 = (test["sigma"].to_numpy() < np.quantile(val["sigma"].to_numpy(), 0.8)) & (ct >= np.median(cv[vs]))
    return {"R0": np.ones(len(test), bool), "H1a": h1a, "H1b": h1b, "H3": h3, "H13": h1b & h3}, u_t, best_q


def pool(folds):
    rows, keeps, qs = [], {r: [] for r in RULES}, []
    for k, val, test in folds:
        m, u, q = fold_masks(val, test)
        side, *_ = prep(test)
        qs.append((k, q))
        rows.append(pd.DataFrame({"side": side, "y": test["y_ret"].to_numpy(), "u": u,
                                  "day": pd.to_datetime(test["end_time"]).to_numpy().astype("datetime64[D]")}))
        for r in RULES:
            keeps[r].append(m[r])
    return pd.concat(rows, ignore_index=True), {r: np.concatenate(keeps[r]) for r in RULES}, qs


def stats(pnl, days):
    return {"n": len(pnl), "win": (pnl > 0).mean(), "net": pnl.mean(), **day_profile(pnl, days, LEVEL)}


def report(folds, seed):
    d, keeps, qs = pool(folds)
    pnl = d["side"].to_numpy() * d["y"].to_numpy() - RT
    short, longs = d["u"].to_numpy() & (d["side"].to_numpy() < 0), d["u"].to_numpy() & (d["side"].to_numpy() > 0)
    days = d["day"].to_numpy()
    print(f"\n######## seed {seed}: {len(d):,} TEST rows; U shorts {short.sum():,}, U longs {longs.sum():,}")
    print("H1b q per fold: " + ", ".join(f"f{k}={q}" for k, q in qs))
    out = {}
    print(f"  {'rule':<5}| kept shorts: {'n':>6}{'share':>7}{'win%':>6}{'net%':>8}{'mean d% [CI]':>26}{'busy d%':>9}"
          f" | dropped: {'n':>6}{'net%':>8} | book: {'net%':>8}{'mean d% [CI]':>26}{'win d%':>7}{'w/o best%':>10}"
          f"{'busy d%':>9}")
    for r in RULES:
        ks, dr = short & keeps[r], short & ~keeps[r]
        kp = stats(pnl[ks], days[ks]) if ks.any() else None
        bk = stats(pnl[longs | ks], days[longs | ks])
        dn = pnl[dr].mean() if dr.any() else float("nan")
        out[r] = {"k": kp, "drop": dn, "book": bk}
        kt = (f"{kp['n']:>6,}{kp['n'] / short.sum():>7.1%}{kp['win'] * 100:>6.1f}{kp['net'] * 100:>+8.3f}"
              f"{kp['mean_day'] * 100:>+10.3f} [{kp['ci_lo'] * 100:+.3f}, {kp['ci_hi'] * 100:+.3f}]"
              f"{kp['mean_busy'] * 100:>+9.3f}") if kp else f"{0:>6}" + " " * 56
        print(f"  {r:<5}| {kt} | {dr.sum():>6,}{dn * 100:>+8.3f} | {bk['net'] * 100:>+8.3f}"
              f"{bk['mean_day'] * 100:>+10.3f} [{bk['ci_lo'] * 100:+.3f}, {bk['ci_hi'] * 100:+.3f}]"
              f"{bk['win_days'] * 100:>7.1f}{bk['mean_wo_best'] * 100:>+10.3f}{bk['mean_busy'] * 100:>+9.3f}")
    return out


def accept(res_by_seed):
    print("\n== accept rule (each condition must hold on EVERY seed)")
    for r in RULES[1:]:
        c = [[], [], []]
        for res in res_by_seed.values():
            o, b = res[r], res["R0"]
            k = o["k"]
            c[0].append(bool(k and k["net"] > 0 and k["net"] > b["k"]["net"]))
            c[1].append(bool(k and o["drop"] < k["net"]))
            c[2].append(bool(o["book"]["net"] > b["book"]["net"] and o["book"]["mean_busy"] > b["book"]["mean_busy"]))
        s = list(res_by_seed)
        z = [dict(zip(s, ci, strict=True)) for ci in c]
        print(f"  {r:<4} (1) kept net>0 and >R0 short: {z[0]}; (2) dropped<kept: {z[1]}; "
              f"(3) book net and busy > R0: {z[2]}; accepted: {all(map(all, c))}")


def selftest():
    rng = np.random.default_rng(0)

    def frame(n=4000):
        short = rng.random(n) < 0.5
        sig = rng.permutation(np.linspace(0, 1, n))  # same sigma quantiles in VAL and TEST
        top = sig >= np.quantile(sig, 0.8)
        y = np.where(short, np.where(top, 0.05, -0.005), 0.01)  # short loses 5 % in the top sigma fifth, gains 0.5 %
        return pd.DataFrame({"mu": np.where(short, -0.05, 0.05), "sigma": sig, "p_dir": np.where(short, 0.1, 0.9),
                             "y_ret": y, "dir_true": 1,
                             "end_time": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 60, n), "D")})
    folds = [(k, frame(), frame()) for k in range(7)]
    d, keeps, _ = pool(folds)
    short = d["u"].to_numpy() & (d["side"].to_numpy() < 0)
    top = np.concatenate([t["sigma"].to_numpy() >= np.quantile(t["sigma"].to_numpy(), 0.8) for _, _, t in folds])
    assert not (keeps["H3"] & short & top).any(), "H3 kept a top-fifth-sigma short"
    assert (keeps["R0"] & short).sum() == short.sum() > 0, "R0 dropped a short"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    folder, seeds = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")]
    accept({s: report(load(folder, s), s) for s in seeds})
