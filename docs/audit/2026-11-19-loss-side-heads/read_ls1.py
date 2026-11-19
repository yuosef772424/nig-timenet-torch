"""LS1 walk-forward reading (spec: PREREGISTERED.md section 2 and amendment 1): books P1 / P1L / P3 per arm and seed,
P2 (S against R), health (close rank IC against A0, q_up coverage).

Usage: python docs/audit/2026-11-19-loss-side-heads/read_ls1.py <runs root> [--seeds 2,3] [--arms a0,r,s,w]
<runs root> holds rolling-ls1-<arm>-001/out_s<seed>/fold_00k_seed_<seed>/battery_table_{val,test}.parquet.
Every threshold of fold k comes from fold k's VAL table; the TEST trades of the 7 folds are pooled per seed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import round_trip
from nig_timenet_torch.eval.battery_scorecard import day_profile
from nig_timenet_torch.stats.metrics import spearman

RT = round_trip(0.0008)
Z90 = 1.2815515655446004
LEVEL = 0.99
TOP, VETO = 0.95, 0.8


def scores(df: pd.DataFrame) -> dict:
    """Every score of a battery table (return units unless named log)."""
    mu, sig = df["mu"].to_numpy(), df["sigma"].to_numpy()
    lm = np.log1p(mu)
    s = np.log1p(sig / (1.0 + mu))
    move = np.expm1(lm + s**2 / 2)
    out = {"mu": mu, "-mu": -mu, "E_long": move - RT, "E_short": -move - RT, "nig_q90": lm + Z90 * s}
    for c in ("pay_long", "pay_short", "q_up"):
        if c in df:
            out[c] = df[c].to_numpy()
    return out


def books(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """{book name: (side, TEST mask)}; side +1 long, -1 short."""
    sv, st = scores(val), scores(test)
    ok_v, ok_t = (val["dir_true"] != 0).to_numpy(), (test["dir_true"] != 0).to_numpy()

    def top(name):
        return ok_t & (st[name] >= np.quantile(sv[name][ok_v], TOP))

    out = {"SHORT top5 -mu": (-1, top("-mu")), "SHORT top5 E_short": (-1, top("E_short")),
           "LONG top5 mu": (1, top("mu")), "LONG top5 E_long": (1, top("E_long"))}
    if "pay_short" in st:
        out["SHORT top5 pay_short"] = (-1, top("pay_short"))
        out["LONG top5 pay_long"] = (1, top("pay_long"))
    u_v = ok_v & (sv["-mu"] >= np.quantile(sv["-mu"][ok_v], TOP))
    u_t = out["SHORT top5 -mu"][1]
    for veto in ("q_up", "nig_q90"):
        if veto in st:
            thr = np.quantile(sv[veto][u_v], VETO)
            out[f"P3 kept {veto}"] = (-1, u_t & (st[veto] < thr))
            out[f"P3 dropped {veto}"] = (-1, u_t & (st[veto] >= thr))
    return out


def close_ic(test: pd.DataFrame) -> float:
    """Mean over timestamps (>= 10 rows) of the Spearman of mu with the realised return."""
    vals = [spearman(g["mu"].to_numpy(), g["y_ret"].to_numpy()) for _, g in test.groupby("end_time") if len(g) >= 10]
    return float(np.nanmean(vals))


def read_seed(folder: Path, seed: int) -> dict:
    trades, ics, cover = {}, [], []
    for k in range(7):
        d = folder / f"fold_{k:03d}_seed_{seed}"
        val, test = (pd.read_parquet(d / f"battery_table_{s}.parquet") for s in ("val", "test"))
        ics.append(close_ic(test))
        if "q_up" in test:
            cover.append(((test["y_ret"] <= test["q_up"]).to_numpy(), test["dir_true"].to_numpy() != 0))
        day = pd.to_datetime(test["end_time"]).dt.floor("D").to_numpy()
        for name, (side, m) in books(val, test).items():
            pnl = side * test["y_ret"].to_numpy()[m] - RT
            trades.setdefault(name, []).append((pnl, day[m]))
    out = {"close_ic_fold_mean": float(np.mean(ics))}
    if cover:
        out["q_up_coverage"] = float(np.concatenate([c[0][c[1]] for c in cover]).mean())
    for name, parts in trades.items():
        pnl, days = np.concatenate([p for p, _ in parts]), np.concatenate([d for _, d in parts])
        prof = day_profile(pnl, days, LEVEL) if len(pnl) else {}
        out[name] = {"n": int(len(pnl)), "net": float(pnl.mean()) if len(pnl) else float("nan"), **prof}
    return out


def verdict(res: dict, seeds) -> dict:
    """The pre-registered accept rules (PREREGISTERED.md section 2)."""
    v = {}
    for arm in ("r", "w"):
        if arm not in res:
            continue
        g = {s: res[arm][s] for s in seeds}

        def beats(s, book, others, g=g):
            b = g[s][book]
            return all(b["net"] > g[s][o]["net"] and b["mean_busy"] > g[s][o]["mean_busy"] for o in others)

        v[f"P1 {arm}"] = all(beats(s, "SHORT top5 pay_short", ("SHORT top5 -mu", "SHORT top5 E_short")) for s in seeds)
        v[f"P1L {arm}"] = all(beats(s, "LONG top5 pay_long", ("LONG top5 mu", "LONG top5 E_long")) for s in seeds)
        v[f"P3 {arm}"] = all(g[s]["P3 kept q_up"]["net"] > g[s]["P3 kept nig_q90"]["net"]
                             and g[s]["P3 dropped q_up"]["net"] < g[s]["P3 kept q_up"]["net"] for s in seeds)
        if "a0" in res:
            v[f"H close IC {arm} vs a0"] = all(abs(g[s]["close_ic_fold_mean"] - res["a0"][s]["close_ic_fold_mean"])
                                               <= 0.01 for s in seeds)
        v[f"H q_up coverage {arm}"] = all(0.85 <= g[s]["q_up_coverage"] <= 0.95 for s in seeds)
    if {"s", "r", "a0"} <= set(res):
        v["P2 s vs r"] = all(res["s"][s]["SHORT top5 pay_short"]["net"] > res["r"][s]["SHORT top5 pay_short"]["net"]
                             and res["s"][s]["close_ic_fold_mean"] >= res["a0"][s]["close_ic_fold_mean"] - 0.01
                             for s in seeds)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--seeds", default="2,3")
    ap.add_argument("--arms", default="a0,r,s,w")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    res = {}
    for arm in a.arms.split(","):
        base = a.root / f"rolling-ls1-{arm}-001"
        if all((base / f"out_s{s}" / f"fold_006_seed_{s}" / "battery_table_test.parquet").is_file() for s in seeds):
            res[arm] = {s: read_seed(base / f"out_s{s}", s) for s in seeds}
    cols = ("n", "net", "mean_day", "ci_lo", "ci_hi", "win_days", "mean_busy", "mean_wo_best")
    for arm, by_seed in res.items():
        for s, r in by_seed.items():
            extra = f"  q_up coverage {r['q_up_coverage']:.3f}" if "q_up_coverage" in r else ""
            print(f"\n== arm {arm} seed {s}: close rank IC (fold mean) {r['close_ic_fold_mean']:.4f}{extra}")
            print(f"  {'book':<26}" + "".join(f"{c:>12}" for c in cols))
            for name, b in r.items():
                if isinstance(b, dict):
                    print(f"  {name:<26}" + "".join(
                        f"{b[c]:>12d}" if c == "n" else (f"{100 * b[c]:>11.3f}%" if c != "win_days" else
                                                         f"{100 * b[c]:>11.1f}%") for c in cols if c in b))
    v = verdict(res, seeds)
    print("\nverdict:", json.dumps(v, indent=1))
    (a.root / "ls1_reading.json").write_text(json.dumps({"results": {k: {str(s): r for s, r in by.items()}
                                                                     for k, by in res.items()}, "verdict": v},
                                                        indent=1, default=float))


if __name__ == "__main__":
    main()
