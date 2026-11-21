"""LS1 known-answer test on E1 (spec: PREREGISTERED.md section 1): arms base / R / S x seeds, 3 epochs, CPU.

Usage: python docs/audit/2026-11-19-loss-side-heads/e1_known_answer.py <work dir> [--seeds 0,1,2] [--epochs 3]
Writes <work dir>/e1_known_answer.json and prints one table. E1's close label is Gaussian in log units with the
oracle mean m and sd v per row, so the oracle payoffs and quantile are closed forms of (m, v).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nig_timenet_torch.data_io.synthetic import oracle_keys, write_synthetic
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.metrics import spearman
from nig_timenet_torch.stats.nig import nig_predictive_std
from nig_timenet_torch.trainer.tasks import side_payoffs
from nig_timenet_torch.workflow import e1_arms, run

COST = 0.0008
Z90 = 1.2815515655446004
ARMS = {
    "base": {},
    "R": {"model": {"aux_heads": ["side_payoff", "upper_quantile"], "aux_trunk_grad": 0.0}},
    "S": {"model": {"aux_heads": ["side_payoff"], "aux_trunk_grad": 0.3}},
    # amendment 1 (PREREGISTERED.md): the side-payoff rows weighted by 1 / NIG predictive variance (WLS)
    "W": {"model": {"aux_heads": ["side_payoff", "upper_quantile"], "aux_trunk_grad": 0.0},
          "train": {"aux_variance_weight": 1.0}},
}
NIG = head_spec("nig_regression", "close")
SIDE = head_spec("side_payoff", "close").output_keys
UPQ = head_spec("upper_quantile", "close").output_keys["q"]


def closed_form(m: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(long, short) expected payoff after cost of a Gaussian log move (mean m, sd v)."""
    move = np.expm1(m + v**2 / 2)
    return move - 2 * COST, -move - 2 * COST


def rho(a, b) -> float:
    return float(spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def one(arm: str, seed: int, env: Path, root: Path, epochs: int) -> dict:
    t0 = time.monotonic()
    settings = e1_arms.arm_settings(ARMS[arm], root / arm / f"s{seed}", env, seed, epochs)
    r = run.build(run.prepare(settings))
    run.train(r)
    tab = r.table("test")
    idx = r.idx["test"]
    m_key, v_key = oracle_keys("close")
    m, v = np.asarray(r.dataset[m_key])[idx], np.asarray(r.dataset[v_key])[idx]
    y = np.asarray(r.dataset[NIG.label_key])[idx]
    mu = tab[NIG.output_keys["mu"]].to_numpy()
    s = nig_predictive_std(*(tab[NIG.output_keys[k]].to_numpy() for k in ("nu", "alpha", "beta")))
    o_long, o_short = closed_form(m, v)
    e_long, e_short = closed_form(mu, s)
    out = {"arm": arm, "seed": seed, "rows": len(idx), "mu_rho": rho(mu, m), "E_long_rho": rho(e_long, o_long),
           "E_short_rho": rho(e_short, o_short), "best_epoch": r.result.best_epoch}
    if SIDE["long"] in tab:
        p_long, p_short = tab[SIDE["long"]].to_numpy(), tab[SIDE["short"]].to_numpy()
        out.update(pay_long_rho=rho(p_long, o_long), pay_short_rho=rho(p_short, o_short),
                   pay_long_bias_sd=float((p_long.mean() - o_long.mean()) / o_long.std()),
                   pay_short_bias_sd=float((p_short.mean() - o_short.mean()) / o_short.std()))
        r_long, r_short = (t.numpy() for t in side_payoffs(torch.as_tensor(y), COST))
        out.update(realised_long=float(r_long.mean()), realised_short=float(r_short.mean()))
    if UPQ in tab:
        q = tab[UPQ].to_numpy()
        out.update(q_coverage=float((y <= q).mean()), q_rho=rho(q, m + Z90 * v),
                   q_bias_sd=float(((q - (m + Z90 * v)).mean()) / v.mean()))
    out["seconds"] = round(time.monotonic() - t0)
    return out


def verdict(res: list[dict]) -> dict:
    base_mu = [r["mu_rho"] for r in res if r["arm"] == "base"]
    lo, hi = min(base_mu) - 0.02, max(base_mu) + 0.02
    checks = {}
    for r in res:
        k = f"{r['arm']}_s{r['seed']}"
        if "pay_short_rho" in r:
            checks[f"E1-a {k}"] = r["pay_short_rho"] >= r["E_short_rho"] - 0.05
            checks[f"E1-b {k}"] = r["pay_long_rho"] >= r["E_long_rho"] - 0.05
            checks[f"E1-c {k}"] = abs(r["pay_long_bias_sd"]) <= 0.3 and abs(r["pay_short_bias_sd"]) <= 0.3
        if "q_coverage" in r:
            checks[f"E1-d {k}"] = 0.87 <= r["q_coverage"] <= 0.93 and r["q_rho"] >= 0.8
        if r["arm"] in ("R", "W"):
            checks[f"E1-e {k}"] = lo <= r["mu_rho"] <= hi
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args(argv)
    env = a.work / "e1"
    if not (env / "manifest.json").exists():
        write_synthetic(env, overwrite=True)
    res = [one(arm, int(s), env, a.work / "runs", a.epochs) for s in a.seeds.split(",") for arm in ARMS]
    checks = verdict(res)
    (a.work / "e1_known_answer.json").write_text(json.dumps({"runs": res, "checks": checks}, indent=1))
    keys = ["mu_rho", "E_short_rho", "pay_short_rho", "E_long_rho", "pay_long_rho", "pay_short_bias_sd",
            "pay_long_bias_sd", "q_coverage", "q_rho", "best_epoch", "seconds"]
    print("arm seed " + " ".join(keys))
    for r in res:
        print(r["arm"], r["seed"], " ".join("-" if k not in r else (f"{r[k]:.3f}" if isinstance(r[k], float)
                                                                     else str(r[k])) for k in keys))
    print("checks:", sum(checks.values()), "/", len(checks), "pass;", [k for k, ok in checks.items() if not ok])
    assert math.isfinite(sum(r["mu_rho"] for r in res))


if __name__ == "__main__":
    main()
