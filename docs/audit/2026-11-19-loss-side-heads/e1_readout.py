"""LS1 E1 diagnostic: can a linear read-out of the trunk h rank the oracle payoff as well as the closed form of mu?

Usage: python docs/audit/2026-11-19-loss-side-heads/e1_readout.py <work dir of e1_known_answer.py> [--seeds 0,1,2]
Restores arm R of each seed (best weights), fits a ridge of each side's realised payoff on 50k train rows of h and of
[h, mu, s] and reads the Spearman with the oracle payoff on TEST. Ridge near the closed form = the information is in h
and the side head's own training loses it; ridge near the head = the information is not linearly in h either.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_known_answer import ARMS, COST, closed_form, rho  # noqa: E402

from nig_timenet_torch.data_io.synthetic import oracle_keys  # noqa: E402
from nig_timenet_torch.model.heads import head_spec  # noqa: E402
from nig_timenet_torch.stats.nig import nig_predictive_std  # noqa: E402
from nig_timenet_torch.trainer.tasks import side_payoffs  # noqa: E402
from nig_timenet_torch.workflow import e1_arms, flat_run, run  # noqa: E402

NIG = head_spec("nig_regression", "close")


def feats(r, idx):
    x, _ = flat_run.flat_inputs(r.dataset, idx)
    m = r.trainer.model.eval()
    with torch.no_grad():
        h = torch.cat([m.trunk(x[i:i + 4096]) for i in range(0, len(x), 4096)]).numpy()
        out = {k: torch.cat([m(x[i:i + 4096])[k] for i in range(0, len(x), 4096)]).numpy()
               for k in (NIG.output_keys[n] for n in ("mu", "nu", "alpha", "beta"))}
    mu = out[NIG.output_keys["mu"]]
    s = nig_predictive_std(*(out[NIG.output_keys[n]] for n in ("nu", "alpha", "beta")))
    return h, mu, s


def ridge(z_tr, y, z_te, lam=1.0):
    m, sd = z_tr.mean(0), z_tr.std(0) + 1e-8
    a = np.c_[np.ones(len(z_tr)), (z_tr - m) / sd]
    b = np.c_[np.ones(len(z_te)), (z_te - m) / sd]
    return b @ np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    env = a.work / "e1"
    print("seed  side   closed_form  ridge(h)  ridge(h,mu,s)  ridge(mu,s)")
    for seed in (int(s) for s in a.seeds.split(",")):
        st = e1_arms.arm_settings(ARMS["R"], a.work / "runs" / "R" / f"s{seed}", env, seed, 3)
        r = run.restore(run.build(run.prepare(st)))
        tri, tei = r.idx["train"][:50000], r.idx["test"]
        h_tr, mu_tr, s_tr = feats(r, tri)
        h_te, mu_te, s_te = feats(r, tei)
        y_tr = np.asarray(r.dataset[NIG.label_key])[tri]
        m_key, v_key = oracle_keys("close")
        o_long, o_short = closed_form(np.asarray(r.dataset[m_key])[tei], np.asarray(r.dataset[v_key])[tei])
        e_long, e_short = closed_form(mu_te, s_te)
        p_long, p_short = (t.numpy() for t in side_payoffs(torch.as_tensor(y_tr), COST))
        for side, p, o, e in (("long", p_long, o_long, e_long), ("short", p_short, o_short, e_short)):
            r_h = ridge(h_tr, p, h_te)
            r_hm = ridge(np.c_[h_tr, mu_tr, s_tr], p, np.c_[h_te, mu_te, s_te])
            r_m = ridge(np.c_[mu_tr, s_tr], p, np.c_[mu_te, s_te])
            print(f"{seed:>4}  {side:<5}  {rho(e, o):>11.3f}  {rho(r_h, o):>8.3f}  {rho(r_hm, o):>13.3f}  "
                  f"{rho(r_m, o):>11.3f}")


if __name__ == "__main__":
    main()
