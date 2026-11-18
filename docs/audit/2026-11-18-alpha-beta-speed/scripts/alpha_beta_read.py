"""Run a trained best.pt (from Drive) on a contract artifact and read its NIG alpha / beta against their floors."""
import sys, json, math, numpy as np, torch
from scipy import optimize, special
from nig_timenet_torch.core.config import updated
from nig_timenet_torch.workflow.settings import RunSettings
from nig_timenet_torch.workflow import run as steps

art, weights, rd = sys.argv[1], sys.argv[2], sys.argv[3]
s = updated(RunSettings(run_dir=rd), {"data.name": art, "train.device": "cpu", "evaluation.battery.enabled": False})
run = steps.prepare(s, rd + "_out")
steps.build(run)
m = run.trainer.model
sd = torch.load(weights, map_location="cpu", weights_only=True)
missing, unexpected = m.load_state_dict(sd, strict=True), None
m.eval()
heads = m.head_modules()
out = {}
for stage in ("val", "test"):
    from nig_timenet_torch.workflow import flat_run
    if stage not in run.x:
        run.x[stage], run.y[stage] = flat_run.flat_inputs(run.art.dataset, run.idx[stage], False)
    x = run.x[stage]
    with torch.no_grad():
        parts = [m(xb) for xb in x.split(4096)]
    o = {k: torch.cat([p[k] for p in parts]).squeeze(-1).numpy().astype(np.float64) for k in parts[0]}
    # raw pre-softplus params of each NIG head (via a hook on head.out)
    out[stage] = (o, {k: v.numpy().astype(np.float64).ravel() for k, v in run.y[stage].items()})

def tfit(z):
    """MLE of a Student-t on z with free df and scale multiplier c (loc 0)."""
    def nll(p):
        df, c = math.exp(p[0]), math.exp(p[1])
        u = z / c
        return -np.mean(special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(c)
                        - (df + 1) / 2 * np.log1p(u * u / df))
    r = optimize.minimize(nll, [math.log(6.0), 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
    return math.exp(r.x[0]), math.exp(r.x[1])

rep = {}
for stage, (o, y) in out.items():
    for t, h in heads.items():
        s_ = float(h._s()) if h._s() is not None else 1.0
        from nig_timenet_torch.model.heads import head_spec; K = head_spec("nig_regression", t).output_keys; mu, nu, al, be = (o[K["mu"]], o[K["nu"]], o[K["alpha"]], o[K["beta"]])
        yy = y[f"y_{t}_reg"]
        fl_b = s_ * s_ * h.beta_min
        scale = np.sqrt(be * (1 + nu) / (nu * al))
        z = (yy - mu) / scale
        ok = np.isfinite(z)
        df_hat, c_hat = tfit(z[ok])
        # NLL gradient wrt alpha at the model's outputs (Amini NLL, label units)
        err = yy - mu; om = 2 * be * (1 + nu)
        g_alpha = -np.log(om) + np.log(nu * err**2 + om) + special.digamma(al) - special.digamma(al + 0.5)
        rep[f"{stage}/{t}"] = {
            "rows": int(ok.sum()), "label_scale": s_,
            "alpha_q": np.quantile(al, [0, .01, .5, .99, 1]).round(5).tolist(),
            "alpha_excess_share<1e-3": float(np.mean(al - h.alpha_min < 1e-3)),
            "alpha_excess_share<1e-2": float(np.mean(al - h.alpha_min < 1e-2)),
            "alpha_share_within5pct(report rule)": float(np.mean(al - h.alpha_min <= 0.05 * h.alpha_min)),
            "beta_floor_report(unscaled)": h.beta_min, "beta_floor_true(label units)": fl_b,
            "beta_share_report_rule": float(np.mean(be - h.beta_min <= 0.05 * h.beta_min)),
            "beta_share_true_floor_5pct": float(np.mean(be - fl_b <= 0.05 * fl_b)),
            "beta/true_floor_q": np.quantile(be / fl_b, [0, .01, .5, .99, 1]).round(2).tolist(),
            "student_t_mle_df_of_residuals": round(df_hat, 3), "implied_alpha=df/2": round(df_hat / 2, 3),
            "scale_mult": round(c_hat, 3),
            "mean_dNLL_dalpha(>0 pushes alpha down)": float(np.mean(g_alpha)),
            "share_rows_dNLL_dalpha>0": float(np.mean(g_alpha > 0)),
            "reg_grad_dalpha=lambda*|err| (lambda 0.01)": float(0.01 * np.mean(np.abs(err))),
            "kurtosis_excess_z": float(((z[ok]-z[ok].mean())**4).mean()/z[ok].var()**2 - 3),
        }
print(json.dumps(rep, indent=1))
