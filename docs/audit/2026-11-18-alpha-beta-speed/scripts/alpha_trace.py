"""Train the default model from scratch and trace alpha / raw-alpha / softplus slope per head every k steps."""
import sys, json, numpy as np, torch
from nig_timenet_torch.core.config import updated
from nig_timenet_torch.workflow.settings import RunSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.model.heads import head_spec

art, rd, n_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
torch.set_num_threads(4)
s = updated(RunSettings(run_dir=rd), {"data.name": art, "train.device": "cpu", "evaluation.battery.enabled": False})
if over:
    s = updated(s, {"model.overrides": {**s.model.overrides, **over}})
run = steps.prepare(s, rd + "_out")
steps.build(run)
tr, m = run.trainer, run.trainer.model
for g in tr.optimizer.param_groups:
    g["lr"] = s.train.lr
heads = m.head_modules()
raw = {}
for t, h in heads.items():
    h.out.register_forward_hook(lambda _m, _i, o, t=t: raw.__setitem__(t, o.detach()))
xv = run.x["val"][:4096]
yv = {k: v[:4096] for k, v in run.y["val"].items()}
x, y = run.x["train"], run.y["train"]
bs = s.train.batch_size
rows = []
def probe(step):
    m.eval()
    with torch.no_grad():
        o = m(xv)
    r = {"step": step}
    for t in heads:
        K = head_spec("nig_regression", t).output_keys
        a = o[K["alpha"]].squeeze(-1).numpy()
        ra = raw[t][:, 2].numpy()
        r[t] = {"alpha_med": round(float(np.median(a)), 4), "share_excess<0.01": round(float(np.mean(a - heads[t].alpha_min < 0.01)), 3),
                "raw_med": round(float(np.median(ra)), 2), "softplus_slope_med": round(float(np.median(1 / (1 + np.exp(-ra)))), 4)}
    rows.append(r)
    print(json.dumps(r), flush=True)
    m.train()
probe(0)
g = torch.Generator().manual_seed(0)
step = 0
while step < n_steps:
    perm = torch.randperm(len(x), generator=g)
    for ix in perm.split(bs):
        tr.train_step(x[ix], {k: v[ix] for k, v in y.items()}, as_tensors=True)
        step += 1
        if step in (5, 10, 20, 50, 100, 200) or step % 400 == 0 or step == n_steps:
            probe(step)
        if step >= n_steps:
            break
