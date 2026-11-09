"""PURPOSE: E1 arm runner: train configuration arms on the synthetic E1 env, gate A start / end + gate B / C rows,
    and per-head mu calibration against the known E1 conditional mean (mean ratio, spread ratio, corr, slope).
TAGS: e1, arms, synthetic, mu calibration, slope, oracle, readout gap, model doctor, exploration
PITFALLS: exploratory (rule C5): an arm is judged on >= 2 seeds at equal epochs (rule C4; training length alone
    moves the slope, FX1); ``readout_gap`` refits a ridge readout of the LABEL from each NIG head's hidden layer on
    train rows, so "readout slope 1, model slope 1.5" means the head's own last step loses the scale, not the trunk.

``slope_oracle_on_mu`` is the regression slope of the true conditional mean on the model's mu: 1.0 for a calibrated
mean, > 1 when mu is compressed towards its average (E1 baseline 1.5-1.7). CLI:
``python -m nig_timenet_torch.workflow.e1_arms <out_dir> '<json arms>' <seeds csv> [<env dir>] [--epochs N]``;
an arm is ``{"train": {TrainSettings fields}, "model": {overrides over the default preset}, "architecture": name,
"no_preset": bool}``; one ``res_<arm>.json`` per arm and one summary line per arm.
"""

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, oracle_keys, write_synthetic
from nig_timenet_torch.eval import gates
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.taps import record_taps
from nig_timenet_torch.stats.metrics import pearson
from nig_timenet_torch.trainer.grad_report import grad_report
from nig_timenet_torch.workflow import flat_run, gate_run, run
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings, default_model_settings

__all__ = ["mu_calibration", "readout_gap", "arm_settings", "run_arm", "summary_line", "main"]

ARM_KEYS = ("train", "model", "architecture", "no_preset")


def _calib(oracle: np.ndarray, mu: np.ndarray) -> dict:
    c = np.cov(oracle, mu)
    return {"mean_ratio": float(mu.mean() / oracle.mean()) if abs(oracle.mean()) > 1e-12 else None,
            "sd_ratio": float(mu.std() / oracle.std()), "corr": pearson(oracle, mu),
            "slope_oracle_on_mu": float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else None}


def mu_calibration(dataset: dict, idx: np.ndarray, table, targets) -> dict:
    """{target: calibration of the table's mu against the E1 oracle mean of the same rows}."""
    return {t: _calib(np.asarray(dataset[oracle_keys(t)[0]])[idx],
                      table[head_spec("nig_regression", t).output_keys["mu"]].to_numpy()) for t in targets}


@torch.no_grad()
def _head_inputs(model, x: torch.Tensor, chunk: int = 2048) -> tuple[dict, dict]:
    """({target: silu(head.fc) activations}, {target: model mu}) of every row of ``x`` (eval mode)."""
    heads = model.head_modules()
    names = {m: n for n, m in model.named_modules()}
    acts = {t: [] for t in heads}
    mus = {t: [] for t in heads}
    model.eval()
    for i in range(0, len(x), chunk):
        with record_taps(model, [names[h.fc] for h in heads.values()]) as taps:
            out = model(x[i:i + chunk])
        for t, h in heads.items():
            acts[t].append(torch.nn.functional.silu(taps[names[h.fc]]).cpu().numpy())
            mus[t].append(out[head_spec("nig_regression", t).output_keys["mu"]].cpu().numpy().ravel())
    return {t: np.concatenate(v) for t, v in acts.items()}, {t: np.concatenate(v) for t, v in mus.items()}


def readout_gap(r: run.Run, *, n_train: int = 20000, ridge: float = 1.0) -> dict:
    """Per NIG head: the model's test mu calibration next to a ridge readout of the label fitted on ``n_train`` train
    rows from the head's own hidden layer (the information the head had). A readout near slope 1 with a model slope
    well above 1 locates the loss of scale in the head's last step (parametrisation / optimisation), not the trunk."""
    ds, model = r.dataset, r.trainer.model
    tri, tei = r.idx["train"][:n_train], r.idx["test"]
    x_tr, _ = flat_run.flat_inputs(ds, tri)
    x_te, _ = flat_run.flat_inputs(ds, tei)
    a_tr, _ = _head_inputs(model, x_tr.to(r.device))
    a_te, mu_te = _head_inputs(model, x_te.to(r.device))
    out = {}
    for t in a_tr:
        f_tr, f_te = a_tr[t], a_te[t]
        m, s = f_tr.mean(0), f_tr.std(0) + 1e-8
        z_tr = np.c_[np.ones(len(f_tr)), (f_tr - m) / s]
        z_te = np.c_[np.ones(len(f_te)), (f_te - m) / s]
        y = np.asarray(ds[head_spec("nig_regression", t).label_key])[tri]
        w = np.linalg.solve(z_tr.T @ z_tr + ridge * np.eye(z_tr.shape[1]), z_tr.T @ y)
        oracle = np.asarray(ds[oracle_keys(t)[0]])[tei]
        out[t] = {"model": _calib(oracle, mu_te[t]), "readout": _calib(oracle, z_te @ w)}
    return out


def arm_settings(arm: dict, run_dir: Path, env: Path, seed: int, epochs: int) -> RunSettings:
    """The RunSettings of one arm (unknown arm keys raise): model overrides merge over the default preset unless the
    arm names another architecture or ``no_preset``; batch 512 on CPU, ``epochs`` unless the arm's train sets it."""
    unknown = sorted(set(arm) - set(ARM_KEYS))
    if unknown:
        raise KeyError(f"unknown arm key(s) {unknown}; known: {ARM_KEYS}")
    base = default_model_settings()
    arch = arm.get("architecture", base.architecture)
    keep = arch == base.architecture and not arm.get("no_preset", False)
    ms = ModelSettings(arch, {**(base.overrides if keep else {}), **arm.get("model", {})})
    train = from_dict(TrainSettings, {"device": "cpu", "batch_size": 512, "epochs": epochs, **arm.get("train", {})})
    return RunSettings(run_dir=str(run_dir), model=ms, train=replace(train, seed=seed), split=SplitSettings(),
                       data=LoaderSettings(name=str(env), mmap=False, local_dir=""))


def run_arm(name: str, arm: dict, seed: int, env: Path, root: Path, *, epochs: int = 3) -> dict:
    """Train one arm x seed on ``env`` (folder ``root/name/s<seed>`` rebuilt), return gate rows + calibration."""
    t0 = time.monotonic()
    run_dir = root / name / f"s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)  # an arm never resumes (this tool owns the folder)
    r = run.build(run.prepare(arm_settings(arm, run_dir, env, seed, epochs)))
    batches = flat_run.batch_set(r.dataset, r.idx["train"], r.fit_settings.batch_size, seed)
    a0 = grad_report(r.trainer, r.x["train"], r.y["train"], batches)
    run.train(r)
    a1 = grad_report(r.trainer, r.x["train"], r.y["train"], batches)
    st = gate_run._stage(r, gates.ENVS[0], seed, with_oracle=True, with_losses=False)
    rows = (gates.gate_a_rows(gates.ENVS[0], seed, "start", a0) + gates.gate_a_rows(gates.ENVS[0], seed, "end", a1)
            + gates.stage_rows(st))
    return {"seed": seed, "rows": rows, "mu": mu_calibration(r.dataset, r.idx["test"], st.table, r.targets),
            "readout": readout_gap(r), "best_epoch": r.result.best_epoch, "history": r.result.history,
            "seconds": round(time.monotonic() - t0)}


def _row(res: dict, gate: str, check: str, head: str) -> dict | None:
    return next((x for x in res["rows"] if x["gate"] == gate and x["check"] == check and x["head"] == head), None)


def _fails(res: dict) -> int:
    return sum(1 for x in res["rows"] if x["blocking"] and x["verdict"] == "fail")


def summary_line(name: str, runs: list[dict]) -> str:
    """One line per arm, means over seeds: per head mu mean ratio, slope, corr and T-sharp; blocking gate fails."""
    def g(fn):
        vals = [v for v in (fn(r) for r in runs) if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    parts = [f"{name:18s} n={len(runs)}"]
    for t in runs[0]["mu"]:
        tsharp = g(lambda r, t=t: (_row(r, "C", "t_sharp_vs_oracle", t) or {}).get("value"))
        parts.append(f"{t} ratio {g(lambda r, t=t: r['mu'][t]['mean_ratio']):.2f} "
                     f"slope {g(lambda r, t=t: r['mu'][t]['slope_oracle_on_mu']):.2f} "
                     f"corr {g(lambda r, t=t: r['mu'][t]['corr']):+.3f} Tsharp {tsharp:.2f}")
    parts.append(f"fails {g(_fails):.1f}")
    return " | ".join(parts)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir")
    p.add_argument("arms", help="JSON {arm name: arm}")
    p.add_argument("seeds", help="comma-separated seeds")
    p.add_argument("env", nargs="?", default=None, help="E1 env folder (default: <out_dir>/envs/E1, written once)")
    p.add_argument("--epochs", type=int, default=3)
    a = p.parse_args(argv)
    root = Path(a.out_dir)
    env = Path(a.env) if a.env else root / "envs" / "E1"
    if not (env / "manifest.json").exists():
        write_synthetic(env, from_dict(SyntheticSettings, {}), overwrite=True)
    results = {}
    for name, arm in json.loads(a.arms).items():
        results[name] = [run_arm(name, arm, int(s), env, root, epochs=a.epochs) for s in a.seeds.split(",")]
        (root / f"res_{name}.json").write_text(json.dumps(results[name], default=float))
        print(summary_line(name, results[name]), flush=True)
    return results


if __name__ == "__main__":
    main()
