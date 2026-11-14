"""PURPOSE: Card B — does the trainer guide the model to learn? Measured answers, not a loss curve: one-batch
    overfit, gradient and update health (gate A), what weight decay / clipping / EMA actually do, loss-term balance,
    a learning-rate range test, the oracle test on E1, the shuffled-label collapse (E2), the epoch the selection metric
    picks, resume == uninterrupted, and train mode vs eval mode. Verdict: learns / does not learn / learns the wrong
    thing, with the failing check named.
TAGS: doctor, card b, trainer checks, overfit one batch, lr range test, update ratio, loss balance, shuffled labels,
    resume, train eval gap, batch norm moving statistics, selection regret, oracle
PITFALLS: every check builds its OWN fresh model (``common.fresh_trainer``) or restores the state it changed, so the
    context's trained model is untouched; the E1 / E2 sub-runs train the SAME configuration on a small synthetic set
    written under ``out`` (seconds on a CPU); thresholds are named constants below, each with its reason.

``trainer_checks(ctx) -> list[Check]``; ``verdict(checks)`` -> ("learns" | "does not learn" | "learns the wrong
thing", [failing check names]).
"""

import math
import shutil
from pathlib import Path

import numpy as np
import torch

from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io.environments import permute_labels
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, build_synthetic
from nig_timenet_torch.doctor.common import (
    Check,
    DoctorContext,
    fresh_trainer,
    judge,
    make_context,
    predict_outputs,
    reg_key,
    stride,
)
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.layers import BatchNormChannelsLast
from nig_timenet_torch.stats.metrics import pearson, spearman
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.grad_report import grad_report
from nig_timenet_torch.trainer.trainer import FitSettings
from nig_timenet_torch.workflow import flat_run, run_common

__all__ = ["trainer_checks", "verdict", "overfit_one_batch", "lr_range_test", "mode_gap", "resume_check",
           "selection_check", "constant_loss", "oracle_check", "shuffled_check", "NOT_LEARNING", "WRONG_THING"]

OVERFIT_ROWS = 64            # one small batch: a healthy model + optimiser fits it almost exactly
OVERFIT_CORR = 0.9           # corr(mu, label) on that batch after the steps (0.5-0.9 = warn)
MODE_SCALE_BAND = (0.8, 1.25)  # sd(mu in eval mode) / sd(mean mu in train mode); outside = a different function
BN_VAR_RATIO = 2.0           # moving variance / train batch variance of a BatchNorm, either way
E1_CORR = {"high": 0.9, "low": 0.9, "close": 0.5}   # corr(mu, oracle mean) the short E1 run must reach
E1_SLOPE_BAND = (0.75, 1.33)  # slope of the oracle mean on mu (1 = calibrated, > 1 = mu shrunk to its average)
#: check name prefixes that decide the summary verdict
NOT_LEARNING = ("overfit one batch (no regularisation)", "gradients finite", "dead parameters",
                "val loss vs constant predictor", "lr range")
WRONG_THING = ("shuffled labels", "train vs eval mode", "E1 oracle", "batch norm moving", "resume")


def _mu_key(t: str) -> str:
    return head_spec("nig_regression", t).output_keys["mu"]


# ------------------------------------------------------------------------------------------ B1 one-batch overfit
def overfit_one_batch(ctx: DoctorContext, *, steps: int = 300, no_regularisation: bool = True) -> dict:
    """Train a FRESH model on one batch of ``OVERFIT_ROWS`` train rows for ``steps`` steps at the configured optimiser;
    -> {target: corr(mu, label) on the batch in eval mode, "loss_first" / "loss_last"}. ``no_regularisation`` switches
    dropout, input noise and feature dropout off (the question is "can it fit", not "does the regulariser bite")."""
    over = {"dropout": 0.0, "input_noise_std": 0.0, "feature_dropout": 0.0} if no_regularisation else None
    over = over if over is None or ctx.run.ms.architecture == "nig_timenet_v2" else None
    trainer = fresh_trainer(ctx, model=over)
    rows = stride(np.arange(len(ctx.run.x["train"])), OVERFIT_ROWS)
    dev = ctx.run.device
    xb = ctx.run.x["train"][rows].to(dev)
    yb = {k: v[rows].to(dev) for k, v in ctx.run.y["train"].items()}
    for name, sched in trainer.schedules.items():       # the loss as it is at the END of its warm-up
        trainer.scheduled_vars[name] = float(sched.end)
    losses = [trainer.train_step(xb, yb)["raw_loss"] for _ in range(steps)]
    out = predict_outputs(trainer.model, xb)
    res = {t: pearson(out[_mu_key(t)], yb[reg_key(t)].cpu().numpy()) for t in ctx.targets}
    res.update(loss_first=float(np.mean(losses[:5])), loss_last=float(np.mean(losses[-5:])))
    return res


# ------------------------------------------------------------------------------------------ B5 lr range test
def lr_range_test(ctx: DoctorContext, *, lo: float = 1e-6, hi: float = 1.0, steps: int = 80) -> dict:
    """Exponential learning-rate sweep on a fresh model (one batch per step, cycling the train set): the smoothed
    loss per lr, ``lr_best`` (lowest smoothed loss) and ``lr_diverge`` (first lr whose loss is worse than the start
    by as much as the best was better, or non-finite; ``hi`` if none)."""
    trainer = fresh_trainer(ctx)
    for name, sched in trainer.schedules.items():
        trainer.scheduled_vars[name] = float(sched.end)
    bs = ctx.run.fit_settings.batch_size
    x, y, dev = ctx.run.x["train"], ctx.run.y["train"], ctx.run.device
    g = torch.Generator().manual_seed(ctx.seed)
    order = torch.randperm(len(x), generator=g)
    lrs = np.exp(np.linspace(math.log(lo), math.log(hi), steps))
    raw = []
    for i, lr in enumerate(lrs):
        for group in trainer.optimizer.param_groups:
            group["lr"] = float(lr)
        ix = order[(i * bs) % max(len(x) - bs, 1):][:bs]
        raw.append(trainer.train_step(x[ix].to(dev), {k: v[ix].to(dev) for k, v in y.items()})["raw_loss"])
    raw = np.asarray(raw, dtype=np.float64)
    smooth = np.array([np.mean(raw[max(0, i - 2):i + 3]) if np.isfinite(raw[max(0, i - 2):i + 3]).all() else np.inf
                       for i in range(steps)])
    start = float(np.mean(raw[:5]))
    best_i = int(np.argmin(smooth))
    gain = start - smooth[best_i]
    after = np.flatnonzero((smooth[best_i:] > start + max(gain, 1e-9)) | ~np.isfinite(smooth[best_i:]))
    diverge = float(lrs[best_i + after[0]]) if len(after) else float(hi)
    return {"lr": lrs.tolist(), "loss": raw.tolist(), "lr_best": float(lrs[best_i]), "lr_diverge": diverge,
            "loss_start": start, "loss_best": float(smooth[best_i])}


# ------------------------------------------------------------------------------------------ B10 mode gap
def mode_gap(ctx: DoctorContext, *, rows: int = 2048, passes: int = 8) -> dict:
    """Per head: the eval-mode mu against the MEAN train-mode mu over ``passes`` forward passes (dropout / noise /
    BatchNorm batch statistics on) on validation rows -> corr, sd ratio (eval / train) and mean shift in sd units;
    plus per BatchNorm layer the worst ratio of its moving variance to the variance of a train batch."""
    _, x, _ = ctx.xy("val", rows)
    model = ctx.model
    ev = predict_outputs(model, x)
    torch.manual_seed(ctx.seed)
    tr = [predict_outputs(model, x, train_mode=True) for _ in range(passes)]
    out = {}
    for t in ctx.targets:
        k = _mu_key(t)
        m = np.mean([p[k] for p in tr], axis=0)
        sd = float(m.std())
        out[t] = {"corr": pearson(ev[k], m), "sd_ratio": float(ev[k].std() / sd) if sd > 0 else float("nan"),
                  "mean_shift_sd": float((ev[k].mean() - m.mean()) / sd) if sd > 0 else float("nan")}
    bn = {}
    xt = ctx.run.x["train"][torch.as_tensor(stride(np.arange(len(ctx.run.x["train"])), rows))].to(ctx.run.device)
    hooks, seen = [], {}
    for name, mod in model.named_modules():
        if isinstance(mod, BatchNormChannelsLast):
            hooks.append(mod.register_forward_hook(lambda m, inp, o, name=name: seen.__setitem__(name, inp[0])))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(xt)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    for name, inp in seen.items():
        mod = dict(model.named_modules())[name]
        var = inp.reshape(-1, inp.shape[-1]).var(dim=0, unbiased=False)
        ratio = (mod.moving_variance + mod.epsilon) / (var + mod.epsilon)
        bn[name] = float(torch.exp(torch.log(ratio).abs().max()))
    return {"heads": out, "batch_norm_var_ratio": bn}


# ------------------------------------------------------------------------------------------ B9 resume
def resume_check(ctx: DoctorContext, *, rows: int = 2048) -> dict:
    """2 epochs in one go vs 1 epoch + a resumed second epoch (new Trainer, same seed) on a small strided subset:
    the largest absolute difference between the two final checkpoints' weights (0.0 = resume is exact)."""
    r = ctx.run
    tr = torch.as_tensor(stride(np.arange(len(r.x["train"])), rows))
    va = torch.as_tensor(stride(np.arange(len(r.x["val"])), rows // 4))
    data = (r.x["train"][tr], {k: v[tr] for k, v in r.y["train"].items()},
            r.x["val"][va], {k: v[va] for k, v in r.y["val"].items()})
    base = Path(ctx.out) / "resume_check"
    shutil.rmtree(base, ignore_errors=True)

    def fit(run_dir, epochs):
        trainer = fresh_trainer(ctx)
        fs = FitSettings(epochs=epochs, batch_size=r.fit_settings.batch_size, seed=r.fit_settings.seed,
                         early_stop=r.fit_settings.early_stop, grad_clip_norm=r.fit_settings.grad_clip_norm,
                         ema_decay=r.fit_settings.ema_decay, init=r.fit_settings.init)
        ident = run_common.run_identity(trainer, fs, r.ms, r.art, r.settings)
        trainer.fit(*data, fs, run_dir=run_dir, identity=ident)
        return checkpoint.load_checkpoint(run_dir)["model"]

    a = fit(base / "straight", 2)
    fit(base / "resumed", 1)
    b = fit(base / "resumed", 2)
    diff = max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
    shutil.rmtree(base, ignore_errors=True)
    return {"max_abs_diff": diff}


# ------------------------------------------------------------------------------------------ B8 selection
def selection_check(history: list[dict], best_epoch: int | None, targets) -> dict:
    """From the fit history: the epoch the tracker picked vs the epoch with the lowest mean (over heads) validation
    MAE relative to each head's own best; ``regret`` = that relative MAE at the picked epoch minus the best."""
    if not history or best_epoch is None:
        return {"epochs": len(history), "picked": best_epoch, "regret": float("nan")}
    keys = [f"val_{head_spec('nig_regression', t).name}_mae" for t in targets]
    keys = [k for k in keys if k in history[0]]
    mae = np.array([[row[k] for k in keys] for row in history], dtype=np.float64)       # (epochs, heads)
    rel = (mae / mae.min(axis=0)).mean(axis=1)
    sel = np.array([row["val_selection_metric"] for row in history], dtype=np.float64)
    return {"epochs": len(history), "picked": int(best_epoch), "best_by_mae": int(rel.argmin()) + 1,
            "best_by_raw_selection": int(np.nanargmin(sel)) + 1, "regret": float(rel[best_epoch - 1] - rel.min()),
            "rank_corr_selection_vs_mae": float(spearman(sel, rel)) if len(history) >= 4 else float("nan")}


# ------------------------------------------------------------------------------------------ B11 constant loss
@torch.no_grad()
def constant_loss(ctx: DoctorContext, stage: str = "val", rows: int = 20000) -> dict:
    """The raw loss of the model on ``stage`` next to the raw loss of the SAME model's mean output on every row (a
    constant predictor with the model's own average uncertainty)."""
    _, x, _ = ctx.xy(stage, rows)
    r = ctx.run
    pick = torch.as_tensor(stride(np.arange(len(r.idx[stage])), rows))
    if stage not in r.y:
        r.x[stage], r.y[stage] = flat_run.flat_inputs(ctx.dataset, r.idx[stage])
    y = {k: v[pick].to(r.device) for k, v in r.y[stage].items()}
    model = ctx.model
    model.eval()
    parts = [model(x[i:i + 4096].to(r.device)) for i in range(0, len(x), 4096)]
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    const = {k: v.mean(dim=0, keepdim=True).expand_as(v) for k, v in out.items()}
    return {"model": float(r.trainer.compute_losses(None, y, outputs=out)[1]),
            "constant": float(r.trainer.compute_losses(None, y, outputs=const)[1])}


# ------------------------------------------------------------------------------------------ B6 / B7 E1 and E2
def _small_e1(ctx: DoctorContext, name: str, permute: bool) -> Path | None:
    """A small E1 (or its label-permuted E2) with the context's window length, written under ``out``; None when the
    configuration cannot read it (E1 has the three log-move channels)."""
    t, f = ctx.run.x["train"].shape[1], ctx.run.x["train"].shape[2]
    if f != 3 or ctx.run.ms.architecture not in ("nig_timenet_v2", "tiny_mlp"):
        return None
    path = Path(ctx.out) / name
    ds = build_synthetic(SyntheticSettings(n_coins=16 if ctx.quick else 32, n_bars=2000 if ctx.quick else 4000,
                                           window=t, burn_in_bars=max(128, 4 * t)))
    if permute:
        ds = permute_labels(ds, seed=0)
    write_artifact(ds, path, source_commit="", overwrite=True, producer="nig_timenet_torch.doctor")
    return path


def _sub_context(ctx: DoctorContext, path: Path, tag: str) -> DoctorContext:
    s = ctx.run.settings
    shutil.rmtree(Path(ctx.out) / f"{tag}_run", ignore_errors=True)
    over = {k: getattr(s.train, k) for k in ("lr", "weight_decay", "label_smoothing", "selection_metric",
                                             "grad_clip_norm", "ema_decay", "init", "weight_decay_groups",
                                             "uncertainty_weighting", "class_loss_weight", "lambda_reg_end",
                                             "lambda_calib_end", "beta_nll", "beta_nll_normalize")}
    return make_context(str(path), Path(ctx.out) / tag, run_dir=str(Path(ctx.out) / f"{tag}_run"),
                        model=ctx.run.ms.overrides, architecture=ctx.run.ms.architecture, no_preset=True,
                        train=over, epochs=3 if ctx.quick else 6, batch_size=min(ctx.run.fit_settings.batch_size, 256),
                        seed=ctx.seed, quick=ctx.quick)


def oracle_check(e1: DoctorContext) -> dict:
    """On an E1 context: per head corr and slope of the oracle mean on the model's mu (test rows), and the model's
    rank correlation with the label next to the oracle's own."""
    idx, x, y = e1.xy("test")
    out = predict_outputs(e1.model, x)
    res = {}
    for t in e1.targets:
        oracle = np.asarray(e1.dataset[f"oracle_{reg_key(t)}_mean"])[idx]
        mu = out[_mu_key(t)]
        c = np.cov(oracle, mu)
        res[t] = {"corr": pearson(oracle, mu), "slope": float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else float("nan"),
                  "ic_model": float(spearman(y[reg_key(t)], mu)), "ic_oracle": float(spearman(y[reg_key(t)], oracle))}
    return res


def shuffled_check(e2: DoctorContext) -> dict:
    """On an E2 (permuted labels) context: validation loss vs the constant predictor, the worst |rank correlation| of
    a head with its (permuted) label on test rows, and how far the TRAIN loss fell below the constant (memorising)."""
    idx, x, y = e2.xy("test")
    out = predict_outputs(e2.model, x)
    ic = {t: float(spearman(y[reg_key(t)], out[_mu_key(t)])) for t in e2.targets}
    return {"val": constant_loss(e2, "val"), "train": constant_loss(e2, "train"), "ic": ic}


# ------------------------------------------------------------------------------------------------------ all
def _gate_a(ctx: DoctorContext, checks: list[Check]) -> None:
    r = ctx.run
    batches = flat_run.batch_set(r.dataset, r.idx["train"], r.fit_settings.batch_size, ctx.seed)
    if ctx.quick:
        batches = batches[:4]
    names = {"A1": "gradients finite", "A2": "dead parameters", "A3": "layer gradient scale / batch spike",
             "A4": "median update / weight of one step", "A5": "head share of the trunk gradient",
             "A6": "head gradient conflict (mean cosine)", "A7": "units that never activate"}
    actions = {"A1": "a NaN / inf gradient: find the layer (trainer/grad_report.py layers)",
               "A2": "these tensors never get a gradient: a detached path or a loss term that is off",
               "A3": "one layer's gradient is orders of magnitude off the others: init or normalisation of that layer",
               "A4": "the step is too small or too large for the weights: learning rate / init scale",
               "A5": "one head owns the trunk or a head is starved: loss weights / task weighting",
               "A6": "the heads pull the trunk in opposite directions: separate the conflicting head or detach it",
               "A7": "units that are never active are capacity the model cannot use: input scale / init of that layer"}
    for phase, trainer in (("start", fresh_trainer(ctx)), ("end", r.trainer)):
        rep = grad_report(trainer, r.x["train"], r.y["train"], batches)
        for a, c in rep["checks"].items():
            v = c["value"]
            v = (f"{len(v)} out of band" if isinstance(v, list) else
                 {k: round(x, 3) for k, x in v.items()} if isinstance(v, dict) else v)
            checks.append(Check("B", f"trainer/gradients ({phase})", f"{names[a]} [{a}]", str(v) if isinstance(
                v, dict) else v, c.get("rule", "")[:70], c["verdict"],
                actions[a] if c["verdict"] != "pass" else "", {k: x for k, x in c.items() if k != "rule"}))
        if phase == "end":
            ctx.notes["update_ratio_end"] = rep["checks"]["A4"]["value"]


def trainer_checks(ctx: DoctorContext) -> list[Check]:
    checks: list[Check] = []
    t = ctx.run.settings.train
    # B1 one-batch overfit
    for label, noreg in (("no regularisation", True), ("as configured", False)):
        res = overfit_one_batch(ctx, steps=200 if ctx.quick else 400, no_regularisation=noreg)
        worst_c = min(res[h] for h in ctx.targets)
        v = judge(worst_c, fail_below=0.5, warn_below=OVERFIT_CORR) if noreg else "info"
        checks.append(Check("B", "trainer/optimisation", f"overfit one batch ({label}): worst head corr(mu, label)",
                            worst_c, f">= {OVERFIT_CORR} (a constant predictor: 0)", v,
                            "the model cannot fit 64 rows: the loss, the head parametrisation or the optimiser step "
                            "is broken (look at A4 and the lr range test)" if v in ("warn", "fail") else "", res))
    # B2 / B4 gate A at the start and at the end
    _gate_a(ctx, checks)
    # B3 what do weight decay / clipping / EMA do
    upd = ctx.notes.get("update_ratio_end", float("nan"))
    decay = t.lr * t.weight_decay
    checks.append(Check("B", "trainer/optimiser", "weight decay per step (lr x wd) / median update per step",
                        decay / upd if upd and upd > 0 else float("nan"), "< 0.001 = no effect", "info",
                        "weight decay is a no-op at this size" if upd and decay / upd < 1e-3 else "",
                        {"lr_x_wd": decay, "update_ratio": upd}))
    checks.append(Check("B", "trainer/optimiser", "gradient clipping / EMA",
                        f"clip {t.grad_clip_norm}, ema {t.ema_decay}", "None = off", "info"))
    # B5 lr range
    lr = lr_range_test(ctx, steps=60 if ctx.quick else 120)
    v = "fail" if t.lr > lr["lr_diverge"] / 3 else "warn" if t.lr < lr["lr_best"] / 100 else "pass"
    checks.append(Check("B", "trainer/optimiser", "lr range: configured lr vs the sweep",
                        t.lr, f"best {lr['lr_best']:.1e}, diverges at {lr['lr_diverge']:.1e}", v,
                        {"fail": "the configured learning rate is within 3x of divergence: lower it",
                         "warn": "the configured learning rate is > 100x below the best of the sweep: raise it",
                         "pass": ""}[v], lr))
    # B11 val loss vs the constant predictor
    cl = constant_loss(ctx, "val")
    checks.append(Check("B", "trainer/result", "val loss vs constant predictor (same model's mean output)",
                        cl["model"], cl["constant"], "pass" if cl["model"] < cl["constant"] else "fail",
                        "" if cl["model"] < cl["constant"] else "the trained model is no better than its own average "
                        "output on validation: nothing was learned that generalises"))
    # B8 selection
    sel = selection_check(ctx.run.result.history, ctx.run.result.best_epoch, ctx.targets)
    v = "info" if not math.isfinite(sel["regret"]) or sel["epochs"] < 3 else judge(sel["regret"], warn_above=0.02,
                                                                                    fail_above=0.10)
    checks.append(Check("B", "trainer/selection", "picked epoch vs the best-MAE epoch (relative MAE regret)",
                        sel["regret"], f"picked {sel['picked']}, best by MAE {sel.get('best_by_mae')}, "
                        f"by raw metric {sel.get('best_by_raw_selection')} of {sel['epochs']}", v,
                        "the selection metric (or its smoothing) picks an epoch with a clearly worse error: select on "
                        "a target metric, or shorten the smoothing window" if v in ("warn", "fail") else "", sel))
    # B10 train vs eval
    gap = mode_gap(ctx)
    for h, g in gap["heads"].items():
        ok = MODE_SCALE_BAND[0] <= g["sd_ratio"] <= MODE_SCALE_BAND[1] and g["corr"] >= 0.9
        checks.append(Check("B", "trainer/modes", f"train vs eval mode, {h}: sd ratio eval / train (corr)",
                            f"{g['sd_ratio']:.2f} ({g['corr']:.2f})", f"{MODE_SCALE_BAND} (corr >= 0.9)",
                            "pass" if ok else "fail",
                            "" if ok else "the evaluated function differs from the trained one: BatchNorm moving "
                            "statistics (bn_warmup), dropout / noise scale", g))
    for name, ratio in gap["batch_norm_var_ratio"].items():
        ok = ratio <= BN_VAR_RATIO
        checks.append(Check("B", "trainer/modes", f"batch norm moving variance vs a train batch: {name}", ratio,
                            f"<= {BN_VAR_RATIO}", "pass" if ok else "fail",
                            "" if ok else "the moving statistics did not reach the data's (short run with momentum "
                            "0.99): model override bn_warmup"))
    # B9 resume
    res = resume_check(ctx)
    checks.append(Check("B", "trainer/resume", "resume vs uninterrupted: max weight difference", res["max_abs_diff"],
                        "0 (bitwise on CPU)", "pass" if res["max_abs_diff"] <= 1e-6 else "fail",
                        "" if res["max_abs_diff"] <= 1e-6 else "a resumed run is not the run it continues: some "
                        "state is missing from the checkpoint or the per-epoch reseed"))
    # B6 oracle on E1, B7 shuffled labels
    e1 = ctx if ctx.is_e1 else None
    if e1 is None:
        path = _small_e1(ctx, "e1", permute=False)
        e1 = _sub_context(ctx, path, "e1") if path is not None else None
    if e1 is None:
        checks.append(Check("B", "trainer/known truth", "E1 oracle", "skipped", "", "info",
                            "E1 has 3 input channels; this configuration reads another input"))
    else:
        for h, o in oracle_check(e1).items():
            ok = o["corr"] >= E1_CORR.get(h, 0.9) and E1_SLOPE_BAND[0] <= o["slope"] <= E1_SLOPE_BAND[1]
            checks.append(Check("B", "trainer/known truth", f"E1 oracle, {h}: corr(mu, true mean) / slope",
                                f"{o['corr']:.2f} / {o['slope']:.2f}",
                                f">= {E1_CORR.get(h, 0.9)} / {E1_SLOPE_BAND}", "pass" if ok else "fail",
                                "" if ok else "on data with a known answer the mean is not recovered: slope > 1 = "
                                "mu shrunk (mode gap, head scale); low corr = the signal is not learned", o))
        path = _small_e1(ctx, "e2", permute=True)
        e2 = _sub_context(ctx, path, "e2")
        sh = shuffled_check(e2)
        worst_ic = max(abs(v) for v in sh["ic"].values())
        ok = worst_ic < 0.06 and sh["val"]["model"] >= sh["val"]["constant"] - 0.05 * abs(sh["val"]["constant"])
        checks.append(Check("B", "trainer/known truth", "shuffled labels (E2): worst |IC| on test",
                            worst_ic, "~0 (labels carry no signal)", "pass" if ok else "fail",
                            "" if ok else "skill on permuted labels: a leak between inputs and labels, or the "
                            "evaluation rewards something trivial", sh))
        checks.append(Check("B", "trainer/known truth", "shuffled labels (E2): train loss below the constant's",
                            sh["train"]["constant"] - sh["train"]["model"], "0 = nothing memorised", "info",
                            "", sh["train"]))
    return checks


def verdict(checks: list[Check]) -> tuple[str, list[str]]:
    """("learns" | "does not learn" | "learns the wrong thing", the names of the failing checks that decide it)."""
    failing = [c for c in checks if c.card == "B" and c.verdict == "fail"]
    not_learning = [c.check for c in failing if c.check.startswith(NOT_LEARNING) or "[A1]" in c.check
                    or "[A2]" in c.check]
    wrong = [c.check for c in failing if c.check.startswith(WRONG_THING)]
    if not_learning:
        return "does not learn", not_learning
    if wrong:
        return "learns the wrong thing", wrong
    return "learns", [c.check for c in failing]
