"""PURPOSE: Gate A of MODEL_DEV_PROTOCOL section 2 (gradient health, checks A1-A7) on the representative batch set,
    with the planned optimizer and learning rate; one call = one measurement (step 0 or the end of a short run).
TAGS: gate a, gradient health, grad report, dead parameters, update weight ratio, head gradient share, head conflict,
    dead units, representative batch set
PITFALLS: A5 / A6 use the shared TRUNK (parameters two heads reach, minus the head modules' own), unlike the T4
    trace's gradient share, which also counts the close NIG head the ordered means share; gradient code lives here,
    not in eval/ or workflow/ (rule G5); A4 takes ONE real optimizer step and then restores the model, optimizer and
    Kendall state exactly; A3's spike test compares the R batch norms of one
    measurement (the protocol's in-run spike needs per-step norms, not logged yet); A7 hooks the pre-activation layers
    the model lists in ``activation_points`` (activations are functional here), a unit is dead when its pre-activation
    is <= 0 on every row of the batch set.

``grad_report(trainer, x, y, batches)`` -> ``{"checks": {A1..A7: {"verdict", "value", ...}}, "layers": {...}}`` with
``verdict`` in pass / warn / fail. Thresholds are the protocol's table (constants below). "Layer" = a top-level child
module of the model that holds trainable parameters. The judged heads of A5 / A6 are the trainer's targets with a
loss weight > 0. ``x`` / ``y`` are the TRAINING tensors (any device; batches move to the model's device).
"""

import copy
import itertools
import math

import numpy as np
import torch

from nig_timenet_torch.trainer.trace import head_gradients

__all__ = ["grad_report", "A2_MIN_NORM", "A3_BAND", "A3_SPIKE", "A4_PASS", "A4_FAIL", "A4_MATRIX", "A5_MIN_SHARE",
           "A5_MAX_SHARE", "A6_FAIL", "A7_WARN", "A7_FAIL", "worst"]

A2_MIN_NORM = 1e-12                 # a tensor is dead when its grad norm is <= this on every batch
A3_BAND = (1e-3, 1e3)               # per-layer RMS gradient / median layer
A3_SPIKE = (10.0, 100.0)            # batch global grad norm / median: warn, fail
A4_PASS = (1e-4, 1e-2)              # median update / weight ratio of one optimizer step
A4_FAIL = (1e-5, 1e-1)              # outside: fail (between PASS and FAIL: warn)
A4_MATRIX = (1e-6, 1e-1)            # any weight matrix outside: fail
A5_MIN_SHARE, A5_MAX_SHARE = 0.02, 0.60
A6_FAIL = -0.10                     # mean pairwise cosine below: fail (between it and 0: warn)
A7_WARN, A7_FAIL = 0.20, 0.50       # share of never-active units per layer
_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def worst(verdicts) -> str:
    """The worst of pass / warn / fail."""
    return max(verdicts, key=_ORDER.__getitem__, default="pass")


def _device(trainer) -> torch.device:
    first = next(iter(trainer.model.parameters()), None)
    return torch.device("cpu") if first is None else first.device


def _batches(x, y, batches, dev):
    for b in batches:
        ix = torch.as_tensor(np.asarray(b), dtype=torch.long)
        yield x[ix].to(dev), {k: v[ix].to(dev) for k, v in y.items()}


def _layer_of(name: str) -> str:
    return name.split(".", 1)[0]


def _total_grads(trainer, xb, yb):
    """(total loss, per named parameter gradient or None) of one batch in train() mode, objective as optimised."""
    model = trainer.model
    was = model.training
    model.train()
    try:
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):   # the #23 class (RNN backward)
            total, *_ = trainer.compute_losses(xb, yb, training=True)
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            grads = torch.autograd.grad(total, [p for _, p in named], allow_unused=True)
    finally:
        model.train(was)
    return total, {n: g for (n, _), g in zip(named, grads, strict=True)}


def _a1_a3(trainer, x, y, batches, dev, allow_dead) -> tuple[dict, dict]:
    numel = {n: p.numel() for n, p in trainer.model.named_parameters() if p.requires_grad}
    finite, max_norm, layer_rms, glob = True, dict.fromkeys(numel, 0.0), {}, []
    for xb, yb in _batches(x, y, batches, dev):
        total, grads = _total_grads(trainer, xb, yb)
        finite &= bool(torch.isfinite(total))
        sq, layer_sq = 0.0, dict.fromkeys({_layer_of(n) for n in numel}, 0.0)
        for n, g in grads.items():
            v = 0.0 if g is None else float(g.detach().double().pow(2).sum())
            finite &= g is None or bool(torch.isfinite(g).all())
            max_norm[n] = max(max_norm[n], math.sqrt(v) if math.isfinite(v) else math.inf)
            sq += v
            layer_sq[_layer_of(n)] += v
        glob.append(math.sqrt(sq))
        for layer, v in layer_sq.items():
            n_el = sum(numel[n] for n in numel if _layer_of(n) == layer)
            layer_rms.setdefault(layer, []).append(math.sqrt(v / n_el))
    dead = sorted(n for n, v in max_norm.items() if v <= A2_MIN_NORM and n not in allow_dead)
    off = _scheduled_off(trainer, x, y, batches, dev, dead)
    rms = {k: float(np.mean(v)) for k, v in layer_rms.items()}
    med = float(np.median([v for v in rms.values() if v > 0])) if any(v > 0 for v in rms.values()) else 0.0
    ratio = {k: (v / med if med > 0 else float("nan")) for k, v in rms.items()}
    off_layers = {k for k in rms if all(n in off for n in numel if _layer_of(n) == k)}   # judged by A2
    out_band = sorted(k for k, r in ratio.items() if k not in off_layers and not (A3_BAND[0] <= r <= A3_BAND[1]))
    spike = float(max(glob) / np.median(glob)) if glob and np.median(glob) > 0 else float("nan")
    a3 = "fail" if out_band or not spike < A3_SPIKE[1] else "warn" if spike > A3_SPIKE[0] else "pass"
    checks = {"A1": {"verdict": "pass" if finite else "fail", "value": finite, "rule": "all finite on all R batches"},
              "A2": {"verdict": "fail" if set(dead) - set(off) else "warn" if dead else "pass", "value": len(dead),
                     "dead": dead, "scheduled_off": off,
                     "rule": f"every trainable tensor has grad norm > {A2_MIN_NORM:g} on at least one batch; a tensor "
                             f"that gets a gradient once the loss schedules reach their end value is 'scheduled "
                             f"off' (warn)"},
              "A3": {"verdict": a3, "value": out_band, "spike_max_over_median": spike,
                     "rule": f"layer RMS gradient within {list(A3_BAND)} x the median layer; batch norm spike < "
                             f"{A3_SPIKE[1]:g} x median ({A3_SPIKE[0]:g} = warn)"}}
    return checks, {"rms": rms, "rms_over_median": ratio}


def _scheduled_off(trainer, x, y, batches, dev, dead: list[str]) -> list[str]:
    """The dead tensors that DO get a gradient on the first batch when every scheduled loss variable is set to its
    schedule's end value (e.g. the confidence head while ``lambda_calib`` is still 0): off by the schedule, not by a
    wiring bug. The trainer's scheduled values are restored."""
    if not dead or not trainer.schedules:
        return []
    saved = dict(trainer.scheduled_vars)
    try:
        trainer.scheduled_vars.update({n: float(sch.end) for n, sch in trainer.schedules.items()})
        xb, yb = next(_batches(x, y, batches[:1], dev))
        _, grads = _total_grads(trainer, xb, yb)
    finally:
        trainer.scheduled_vars.clear()
        trainer.scheduled_vars.update(saved)
    return sorted(n for n in dead if grads[n] is not None and float(grads[n].detach().norm()) > A2_MIN_NORM)


def _a4(trainer, x, y, batches, dev, skip=()) -> dict:
    """One real optimizer step on the first batch; the model, optimizer and Kendall states are restored after."""
    model, opt = trainer.model, trainer.optimizer
    saved = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()),
             None if trainer.uncertainty_layer is None else copy.deepcopy(trainer.uncertainty_layer.state_dict()))
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rng = torch.get_rng_state()
    try:
        xb, yb = next(_batches(x, y, batches[:1], dev))
        trainer.train_step(xb, yb)
        ratios, matrix = {}, {}
        for n, p in model.named_parameters():
            if n in before and n not in skip and float(before[n].norm()) > 0:
                r = float((p.detach() - before[n]).norm() / before[n].norm())
                ratios[n] = r
                if before[n].ndim >= 2:
                    matrix[n] = r
    finally:
        model.load_state_dict(saved[0])
        opt.load_state_dict(saved[1])
        if saved[2] is not None:
            trainer.uncertainty_layer.load_state_dict(saved[2])
        torch.set_rng_state(rng)
    med = float(np.median(list(ratios.values()))) if ratios else float("nan")
    bad = sorted(n for n, r in matrix.items() if not (A4_MATRIX[0] <= r <= A4_MATRIX[1]))
    verdict = ("fail" if bad or not (A4_FAIL[0] <= med <= A4_FAIL[1]) else
               "pass" if A4_PASS[0] <= med <= A4_PASS[1] else "warn")
    return {"verdict": verdict, "value": med, "matrices_outside": bad,
            "rule": f"median update / weight in {list(A4_PASS)} (fail outside {list(A4_FAIL)}); every weight matrix "
                    f"in {list(A4_MATRIX)} (A2's dead tensors are left out: their update is the weight decay only)"}


def _head_param_ids(model) -> set[int]:
    """ids of the parameters inside the model's head modules (``head_modules()``, e.g. the NIG heads): not trunk."""
    heads = getattr(model, "head_modules", None)
    return set() if heads is None else {id(p) for m in heads().values() for p in m.parameters()}


def _a5_a6(trainer, x, y, batches, dev) -> dict:
    """On the SHARED TRUNK: parameters at least two heads reach, minus those inside a head module (the ordered means
    make the close NIG head reachable from the high / low losses; that coupling is not the trunk)."""
    judged = [t for t in trainer.target_names if trainer.loss_weights[t] > 0]
    head_ids = _head_param_ids(trainer.model)
    shares, cos = {t: [] for t in judged}, {p: [] for p in itertools.combinations(judged, 2)}
    for xb, yb in _batches(x, y, batches, dev):
        params, grads, shared = head_gradients(trainer, xb, yb)
        shared = [i for i in shared if id(params[i]) not in head_ids]
        full = {t: _shared_vector(grads[t], grads, shared, dev) for t in judged}
        norms = {t: float(v.norm()) for t, v in full.items()}
        tot = sum(norms.values())
        for t in judged:
            shares[t].append(norms[t] / tot if tot > 0 else 1.0 / len(judged))
        for a, b in cos:
            den = norms[a] * norms[b]
            cos[(a, b)].append(float(full[a] @ full[b]) / den if den > 0 else 0.0)
    share = {t: float(np.mean(v)) for t, v in shares.items()}
    pair = {f"{a}|{b}": float(np.mean(v)) for (a, b), v in cos.items()}
    starved = sorted(t for t, v in share.items() if v < A5_MIN_SHARE)
    a5 = "fail" if starved or max(share.values(), default=0.0) > A5_MAX_SHARE else "pass"
    low = min(pair.values(), default=0.0)
    a6 = "fail" if low < A6_FAIL else "warn" if low < 0.0 else "pass"
    return {"A5": {"verdict": a5, "value": share, "starved": starved,
                   "rule": f"every judged head >= {A5_MIN_SHARE}, the largest <= {A5_MAX_SHARE}"},
            "A6": {"verdict": a6, "value": pair, "min_pair": low,
                   "rule": f"mean pairwise cosine >= {A6_FAIL} for every pair (between it and 0 = warn); a fail "
                           f"blocks only on every seed"}}


def _shared_vector(g: tuple, grads: dict, shared: list[int], dev) -> torch.Tensor:
    """One head's gradient on the shared parameters as one float64 vector (zeros where the head has none)."""
    parts = []
    for i in shared:
        size = next(int(h[i].numel()) for h in grads.values() if h[i] is not None)
        parts.append(torch.zeros(size, dtype=torch.float64, device=dev) if g[i] is None
                     else g[i].detach().double().reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64, device=dev)


def _activation_points(model) -> list[str]:
    points = getattr(model, "activation_points", None)
    if points is not None:
        return list(points)
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.out_features >= 8]


@torch.no_grad()
def _a7(trainer, x, y, batches, dev) -> dict:
    model = trainer.model
    modules = dict(model.named_modules())
    points = _activation_points(model)
    unknown = [p for p in points if p not in modules]
    if unknown:
        raise KeyError(f"activation_points not in the model: {unknown}")
    active: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _inp, out):
            a = (out.detach() > 0).reshape(-1, out.shape[-1]).any(dim=0)
            active[name] = a if name not in active else active[name] | a
        return fn

    handles = [modules[p].register_forward_hook(hook(p)) for p in points]
    was = model.training
    model.eval()
    try:
        for xb, _ in _batches(x, y, batches, dev):
            model(xb)
    finally:
        for h in handles:
            h.remove()
        model.train(was)
    dead = {p: float(1.0 - active[p].float().mean()) for p in points if p in active}
    worst_share = max(dead.values(), default=0.0)
    verdict = "fail" if worst_share > A7_FAIL else "warn" if worst_share > A7_WARN else "pass"
    return {"verdict": verdict, "value": worst_share, "per_layer": dead,
            "rule": f"share of never-active units per layer <= {A7_WARN} ({A7_FAIL} = fail)"}


def grad_report(trainer, x, y, batches, *, allow_dead=()) -> dict:
    """Gate A (A1-A7) of one measurement on ``batches`` (index arrays into ``x`` / ``y``); ``allow_dead``: parameter
    names a card lists as legitimately gradient-free (A2). The model's state is unchanged afterwards."""
    if trainer.optimizer is None:
        raise RuntimeError("gate A needs the planned optimizer (the trainer has none)")
    if not batches:
        raise ValueError("gate A needs at least one batch")
    dev = _device(trainer)
    checks, layers = _a1_a3(trainer, x, y, batches, dev, set(allow_dead))
    checks["A4"] = _a4(trainer, x, y, batches, dev, skip=set(checks["A2"]["dead"]))
    checks.update(_a5_a6(trainer, x, y, batches, dev))
    checks["A7"] = _a7(trainer, x, y, batches, dev)
    return {"checks": checks, "layers": layers, "n_batches": len(batches),
            "verdict": worst(c["verdict"] for c in checks.values())}
