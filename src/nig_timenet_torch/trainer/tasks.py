"""PURPOSE: TASK_REGISTRY: the task losses (evidential NIG, regression, classification, side payoff, pinball) with, per
    task, the registered ``config_keys`` (the options a target config may carry) and ``default_options``.
TAGS: task loss, task registry, evidential, classification, regression, config_keys, default_options, calibration,
    side payoff, pinball, quantile
PITFALLS: a task fn reads ``target_cfg[key]`` with NO fallback: it needs the target config RESOLVED by
    ``trainer.targets.resolve_targets`` (``with_defaults`` fills a single one); a new option needs a ``config_keys``
    entry (else ``validate_trainer_config`` refuses it) and a ``default_options`` value (a pinned default, rule B1).
    side_payoff / pinball read a LOG-return label (codec logret_close): on another codec their targets mean nothing.

TASK_REGISTRY and the task losses, ported from crypto-signal-prediction trainer/tasks.py @ aeafc564
(BUILD_PLAN.md P17): evidential NIG, regression, binary and multiclass classification (P42), confidence calibration.

Every task fn is ``(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None) -> (loss, stats)``; stats
always carry the registered ``stat_keys``. Losses are float32. Not ported (off in the oracle's config; add when a
step's test needs them): ``huber_weight``, ``normalize_reg``, ``alpha_floor_straight_through`` — each raises
NotImplementedError when requested instead of being silently ignored. ``beta_nll`` is ported (FX1, TF semantics:
NLL x stop_grad(w_St^2)^beta_nll) with the torch-only ``beta_nll_normalize`` (weights / their batch mean).
"""

import inspect
import math

import torch
import torch.nn.functional as F

from nig_timenet_torch.core.costs import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["TASK_REGISTRY", "register_task_type", "with_defaults", "evidential_task_loss", "regression_task_loss",
           "classification_task_loss", "nig_regularizer", "nig_w_st", "compute_confidence_calibration_loss",
           "side_payoff_task_loss", "quantile_task_loss", "side_payoffs"]

KERAS_EPSILON = 1e-7  # keras.backend.epsilon(): probability clip of binary_crossentropy


def _check_task(name, fn, meta):
    problems = []
    if not callable(fn):
        return [f"task fn must be callable, got {type(fn).__name__}"]
    if not all(isinstance(k, str) for k in meta["stat_keys"]):
        problems.append("stat_keys must be strings")
    if not all(isinstance(k, str) for k in meta["config_keys"]):
        problems.append("config_keys must be strings")
    stray = sorted(set(meta["default_options"]) - set(meta["config_keys"]))
    if stray:
        problems.append(f"default_options {stray} are not in config_keys {list(meta['config_keys'])}")
    if meta["supports_sample_weight"] and "sample_weight" not in inspect.signature(fn).parameters:
        problems.append("supports_sample_weight=True but fn has no `sample_weight` parameter")
    return problems


TASK_REGISTRY = Registry("task", Contract("task", required_meta=("stat_keys", "supports_sample_weight", "config_keys",
                                                                    "default_options"),
                                          validator=_check_task),
                         view=lambda e: {"fn": e.obj, **e.meta})


def register_task_type(name: str, fn, stat_keys, supports_sample_weight: bool = False, config_keys=(),
                       default_options=None):
    """``config_keys``: the per-target options the task reads (``trainer.targets.validate_trainer_config`` refuses any
    other key); ``default_options``: the value of each option the target config leaves out (a subset of config_keys)."""
    TASK_REGISTRY.register(name, fn, version=1, stat_keys=list(stat_keys),
                           supports_sample_weight=supports_sample_weight, config_keys=tuple(config_keys),
                           default_options=dict(default_options or {}))


def with_defaults(task_type: str, target_cfg: dict) -> dict:
    """``target_cfg`` with every option it leaves out set to the task's registered ``default_options`` value."""
    return {**TASK_REGISTRY.entry(task_type).meta["default_options"], **target_cfg}


def _flat(x) -> torch.Tensor:
    return torch.as_tensor(x).to(torch.float32).reshape(-1)


def _wmean(x, w=None) -> torch.Tensor:
    """sum(w x) / sum(w) (0, not NaN, when all weights are 0); plain mean when w is None."""
    x = x.to(torch.float32)
    if w is None:
        return x.mean()
    w = w.to(torch.float32)
    den = w.sum()
    num = (x * w).sum()
    return torch.where(den == 0, torch.zeros_like(num), num / torch.where(den == 0, torch.ones_like(den), den))


def _wstd(x, w=None) -> torch.Tensor:
    if w is None:
        return x.std(unbiased=False)  # tf.math.reduce_std: population std
    m = _wmean(x, w)
    return torch.sqrt(_wmean((x - m).square(), w) + 1e-12)


def _unported(target_cfg, keys):
    on = [k for k in keys if target_cfg[k]]
    if on:
        raise NotImplementedError(f"target options {on} are not ported yet (BUILD_PLAN.md P17 'Do NOT')")


# ----------------------------------------------------------------------------------------------- evidential (NIG)

def compute_confidence_calibration_loss(y_true, y_pred, confidence, sample_weight=None) -> torch.Tensor:
    y_true, y_pred, confidence = _flat(y_true), _flat(y_pred), _flat(confidence)
    w = None if sample_weight is None else _flat(sample_weight)
    abs_error = (y_true - y_pred.detach()).abs()  # never pushes mu
    max_expected_error = (_wmean(abs_error, w) + 1.5 * _wstd(abs_error, w)).detach()
    target_confidence = 1.0 - torch.clamp(abs_error / (max_expected_error + 1e-8), 0.0, 1.0)
    diff = confidence - target_confidence
    focal = _wmean(diff.abs().pow(2.0) * diff.square(), w)
    diversity_penalty = torch.clamp_min(0.15 - _wstd(confidence, w), 0.0)
    saturation_penalty = _wmean(torch.clamp_min(confidence - 0.95, 0.0) + torch.clamp_min(0.05 - confidence, 0.0), w)
    return focal + 0.5 * diversity_penalty + 0.3 * saturation_penalty


def nig_regularizer(error, nu, alpha, beta, normalize=False, stop_grad=True) -> torch.Tensor:
    """|y - gamma| * (2 nu + alpha) (Amini et al. 2020). normalize=True (Meinert) is not ported yet."""
    if normalize:
        raise NotImplementedError("normalize_reg is not ported yet (BUILD_PLAN.md P17 'Do NOT')")
    return error.abs() * (2.0 * nu + alpha)


def nig_w_st(nu, alpha, beta) -> torch.Tensor:
    """Width of the NIG's Student-t predictive: sqrt(beta (1 + nu) / (nu alpha)) (Meinert et al. 2023, eq. 9)."""
    return torch.sqrt(beta * (1.0 + nu) / (nu * alpha))


def _beta_nll_weighted(nig_base, nu, alpha, beta, target_cfg) -> torch.Tensor:
    """beta-NLL (Seitzer et al. 2022, TF ``beta_nll``): per-sample NLL x stop_grad(w_St^2)^beta_nll, so calm rows no
    longer dominate the mu gradient (it scales as 1 / w_St^2); ``beta_nll_normalize`` divides the weights by their
    batch mean (the loss keeps its scale next to the other tasks; TF does not). beta_nll = 0 returns ``nig_base``."""
    b = float(target_cfg["beta_nll"])
    if b == 0.0:
        return nig_base
    weight = nig_w_st(nu, alpha, beta).square().detach().pow(b)
    if target_cfg["beta_nll_normalize"]:
        weight = weight / weight.mean()
    return nig_base * weight


def evidential_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    _unported(target_cfg, ("normalize_reg", "huber_weight", "alpha_floor_straight_through"))
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    y_true = _flat(y_true)
    gamma, nu, alpha, beta = (_flat(outputs[keys[k]]) for k in ("mu", "nu", "alpha", "beta"))

    error = y_true - gamma
    omega = torch.clamp_min(2.0 * beta * (1.0 + nu), 1e-6)
    nu = torch.clamp_min(nu, 1e-6)
    alpha = torch.clamp_min(alpha, 1.0)
    nig_base = (0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega)
                + (alpha + 0.5) * torch.log(nu * error.square() + omega)
                + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    nig_pen = nig_regularizer(error, nu, alpha, beta)
    lambda_reg = float(scheduled_vars[target_cfg["lambda_reg_var"]])
    total = _wmean(_beta_nll_weighted(nig_base, nu, alpha, beta, target_cfg)
                   + torch.tensor(lambda_reg, dtype=torch.float32) * nig_pen, w)

    aleatoric = beta / (alpha - 1.0 + 1e-6)
    epistemic = beta / (nu * (alpha - 1.0) + 1e-6)
    stats = {"nig_base": _wmean(nig_base, w), "nig_pen": _wmean(nig_pen, w), "mae": _wmean(error.abs(), w),
             "aleatoric": _wmean(aleatoric, w), "epistemic": _wmean(epistemic, w),
             "confidence": _wmean(1.0 / (1.0 + aleatoric + epistemic), w), "calib_loss": torch.tensor(0.0)}
    if target_cfg["use_calibration_loss"] and "confidence" in keys:
        calib = compute_confidence_calibration_loss(y_true, gamma, outputs[keys["confidence"]], sample_weight=w)
        lambda_calib = float(scheduled_vars[target_cfg["lambda_calib_var"]])
        total = total + calib * torch.tensor(lambda_calib, dtype=torch.float32)
        stats["calib_loss"] = calib
    return total, stats


register_task_type(
    "evidential", evidential_task_loss,
    ["nig_base", "nig_pen", "mae", "aleatoric", "epistemic", "confidence", "calib_loss"], supports_sample_weight=True,
    config_keys=("use_calibration_loss", "lambda_reg_var", "lambda_calib_var", "normalize_reg",
                 "beta_nll", "beta_nll_normalize", "huber_weight", "alpha_floor_straight_through"),
    default_options={"use_calibration_loss": False, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib",
                     "normalize_reg": False, "beta_nll": 0.0, "beta_nll_normalize": False, "huber_weight": 0.0,
                     "alpha_floor_straight_through": False})


# ------------------------------------------------------------------------------------------------------ regression

def regression_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    error = _flat(y_true) - _flat(outputs[keys["pred"]])
    kind = target_cfg["loss_fn"]
    if kind == "mae":
        per = error.abs()
    elif kind == "huber":
        delta = float(target_cfg["huber_delta"])
        a = error.abs()
        q = torch.clamp_max(a, delta)
        per = 0.5 * q.square() + delta * (a - q)
    else:
        per = error.square()
    return _wmean(per, w), {"mae": _wmean(error.abs(), w)}


register_task_type("regression", regression_task_loss, ["mae"], supports_sample_weight=True,
                   config_keys=("loss_fn", "huber_delta"),
                   default_options={"loss_fn": "mse", "huber_delta": 1.0})


# -------------------------------------------------------------------------------------------------- classification

def classification_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    """Binary: Keras binary_crossentropy on PROBABILITIES (the head ends in sigmoid; the oracle's reshape drops
    Keras' cached logits, so it takes the clip-to-[eps, 1-eps] path, eps = 1e-7). Accuracy uses p >= 0.5.
    Multiclass (``binary=False``): the head ends in softmax, so Keras' from_logits=False path — probabilities
    renormalised, clipped to [eps, 1-eps], cross-entropy against the integer label (label_smoothing: one-hot *
    (1 - ls) + ls / K). Accuracy uses argmax."""
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    ls = float(target_cfg["label_smoothing"])
    if not target_cfg["binary"]:
        p = outputs[keys["logits"]]
        y = y_true.reshape(-1).to(torch.int64)
        q = torch.clamp(p / p.sum(dim=-1, keepdim=True), KERAS_EPSILON, 1.0 - KERAS_EPSILON)
        if ls > 0:
            k = p.shape[-1]
            soft = F.one_hot(y, k).to(p.dtype) * (1.0 - ls) + ls / k
            per = -(soft * torch.log(q)).sum(dim=-1)
        else:
            per = -torch.log_softmax(torch.log(q), dim=-1).gather(1, y[:, None])[:, 0]
        acc = _wmean((p.argmax(dim=-1) == y).to(torch.float32), w)
        return _wmean(per, w), {"accuracy": acc}
    y = _flat(y_true)
    p = _flat(outputs[keys["logits"]])
    y_loss = y * (1.0 - ls) + 0.5 * ls if ls > 0 else y
    pc = torch.clamp(p, KERAS_EPSILON, 1.0 - KERAS_EPSILON)
    per = -(y_loss * torch.log(pc) + (1 - y_loss) * torch.log(1 - pc))
    acc = _wmean(((p >= 0.5).to(torch.float32) == y).to(torch.float32), w)
    return _wmean(per, w), {"accuracy": acc}


register_task_type("classification", classification_task_loss, ["accuracy"], supports_sample_weight=True,
                   config_keys=("binary", "label_smoothing"),
                   default_options={"binary": True, "label_smoothing": 0.0})


# ------------------------------------------------------------------- auxiliary heads (owner 2026-11-19, LS1)

def side_payoffs(y_log: torch.Tensor, cost_per_side: float) -> tuple[torch.Tensor, torch.Tensor]:
    """What a long and a short of a fixed notional are paid over the label's horizon, after a round trip of cost:
    long ``e^y - 1 - 2c``, short ``1 - e^y - 2c`` (y = log return). The short is NOT the long's mirror in log units:
    a rise of +l costs it e^l - 1 > l (docs/results/short-side-001.md)."""
    move = torch.expm1(y_log)
    cost = 2.0 * float(cost_per_side)
    return move - cost, -move - cost


def _squared_or_huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    """0.5 e^2 when ``delta`` is 0, else the Huber loss with that delta (in the error's units)."""
    if delta <= 0:
        return 0.5 * error.square()
    a = error.abs()
    q = torch.clamp_max(a, delta)
    return 0.5 * q.square() + delta * (a - q)


def side_payoff_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    """Each side's head regresses that side's payoff (``side_payoffs``): the squared error (or Huber,
    ``huber_delta`` > 0) of (prediction - payoff) / ``scale``, the short term weighted by ``short_weight``. The
    predictions are in price-return units, so ``pred > 0`` reads "this side pays after cost".
    ``variance_weight`` v > 0: each row's term is weighted by (its NIG predictive variance)^-v, detached and scaled to
    a batch mean of 1 (``nig_keys`` names the nu / alpha / beta outputs of the target's NIG head). The weights depend
    on the inputs only, so the minimiser stays the conditional mean payoff; v = 1 is weighted least squares, which
    stops the volatile rows' noise from drowning the signal of the calm ones (LS1 E1 reading)."""
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    pay_long, pay_short = side_payoffs(_flat(y_true), target_cfg["cost_per_side"])
    scale, delta = float(target_cfg["scale"]), float(target_cfg["huber_delta"])
    err_long = _flat(outputs[keys["long"]]) - pay_long
    err_short = _flat(outputs[keys["short"]]) - pay_short
    per = (_squared_or_huber(err_long / scale, delta)
           + float(target_cfg["short_weight"]) * _squared_or_huber(err_short / scale, delta))
    v = float(target_cfg["variance_weight"])
    if v > 0:
        nig = target_cfg["nig_keys"]
        nu, alpha, beta = (_flat(outputs[nig[k]]).detach() for k in ("nu", "alpha", "beta"))
        var = torch.clamp_min(beta * (1.0 + nu) / (nu * torch.clamp_min(alpha - 1.0, 1e-6)), 1e-12)
        rw = var.pow(-v)
        per = per * (rw / rw.mean())
    return _wmean(per, w), {"mae_long": _wmean(err_long.abs(), w), "mae_short": _wmean(err_short.abs(), w)}


register_task_type("side_payoff", side_payoff_task_loss, ["mae_long", "mae_short"], supports_sample_weight=True,
                   config_keys=("cost_per_side", "scale", "huber_delta", "short_weight", "variance_weight",
                                "nig_keys"),
                   default_options={"cost_per_side": DEFAULT_COST_PER_SIDE, "scale": 0.02, "huber_delta": 0.0,
                                    "short_weight": 1.0, "variance_weight": 0.0, "nig_keys": {}})


def quantile_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    """The pinball (quantile) loss of the head's value as the ``quantile`` of the label, divided by ``scale``; stat
    ``coverage`` = share of rows whose label is at or under the prediction (it tends to ``quantile``)."""
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    tau = float(target_cfg["quantile"])
    q = _flat(outputs[keys["q"]])
    error = _flat(y_true) - q
    per = torch.maximum(tau * error, (tau - 1.0) * error) / float(target_cfg["scale"])
    return _wmean(per, w), {"coverage": _wmean((error <= 0).to(torch.float32), w)}


register_task_type("pinball", quantile_task_loss, ["coverage"], supports_sample_weight=True,
                   config_keys=("quantile", "scale"), default_options={"quantile": 0.9, "scale": 0.02})
