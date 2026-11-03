"""T4 trainer trace: what the loss weighting and the schedules did, one row of numbers per epoch (log-only).

``epoch_trace`` is called by ``Trainer.fit`` after an epoch when ``FitSettings.trace`` is on and its result is added to
that epoch's history row, so it is checkpointed, resumed and read by ``workflow.health_report`` like any other column.
It never changes training: the forward/backward passes below run in eval() mode on ONE fixed batch (the first rows of
the training set), draw no random numbers, and make no optimizer step.

Keys (all floats): ``trace_sched_<name>`` the scheduled loss variables of the epoch; ``trace_kendall_weight_<head>``
(normalised effective Kendall weight) and ``trace_kendall_logvar_<head>`` for the Kendall-weighted heads;
``trace_gradshare_<head>`` the head's share of the gradient norm on the SHARED parameters (those that receive a
gradient from at least two heads), measured on the objective as optimised (loss_weight and, for Kendall heads,
0.5 exp(-log_var) applied). The shares sum to 1; a head with ~0 share is not shaping the trunk.
"""

import math

import torch

__all__ = ["epoch_trace", "head_gradient_share"]


def head_gradient_share(trainer, x, y) -> dict[str, float]:
    """{target: share of the shared-parameter gradient norm}; one backward-free ``autograd.grad`` per head."""
    model, was_training = trainer.model, trainer.model.training
    model.eval()
    try:
        with torch.enable_grad():
            _, _, per_target, _, _ = trainer.compute_losses(x, y)
            params = [p for p in model.parameters() if p.requires_grad]
            mult = {t: 1.0 for t in per_target}
            if trainer.uncertainty_layer is not None:
                for t in trainer.kendall_task_names:
                    mult[t] = 0.5 * math.exp(-float(trainer.uncertainty_layer.log_var(t).detach()))
            grads = {t: torch.autograd.grad(loss * mult[t], params, retain_graph=True, allow_unused=True)
                     for t, loss in per_target.items()}
    finally:
        model.train(was_training)
    shared = [i for i in range(len(params)) if sum(g[i] is not None for g in grads.values()) >= 2]
    norms = {t: math.sqrt(sum(float(g[i].detach().double().pow(2).sum()) for i in shared if g[i] is not None))
             for t, g in grads.items()}
    total = sum(norms.values())
    if not shared or total == 0.0:
        return {t: 1.0 / len(norms) for t in norms}
    return {t: v / total for t, v in norms.items()}


def epoch_trace(trainer, x, y) -> dict[str, float]:
    """The trace row of one finished epoch (see the module docstring)."""
    out = {f"trace_sched_{n}": float(v) for n, v in trainer.scheduled_vars.items()}
    layer = trainer.uncertainty_layer
    if layer is not None:
        out.update({f"trace_kendall_weight_{t}": w for t, w in layer.get_effective_weights().items()})
        out.update({f"trace_kendall_logvar_{t}": float(layer.log_var(t).detach()) for t in layer.task_names})
    out.update({f"trace_gradshare_{t}": s for t, s in head_gradient_share(trainer, x, y).items()})
    return out
