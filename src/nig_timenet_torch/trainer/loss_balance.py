"""PURPOSE: Registered per-head loss balancers (LOSS_BALANCERS: none, equal_grad_norm): multipliers on the per-target
    losses so that every head's gradient on the shared representation has the same norm (B-060, gate A5; FX2).
TAGS: loss balance, gradient normalisation, gradnorm, multi-task, head gradient share, gate a5, b-060, registry
PITFALLS: the multipliers are applied to the per-target losses BEFORE Kendall; Kendall would learn them away (its
    log-var settles at log(w L)), so a balancer other than ``none`` refuses Kendall-weighted tasks; the norms are
    measured on the output of the model's ``balance_point`` module (the shared representation h, not the trunk
    parameters: a cheap backward through the heads only); the multipliers are detached constants of the step.

``none`` (default) keeps every multiplier at 1.0 and touches nothing: training is bit-identical to the trainer before
FX2. ``equal_grad_norm``: after every training forward, ``n_t`` = ||d (loss_weight_t L_t) / d h|| per head, an EMA
``m_t`` of it (decay ``EMA_DECAY``), and the multipliers ``w_t = mean_s(m_s) / m_t`` used from the NEXT step on
(sum_t w_t m_t = sum_t m_t: the total gradient on h keeps its size, only the mixture changes). The state (EMA and
step count) goes into the checkpoint.
"""

import torch

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["LOSS_BALANCERS", "make_balancer", "NoBalance", "EqualGradNorm", "EMA_DECAY"]

LOSS_BALANCERS = Registry("loss_balance", Contract("loss_balance", required_meta=("cls", "needs_shared")))
EMA_DECAY = 0.9


class NoBalance:
    """Every multiplier 1.0; nothing is measured (the default)."""
    needs_shared = False

    def __init__(self, targets):
        self.targets = list(targets)

    def multipliers(self) -> dict[str, float]:
        return dict.fromkeys(self.targets, 1.0)

    def update(self, per_target: dict, shared: torch.Tensor | None) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


class EqualGradNorm(NoBalance):
    """Multipliers that equalise the EMA of each head's gradient norm on the shared representation."""
    needs_shared = True

    def __init__(self, targets):
        super().__init__(targets)
        self.ema: dict[str, float] = {}
        self.steps = 0

    def multipliers(self) -> dict[str, float]:
        if not self.ema or min(self.ema.values()) <= 0.0:
            return dict.fromkeys(self.targets, 1.0)
        mean = sum(self.ema.values()) / len(self.ema)
        return {t: mean / self.ema[t] for t in self.targets}

    def update(self, per_target: dict, shared: torch.Tensor | None) -> None:
        """``per_target``: the losses as the step used them (multiplier included); ``shared``: h of that forward."""
        if shared is None:
            raise RuntimeError("equal_grad_norm needs the shared representation (model.balance_point)")
        mult = self.multipliers()
        norms = {}
        for t in self.targets:
            loss = per_target[t]
            g = (torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)[0]
                 if loss.requires_grad else None)
            raw = 0.0 if g is None else float(g.detach().double().norm())
            norms[t] = raw / mult[t]
        for t, v in norms.items():
            self.ema[t] = v if t not in self.ema else EMA_DECAY * self.ema[t] + (1.0 - EMA_DECAY) * v
        self.steps += 1

    def state_dict(self) -> dict:
        return {"ema": dict(self.ema), "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.ema, self.steps = dict(state["ema"]), int(state["steps"])


LOSS_BALANCERS.register("none", NoBalance, version=1, cls=NoBalance, needs_shared=False)
LOSS_BALANCERS.register("equal_grad_norm", EqualGradNorm, version=1, cls=EqualGradNorm, needs_shared=True)


def make_balancer(name: str, targets, kendall_targets) -> NoBalance:
    """The registered balancer ``name`` over ``targets`` (unknown names raise); a balancing one refuses Kendall."""
    entry = LOSS_BALANCERS.entry(name)
    if entry.meta["needs_shared"] and list(kendall_targets):
        raise ValueError(f"loss balance {name!r} replaces Kendall weighting: set use_uncertainty_weighting false "
                         f"(Kendall tasks {list(kendall_targets)})")
    return entry.meta["cls"](targets)
