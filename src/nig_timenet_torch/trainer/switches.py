"""PURPOSE: Registered training switches of TR1 / D-021: weight initialisation (INITS: torch, keras), weight-decay
    parameter groups (WEIGHT_DECAY_GROUPS: all_params, exclude_norm_bias) and the EMA copy of the weights of fit.
TAGS: training switches, init registry, keras init, glorot, orthogonal, weight decay groups, ema weights, tr1
PITFALLS: the defaults ("torch", "all_params", no EMA) reproduce the trainer before D-021 exactly; ``keras``
    re-initialises a BUILT model in place (call it after seeding); the decay groups need the parameter NAMES, so they
    are built from the model, and a parameter the model does not own (the Kendall log-variances) is never decayed by
    ``exclude_norm_bias`` but is decayed by ``all_params`` (as before).

Why these exist (docs/bottlenecks.csv): B-007 (torch default init, TF used glorot / orthogonal), B-018 (no gradient
clipping, no weight-decay exclusions, no EMA in the torch trainer), B-061 (dead units, init candidate).
"""

import torch
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["INITS", "WEIGHT_DECAY_GROUPS", "apply_init", "param_groups", "EmaWeights"]

INITS = Registry("init", Contract("init", required_meta=("fn",)))
WEIGHT_DECAY_GROUPS = Registry("weight_decay_groups", Contract("weight_decay_groups", required_meta=("fn",)))


def _torch_init(model: nn.Module) -> None:
    """Keep the weights the modules were built with (torch defaults)."""


def _keras_init(model: nn.Module) -> None:
    """Keras defaults: glorot uniform kernels (Linear, Conv), glorot input / orthogonal recurrent kernels (GRU, LSTM),
    zero biases; norm layers and the package's own plain parameters (positions, scales, queries) are left as built."""
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, p in mod.named_parameters():
                if name.startswith("weight_hh"):
                    nn.init.orthogonal_(p)
                elif name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)
                elif name.startswith("bias"):
                    nn.init.zeros_(p)


INITS.register("torch", _torch_init, version=1, fn=_torch_init)
INITS.register("keras", _keras_init, version=1, fn=_keras_init)


def apply_init(name: str, model: nn.Module) -> None:
    """Re-initialise ``model`` in place with the registered init ``name`` (unknown names raise)."""
    INITS.entry(name).meta["fn"](model)


def _no_decay(name: str, p: torch.Tensor) -> bool:
    return p.dim() < 2 or name.endswith("bias") or "norm" in name.lower()


def _all_groups(model: nn.Module, params: list, weight_decay: float) -> list[dict]:
    return [{"params": list(params), "weight_decay": weight_decay}]


def _exclude_norm_bias(model: nn.Module, params: list, weight_decay: float) -> list[dict]:
    names = {id(p): n for n, p in model.named_parameters()}
    decay = [p for p in params if id(p) in names and not _no_decay(names[id(p)], p)]
    keep = [p for p in params if id(p) not in {id(q) for q in decay}]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": keep, "weight_decay": 0.0}]


WEIGHT_DECAY_GROUPS.register("all_params", _all_groups, version=1, fn=_all_groups)
WEIGHT_DECAY_GROUPS.register("exclude_norm_bias", _exclude_norm_bias, version=1, fn=_exclude_norm_bias)


def param_groups(name: str, model: nn.Module, params, weight_decay: float) -> list[dict]:
    """Optimizer parameter groups of ``params`` (the trainer's parameters, model first) for the registered grouping
    ``name``; ``all_params`` is one group, i.e. exactly what passing the flat list did before D-021."""
    return WEIGHT_DECAY_GROUPS.entry(name).meta["fn"](model, list(params), weight_decay)


class EmaWeights:
    """An exponential moving average of every floating entry of ``model.state_dict()`` (parameters and float
    buffers; integer buffers are copied), updated after each optimizer step: ``ema = d_n * ema + (1 - d_n) * w`` with
    the warm-up ``d_n = min(decay, (1 + n) / (10 + n))`` of ``tf.train.ExponentialMovingAverage(num_updates=n)``.
    Without the warm-up a decay of 0.999 over a few hundred steps leaves the EMA mostly at the INITIAL weights (E1
    screening, D-021: mu spread 30 x the oracle's)."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def current_decay(self) -> float:
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)
        self.updates += 1

    def state_dict(self) -> dict:
        return {"updates": self.updates, "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def weights(self) -> dict:
        """The averaged weights as a model state_dict (copies)."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        self.updates = int(state["updates"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
