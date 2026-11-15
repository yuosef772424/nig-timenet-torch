"""PURPOSE: LABEL_SCALES, the registered NIG head scale component (B-052, FX2): ``fixed`` (the nig_scale override or
    none) or ``train_std`` (fit on the TRAINING labels: s = mean std of the NIG labels, stored as a model buffer).
TAGS: label scale, nig scale, beta floor, b-052, learned state, fit on train, registry, model buffer
PITFALLS: ``train_std`` is a learned-state transform (rule B4): ``fit`` sees the training split only, the state is the
    model's ``label_scale`` buffers (state_dict, checkpoint, best weights), and every consumer (training, evaluation,
    prediction) reads it through the heads; a restored model gets s from its weights, never refits. ``fit_label_scale``
    on a model built with ``fixed`` does nothing.

With scale s the NIG heads work in label units / s (``nig_layers.NIGHead``): mu = s raw_mu and beta = s^2 (softplus +
beta_min), so the beta floor is ``beta_min`` x s^2, i.e. proportional to the training label variance instead of the
unit-ful 0.01 that is ~25x the variance of a label-scale-1 artifact (B-052). ``fixed`` keeps the FX1 ``nig_scale``
override (None = the TF formula); ``train_std`` sets s = mean over the NIG targets of the training label std (one s
for every head, so the ordered means keep one unit).
"""

import numpy as np
import torch

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["LABEL_SCALES", "fitted", "fit_label_scale", "label_scale_state"]

LABEL_SCALES = Registry("label_scale", Contract("label_scale", required_meta=("fitted", "fit")))


def _train_std(labels: dict[str, np.ndarray]) -> dict:
    std = {t: float(np.std(np.asarray(v, dtype=np.float64))) for t, v in labels.items()}
    if not std or min(std.values()) <= 0.0:
        raise ValueError(f"train_std label scale needs labels with a positive std, got {std}")
    return {"scale": float(np.mean(list(std.values()))), "std": std}


def _fixed(labels: dict[str, np.ndarray]) -> None:
    """Nothing to learn: the model keeps its built ``nig_scale``."""


LABEL_SCALES.register("fixed", _fixed, version=1, fitted=False, fit=_fixed)
LABEL_SCALES.register("train_std", _train_std, version=1, fitted=True, fit=_train_std)


def fitted(name: str) -> bool:
    """True when the registered scale ``name`` is learned from the training labels (unknown names raise)."""
    return bool(LABEL_SCALES.entry(name).meta["fitted"])


def fit_label_scale(model: torch.nn.Module, y_train: dict) -> dict | None:
    """Fit the model's label scale on the TRAINING labels ``y_train`` ({label key: values}) and write it into the
    model's ``label_scale`` buffers; None (nothing done) for a model whose scale is not fitted."""
    name = getattr(model, "label_scale_name", "fixed")
    if not fitted(name):
        return None
    labels = {t: np.asarray(torch.as_tensor(y_train[spec.label_key]).cpu()) for t, spec in model.head_specs.items()}
    state = LABEL_SCALES.entry(name).meta["fit"](labels)
    with torch.no_grad():
        for _, buf in _buffers(model):
            buf.fill_(state["scale"])
    return state


def label_scale_state(model: torch.nn.Module) -> float | None:
    """The scale the model's heads use now (the fitted buffer), or None for a model without one."""
    bufs = [float(b) for _, b in _buffers(model)]
    return bufs[0] if bufs else None


def _buffers(model: torch.nn.Module):
    return [(n, b) for n, b in model.named_buffers() if n.split(".")[-1] == "label_scale"]
