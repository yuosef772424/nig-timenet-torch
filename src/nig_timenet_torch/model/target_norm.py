"""PURPOSE: TARGET_NORMS, the registered target normalisation of the NIG heads (rule B4): ``none`` (labels as they
    are) or ``logret_std`` (per head z = (y - mean_h) / std_h with TRAIN statistics, TF LogretStd's target half).
TAGS: target normalisation, label standardisation, logret_std, per head mean std, learned state, fit on train, inverse,
    registry, model buffer
PITFALLS: the state (mean / std per NIG head) lives in the model's ``target_norm`` buffers (state_dict, checkpoint,
    best weights): ``fit`` sees the TRAINING labels only, a restored model gets it from its weights, never refits.
    The heads, the ordered means and the NIG floors work in the normalised space; the model's outputs are mapped
    back by ``inverse`` (mu = mean + std * mu_z, beta = std^2 * beta_z, nu / alpha unchanged: the exact affine map of
    a NIG), so every consumer keeps label units and nothing is decoded in a second place. With per-head statistics
    and the default ordering, the ordered means hold in z space: each head is centred on its own train mean there
    (high about +0.9 sd of the close's unit, low about -0.9), so "mu_low_z <= mu_close_z <= mu_high_z" is NOT the
    price order and has no meaning: a confident close mean drags the high / low means with it (measured: high / low
    rank IC 0.36 / 0.44 -> 0.18 / 0.33 in the panel walk-forward, B-084, docs/results/panel-rolling-std-001.md), and
    mu_low <= mu_close <= mu_high in label units is not guaranteed. The model override ``order_in_label_units``
    applies the order on prices instead (model/builder.py). The fitted label scale (``model.label_scale``) is then
    fitted on the NORMALISED labels (``normalized_labels``).

Why (owner, 2026-11-17; docs/audit/2026-11-14-model-first/NOTES.md N-01): TF's ``logret_std`` standardised the inputs
AND the targets of each head with train statistics; this port standardises the inputs (``model.scalers``
InputScaler) and scales every head by ONE number (``model.label_scale``), without the per-head centring: the high /
low labels have means of about +-0.02 and their own spread, so the heads started off-centre.
"""

import numpy as np
import torch
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["TARGET_NORMS", "TargetNorm", "fit_target_norm", "normalized_labels", "target_norm_state"]

TARGET_NORMS = Registry("target_norm", Contract("target_norm", required_meta=("fitted",)))


class TargetNorm(nn.Module):
    """Per-head ``mean`` / ``std`` buffers over ``targets`` (NaN until ``fit`` or a state_dict sets them)."""

    def __init__(self, targets):
        super().__init__()
        self.targets = tuple(targets)
        self.register_buffer("mean", torch.full((len(self.targets),), float("nan")))
        self.register_buffer("std", torch.full((len(self.targets),), float("nan")))

    def _i(self, target: str) -> int:
        if target not in self.targets:
            raise KeyError(f"target_norm has no head {target!r} (heads: {self.targets})")
        return self.targets.index(target)

    @torch.no_grad()
    def fit(self, labels: dict) -> dict:
        """Set mean / std from the TRAINING labels ``{target: values}`` (float64); returns the state."""
        missing = [t for t in self.targets if t not in labels]
        if missing:
            raise KeyError(f"target_norm.fit needs the labels of every head, missing {missing}")
        state = {}
        for i, t in enumerate(self.targets):
            v = np.asarray(torch.as_tensor(labels[t]).detach().cpu(), dtype=np.float64).reshape(-1)
            v = v[np.isfinite(v)]
            if v.size < 2 or float(v.std()) <= 0.0:
                raise ValueError(f"target_norm.fit: the {t!r} labels need >= 2 finite values with a positive std")
            self.mean[i], self.std[i] = float(v.mean()), float(v.std())
            state[t] = {"mean": float(v.mean()), "std": float(v.std())}
        return state

    def _stats(self, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        i = self._i(target)
        m, s = self.mean[i], self.std[i]
        if not bool(torch.isfinite(s)):
            raise RuntimeError("target_norm is not fitted: call model.target_norm.fit_target_norm(model, y_train) "
                               "(run.build does) or load a state_dict that holds it")
        return m, s

    def transform(self, y: torch.Tensor, target: str) -> torch.Tensor:
        """Label units -> the heads' normalised space."""
        m, s = self._stats(target)
        return (y - m.to(y)) / s.to(y)

    def std_of(self, target: str) -> torch.Tensor:
        """The fitted label sd of ``target``: the unit of that head's normalised space in label units."""
        return self._stats(target)[1]

    def inverse(self, mu: torch.Tensor, beta: torch.Tensor, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The heads' (mu, beta) -> label units (nu and alpha are unit-free)."""
        m, s = self._stats(target)
        s = s.to(mu)
        return m.to(mu) + s * mu, (s * s) * beta


TARGET_NORMS.register("none", None, version=1, fitted=False)
TARGET_NORMS.register("logret_std", TargetNorm, version=1, fitted=True)


def _module(model: nn.Module) -> TargetNorm | None:
    mod = getattr(model, "target_norm", None)
    return mod if isinstance(mod, TargetNorm) else None


def fit_target_norm(model: nn.Module, y_train: dict) -> dict | None:
    """Fit the model's target normalisation on the TRAINING labels ``y_train`` ({label key: values}); None (nothing
    done) for a model without one."""
    mod = _module(model)
    if mod is None:
        return None
    return mod.fit({t: y_train[model.head_specs[t].label_key] for t in mod.targets})


def normalized_labels(model: nn.Module, labels: dict) -> dict:
    """``{target: values}`` in the space the heads work in: transformed when the model has a fitted target
    normalisation, unchanged otherwise (what the label scale is fitted on)."""
    mod = _module(model)
    if mod is None:
        return labels
    return {t: (mod.transform(torch.as_tensor(np.asarray(v), dtype=torch.float64), t).numpy() if t in mod.targets
                else v) for t, v in labels.items()}


def target_norm_state(model: nn.Module) -> dict | None:
    """{target: {mean, std}} the model decodes with now, or None for a model without a target normalisation."""
    mod = _module(model)
    if mod is None:
        return None
    return {t: {"mean": float(mod.mean[i]), "std": float(mod.std[i])} for i, t in enumerate(mod.targets)}
