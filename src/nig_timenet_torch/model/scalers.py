"""PURPOSE: The train-fitted scaling state of a model, kept INSIDE the model (buffers, saved with the weights):
    InputScaler (per-channel standardisation of the raw inputs, the TF ``logret_std`` input step) and the fitted NIG
    label scale; ``fit_scalers(model, x_train, y_train)`` is the ONE place either is fitted.
TAGS: input scaler, train statistics, standardisation, label scale, fitted state, buffers, logret_std, unfitted
PITFALLS: fitted on the TRAIN rows only (run.build does it); a module that needs a fit and did not get one raises on
    its first forward (never a silent identity); the state lives in the state_dict, so a checkpoint, the best weights
    and every consumer (training, evaluation, live) share it and nothing is decoded in a second place. Outputs stay in
    the artifact's label units, so the codec, the prediction table and every report are unchanged.

Why (docs/audit/2026-11-14-model-first/NOTES.md, N-01): the TF pipeline fits its ``logret_std`` normaliser at split
time (inputs AND targets standardised with train statistics); the artifact is exported ``price_norm_state: unfitted``
and this port trained on the raw values (inputs ~0.03, labels ~0.03), so every unit-ful constant of the model
(InstanceNorm eps 1e-4, the NIG beta floor 0.01, the softplus offsets of the ordered means) acted on the wrong scale.
"""

import math

import torch
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["InputScaler", "INPUT_SCALERS", "FITTED_SCALE", "FittedScale", "fit_scalers", "label_scale"]

INPUT_SCALERS = Registry("input_scaler", Contract("input_scaler", required_meta=("fitted",)))
#: the ``nig_scale`` value that means "fit the label scale on the train rows" (a float keeps the hand-set scale)
FITTED_SCALE = "train_std"


class _Fitted(nn.Module):
    """A module whose buffers must be fitted (or loaded) before the first forward."""

    def __init__(self):
        super().__init__()
        self._checked = False

    def _require_fit(self, value: torch.Tensor, what: str) -> None:
        if self._checked:
            return
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{what} is not fitted: call model.scalers.fit_scalers(model, x_train, y_train) (run."
                               f"build does) or load a state_dict that holds it")
        self._checked = True


class InputScaler(_Fitted):
    """``clip((x - mean) / std, +-clip)`` per input channel with TRAIN statistics (mean / std over every row and bar),
    the input step of TF's LogretStd normaliser (data/normalizers.py: clip 5). Buffers ``mean`` / ``std`` (F,) start
    as NaN and are set by ``fit`` or by a state_dict."""

    def __init__(self, n_features: int, clip: float | None = 5.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.full((n_features,), float("nan")))
        self.register_buffer("std", torch.full((n_features,), float("nan")))

    @torch.no_grad()
    def fit(self, x: torch.Tensor, chunk: int = 200_000) -> dict:
        """Set mean / std from the train inputs ``x`` (N, T, F) (float64 accumulation, in chunks); returns the state."""
        if x.dim() != 3 or x.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"InputScaler.fit expects (N, T, {self.mean.shape[0]}), got {tuple(x.shape)}")
        n, s1, s2 = 0, torch.zeros(x.shape[-1], dtype=torch.float64), torch.zeros(x.shape[-1], dtype=torch.float64)
        for i in range(0, x.shape[0], chunk):
            a = x[i:i + chunk].reshape(-1, x.shape[-1]).to(torch.float64).cpu()
            n, s1, s2 = n + a.shape[0], s1 + a.sum(0), s2 + a.square().sum(0)
        mean = s1 / n
        std = torch.sqrt(torch.clamp_min(s2 / n - mean.square(), 0.0)).clamp_min(1e-12)
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(std.to(self.std))
        self._checked = False
        return {"mean": mean.tolist(), "std": std.tolist(), "rows": int(x.shape[0])}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_fit(self.std, "InputScaler")
        z = (x - self.mean) / self.std
        return torch.clamp(z, -self.clip, self.clip) if self.clip else z


INPUT_SCALERS.register("none", None, version=1, fitted=False)
INPUT_SCALERS.register("train_std", InputScaler, version=1, fitted=True)


class FittedScale(_Fitted):
    """A positive scalar buffer ``value`` (NaN until fitted or loaded): the label scale the NIG heads and the ordered
    means work in when ``nig_scale == "train_std"``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.tensor(float("nan")))

    @torch.no_grad()
    def set(self, scale: float) -> None:
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f"the fitted label scale must be finite and > 0, got {scale!r}")
        self.value.fill_(float(scale))
        self._checked = False

    def forward(self) -> torch.Tensor:
        self._require_fit(self.value, "the NIG label scale (nig_scale='train_std')")
        return self.value


def label_scale(y_train: dict, label_keys) -> float:
    """sqrt(mean over the heads of var(label)) on the train rows: ONE scale for every NIG head, so the ordered means
    (high >= close >= low) keep their meaning in label units."""
    variances = [float(torch.as_tensor(y_train[k]).to(torch.float64).var(unbiased=False)) for k in label_keys]
    if not variances:
        raise ValueError("label_scale needs at least one regression label")
    return math.sqrt(sum(variances) / len(variances))


def fit_scalers(model: nn.Module, x_train, y_train: dict) -> dict:
    """Fit every train-fitted state of ``model`` on the TRAIN rows and return what was fitted (for the run record):
    each InputScaler from ``x_train`` (a tensor; a multi-timeframe model takes {timeframe: tensor}) and each
    FittedScale from the regression labels of the model's NIG heads. A model without such modules is left as it is."""
    from nig_timenet_torch.model.heads import head_spec  # heads imports nothing from here; kept local for clarity

    state: dict = {}
    scalers = {n: m for n, m in model.named_modules() if isinstance(m, InputScaler)}
    for name, mod in scalers.items():
        x = x_train
        if isinstance(x_train, dict):
            tf = name.removeprefix("input_scaler").lstrip("_")
            x = x_train[tf] if tf else next(iter(x_train.values()))
        state[name] = mod.fit(x)
    scales = [m for m in model.modules() if isinstance(m, FittedScale)]
    if scales:
        keys = [head_spec("nig_regression", t).label_key for t in model.nig_targets]
        s = label_scale(y_train, keys)
        for m in scales:
            m.set(s)
        state["nig_label_scale"] = s
    return state
