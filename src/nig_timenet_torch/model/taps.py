"""PURPOSE: Inspection-only taps: record the output of every (or chosen) named module of ANY architecture into a
    dict during a forward pass, per-layer summaries, and a signal trace (ridge R^2 of a known signal per layer).
TAGS: taps, layer outputs, inspection, diagnostics, forward hook, signal trace, dead units, model doctor
PITFALLS: taps are forward hooks, so the model, its outputs, its state_dict and training are unchanged (the dict is
    never seen by a loss); record under ``torch.no_grad`` on a small batch (every layer's output is kept in memory);
    a module that returns a tuple is recorded by its first element.

Why hooks and not an extra output key: every consumer of the model outputs (losses, ``Trainer.evaluate``, the
prediction table, the model contract) iterates the output dict, so a ``taps`` key would leak into them; a hook gives
the same per-layer dict for every architecture registered in ``ARCHITECTURES`` without touching any of them.

``signal_trace`` answers "does a layer erase the signal?": on E1 the true conditional mean of each label is known, so
the out-of-fold ridge R^2 of that mean from each layer's output, next to the same R^2 from the raw input, shows where
along the model the information is lost (``model.probe.layer_probe_report`` does the fitting).
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import nn

from nig_timenet_torch.model.probe import RAW_INPUT, layer_probe_report

__all__ = ["record_taps", "stream_taps", "tap_points", "tap_summary", "signal_trace"]


def tap_points(model: nn.Module, *, leaves_only: bool = False) -> list[str]:
    """Every named sub-module of ``model`` in registration order (``leaves_only``: modules without children)."""
    return [n for n, m in model.named_modules() if n and (not leaves_only or not any(True for _ in m.children()))]


@contextmanager
def record_taps(model: nn.Module, points: list[str] | None = None):
    """``with record_taps(model) as taps: out = model(x)`` -> ``taps`` is {module name: detached output tensor} of
    the last forward pass (``points`` default: every named module). Unknown names raise KeyError."""
    modules = dict(model.named_modules())
    points = tap_points(model) if points is None else list(points)
    unknown = [p for p in points if p not in modules]
    if unknown:
        raise KeyError(f"unknown tap point(s) {unknown}")
    taps: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _inp, out):
            out = out[0] if isinstance(out, tuple) else out
            if isinstance(out, torch.Tensor):
                taps[name] = out.detach()
        return fn

    handles = [modules[p].register_forward_hook(hook(p)) for p in points]
    try:
        yield taps
    finally:
        for h in handles:
            h.remove()


@contextmanager
def stream_taps(model: nn.Module, edit: dict | None = None):
    """``with stream_taps(model) as taps: model(x)`` -> ``taps`` = {stream name: tensor} in forward ORDER, for a model
    that taps its own streams (``model._tap``; NigTimeNetV2: input, scaled, inst_norm, stats, level, tokens, enc_in,
    enc_block_i, enc_out, pooled, readout, linear_path, branch_out, trunk). Unlike module hooks this sees the RESIDUAL
    stream after each block (a sum, not a module output). A model without ``_tap`` raises TypeError: use
    ``record_taps`` (module outputs) for it.

    ``edit`` = {stream name: fn(tensor) -> tensor}: the stream is REPLACED by fn's result for the rest of the forward
    (an ablation, e.g. ``lambda v: v[torch.randperm(len(v))]`` breaks the row <-> value link of one path)."""
    if not hasattr(model, "_tap"):
        raise TypeError(f"{type(model).__name__} has no stream taps; use record_taps (module hooks)")
    taps: dict[str, torch.Tensor] = {}
    model._stream, model._stream_edit = taps, dict(edit or {})
    try:
        yield taps
    finally:
        model._stream, model._stream_edit = None, {}


def tap_summary(taps: dict[str, torch.Tensor]) -> pd.DataFrame:
    """One row per tap: shape, mean, std, share of exactly-zero / non-positive entries, share of features that never
    exceed 0 over the batch (``dead_share``, units that a ReLU-family activation would silence) and the share of
    features whose std over the batch is < 1e-6 of the tap's std (``constant_share``: a feature carries nothing)."""
    rows = []
    for name, t in taps.items():
        a = t.float().reshape(t.shape[0], -1) if t.dim() > 1 else t.float().reshape(-1, 1)
        sd = float(a.std()) if a.numel() > 1 else 0.0
        feat_sd = a.std(dim=0) if a.shape[0] > 1 else torch.zeros(a.shape[1])
        rows.append({"tap": name, "shape": tuple(t.shape), "mean": float(a.mean()), "std": sd,
                     "zero_share": float((a == 0).float().mean()), "nonpos_share": float((a <= 0).float().mean()),
                     "dead_share": float((a.max(dim=0).values <= 0).float().mean()),
                     "constant_share": float((feat_sd <= 1e-6 * max(sd, 1e-12)).float().mean())})
    return pd.DataFrame(rows)


def signal_trace(model: nn.Module, x: torch.Tensor, signals: dict[str, np.ndarray], points: list[str], *,
                 ridge: float = 1.0, batch_size: int = 256) -> pd.DataFrame:
    """Out-of-fold ridge R^2 of each known ``signals[name]`` (e.g. the E1 oracle mean of a label, aligned with ``x``)
    from the raw input and from every module in ``points``; one row per (signal, point) with ``delta_vs_raw``. A
    layer whose R^2 falls well below the raw input's lost that information; the heads can only use what is left."""
    frames = []
    for name, y in signals.items():
        rep = layer_probe_report(model, x, np.asarray(y, dtype=np.float64), probe_points=[RAW_INPUT, *points],
                                 ridge=ridge, batch_size=batch_size)
        rep.insert(0, "signal", name)
        frames.append(rep)
    return pd.concat(frames, ignore_index=True)
