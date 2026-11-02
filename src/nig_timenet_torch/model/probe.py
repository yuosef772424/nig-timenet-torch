"""P35: per-layer linear-probe report — the TOOL for the owner's hypothesis "an early Dense projection may destroy
information" (TF-side S07 layer_probe.csv; crypto-signal-prediction docs/research/MODEL_DEV_PROTOCOL.md section 2-3).
It reports numbers and never a verdict.

For every probe point (a module name inside ``model``; the reserved name "raw_input" is the model's raw input itself),
the module's forward output (first element when it returns a tuple) is captured by a forward hook, flattened per
sample, standardised on the training fold, and a closed-form ridge regression (numpy only) is fitted to ``y``.
Evaluation: 5 contiguous (time-ordered, unshuffled) folds; metric = AUC (stats.metrics.auc) when ``y`` is binary
{0, 1}, else out-of-fold R^2. One row per probe point: mean and std over folds.
"""

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.stats.metrics import auc

__all__ = ["RAW_INPUT", "capture_activations", "ridge_fit_predict", "layer_probe_report"]

RAW_INPUT = "raw_input"


@torch.no_grad()
def capture_activations(model: torch.nn.Module, x: torch.Tensor, probe_points: list[str], *,
                        batch_size: int = 256) -> dict[str, np.ndarray]:
    modules = dict(model.named_modules())
    unknown = [p for p in probe_points if p != RAW_INPUT and p not in modules]
    if unknown:
        raise KeyError(f"unknown probe point(s) {unknown}; module names: {sorted(n for n in modules if n)}")
    store: dict[str, list[np.ndarray]] = {p: [] for p in probe_points}

    def hook(name):
        def fn(_m, _inp, out):
            out = out[0] if isinstance(out, tuple) else out
            store[name].append(out.detach().flatten(1).double().cpu().numpy())
        return fn

    handles = [modules[p].register_forward_hook(hook(p)) for p in probe_points if p != RAW_INPUT]
    was_training = model.training
    model.eval()
    try:
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            if RAW_INPUT in store:
                store[RAW_INPUT].append(xb.flatten(1).double().cpu().numpy())
            model(xb)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {p: np.concatenate(v, axis=0) for p, v in store.items()}


def ridge_fit_predict(a_train: np.ndarray, y_train: np.ndarray, a_test: np.ndarray, ridge: float) -> np.ndarray:
    """Closed-form ridge on standardised features with an (unpenalised) intercept; primal or dual form,
    whichever system is smaller."""
    mu, sd = a_train.mean(0), a_train.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    a, b = (a_train - mu) / sd, (a_test - mu) / sd
    y0 = y_train.mean()
    yc = y_train - y0
    n, d = a.shape
    if d <= n:
        w = np.linalg.solve(a.T @ a + ridge * np.eye(d), a.T @ yc)
    else:
        w = a.T @ np.linalg.solve(a @ a.T + ridge * np.eye(n), yc)
    return b @ w + y0


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan")


def layer_probe_report(model: torch.nn.Module, x: torch.Tensor, y: np.ndarray, *, probe_points: list[str],
                       n_folds: int = 5, ridge: float = 10.0, batch_size: int = 256) -> pd.DataFrame:
    """For each name in ``probe_points`` (module names inside ``model``, e.g. "instance_norm", "patch_embed",
    "block_1", "block_2", "trunk_norm"; plus "raw_input" = the flattened raw input window), fit a ridge probe from
    that activation to ``y`` with ``n_folds``-fold contiguous CV and report mean +/- std of AUC (binary y) or R^2.
    "raw_input" is always added (first row) when missing."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"y has {y.shape[0]} rows, x has {x.shape[0]}")
    points = list(dict.fromkeys([RAW_INPUT, *probe_points]))
    acts = capture_activations(model, x, points, batch_size=batch_size)
    binary = bool(np.isin(np.unique(y), (0.0, 1.0)).all())
    folds = np.array_split(np.arange(len(y)), n_folds)
    rows = []
    for p in points:
        a = acts[p]
        scores = []
        for k, test in enumerate(folds):
            train = np.concatenate([f for j, f in enumerate(folds) if j != k])
            pred = ridge_fit_predict(a[train], y[train], a[test], ridge)
            scores.append(auc(y[test], pred) if binary else _r2(y[test], pred))
        scores = np.asarray(scores, dtype=np.float64)
        rows.append({"probe_point": p, "metric": "auc" if binary else "r2", "mean": float(np.mean(scores)),
                     "std": float(np.std(scores)), "n_features": int(a.shape[1]), "n_folds": n_folds,
                     "n_samples": int(len(y))})
    return pd.DataFrame(rows)
