"""Evaluation of the autoregressive candle transformer: the honest test is the full rollout (the model
eats only its own predictions, exactly like generation in a text model), scored on DIRECTION accuracy
per horizon step and as a whole, always next to the two trivial baselines rule C1 demands (the
persistence sign of the last context return, and "always up"), plus a shuffled-labels control that
must sit at ~50 %: if it does not, the metric is broken, not the model good.
"""

import numpy as np
import torch

from nig_timenet_torch.model.ar_candle_transformer import rollout
from nig_timenet_torch.stats import metrics

__all__ = ["rollout_metrics", "panel_rollout_metrics", "persistence_accuracy", "always_up_accuracy",
           "direction_control", "panel_direction_control"]


def _batched(model, X, dirs, device, batch_size, horizon_len=None):
    """Yield (true dirs, predicted return col, predicted features) per batch of the rollout."""
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        out = rollout(model, xb, horizon_len)
        yield (dirs[i:i + batch_size], out["mu"][:, :, 0].detach().cpu().numpy(),
               out["mu"].detach().cpu().numpy())


def persistence_accuracy(last_ret: np.ndarray, dirs: np.ndarray) -> float:
    """The trivial baseline (rule C1): predict the next direction as the SIGN of the last context
    return, per horizon step. A model must beat this to claim it learned anything."""
    up = np.repeat(np.sign(last_ret)[:, None], dirs.shape[1], axis=1) > 0
    keep = dirs != 0
    return float((up[keep] == (dirs[keep] > 0)).mean())


def always_up_accuracy(dirs: np.ndarray) -> float:
    """The second trivial baseline: "every next candle closes up". Also the expected value of the
    shuffled-labels control."""
    keep = dirs != 0
    return float((dirs[keep] > 0).mean())


def rollout_metrics(model, X: np.ndarray, Y: np.ndarray, dirs: np.ndarray, last_ret: np.ndarray,
                    device: str = "cpu", batch_size: int = 1024, horizon_len: int | None = None) -> dict:
    """Full-rollout test metrics: overall and per-step direction accuracy (vs both trivial baselines)
    and the mean absolute error of the predicted return feature (``stats.metrics.mae``). ``horizon_len``
    is FREE and independent of training: score any prefix of the data's horizon (4, 8, ... steps) with a
    model trained on another length - the rollout simply generates that many candles."""
    h = dirs.shape[1] if horizon_len is None else int(horizon_len)
    if h < 1 or h > dirs.shape[1]:
        raise ValueError(f"horizon_len {h} outside the data's horizon 1..{dirs.shape[1]}")
    Y, dirs = Y[:, :h], dirs[:, :h]
    model.eval()
    pred_dirs, pred_ret = [], []
    with torch.no_grad():
        for _, ret, mu in _batched(model, X, dirs, device, batch_size, h):
            pred_dirs.append(np.sign(ret))
            pred_ret.append(mu)
    return _direction_report(np.concatenate(pred_dirs), np.concatenate(pred_ret), Y, dirs, last_ret)


def panel_rollout_metrics(model, X: np.ndarray, Y: np.ndarray, dirs: np.ndarray, last_ret: np.ndarray,
                         device: str = "cpu", batch_size: int = 256, horizon_len: int | None = None) -> dict:
    """The panel twin of ``rollout_metrics``: X, Y, dirs, last_ret are (windows, coins, ...); the
    panel-wide rollout generates every coin's candles in lockstep and all coins' steps are pooled into
    the same direction report (trivial baselines included, ``horizon_len`` free as everywhere)."""
    from nig_timenet_torch.model.panel_ar_candle import rollout as panel_rollout
    h = dirs.shape[2] if horizon_len is None else int(horizon_len)
    if h < 1 or h > dirs.shape[2]:
        raise ValueError(f"horizon_len {h} outside the data's horizon 1..{dirs.shape[2]}")
    Y, dirs = Y[:, :, :h], dirs[:, :, :h]
    model.eval()
    pred_dirs, pred_ret = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            out = panel_rollout(model, xb, h)
            pred_dirs.append(np.sign(out["mu"][:, :, :, 0].detach().cpu().numpy()))
            pred_ret.append(out["mu"].detach().cpu().numpy())
    flat = lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])  # noqa: E731
    return _direction_report(flat(np.concatenate(pred_dirs)), flat(np.concatenate(pred_ret)),
                             flat(Y), flat(dirs), flat(last_ret))


def _direction_report(pred_dirs: np.ndarray, pred_ret: np.ndarray, Y: np.ndarray, dirs: np.ndarray,
                      last_ret: np.ndarray) -> dict:
    """The shared scoring of one rollout's predictions against the truth (both baselines, per step)."""
    pred_ret = pred_ret[:, :, 0].reshape(-1)
    true_ret = Y[:, :, 0].reshape(-1)
    keep = dirs != 0
    hit = (pred_dirs > 0) == (dirs > 0)
    per_step = [float(hit[dirs[:, j] != 0, j].mean()) for j in range(dirs.shape[1])]
    overall = float(hit[keep].mean())
    return {
        "direction_accuracy": overall,
        "direction_accuracy_per_step": per_step,
        "persistence_accuracy": persistence_accuracy(last_ret, dirs),
        "always_up_accuracy": always_up_accuracy(dirs),
        "n_scored": int(keep.sum()),
        "mae_ret": float(metrics.mae(true_ret[keep.reshape(-1)], pred_ret[keep.reshape(-1)])),
        "trivial_gap": float(overall - persistence_accuracy(last_ret, dirs)),
    }


def _null_score(pred_dirs: np.ndarray, dirs: np.ndarray) -> float:
    """The honest null of the shuffled control: with the predictions fixed and the labels permuted, the
    expected agreement is p_up * up_share + (1 - p_up) * (1 - up_share) over the scored steps (a biased
    model lands here, NOT at the always-up share)."""
    keep = dirs != 0
    up = (dirs[keep] > 0).mean()
    p_up = (pred_dirs[keep] > 0).mean()
    return float(p_up * up + (1.0 - p_up) * (1.0 - up))


def _shuffle_scores(pred_dirs: np.ndarray, dirs: np.ndarray, rng: np.random.Generator, n_shuffles: int) -> list:
    """Score the same predictions against repeatedly shuffled true directions."""
    keep = dirs != 0
    scores = []
    for _ in range(max(1, int(n_shuffles))):
        shuffled = dirs[keep][rng.permutation(int(keep.sum()))]
        scores.append(float(((pred_dirs[keep] > 0) == (shuffled > 0)).mean()))
    return scores


def direction_control(model, X: np.ndarray, dirs: np.ndarray, device: str = "cpu", n_shuffles: int = 5,
                      seed: int = 0, horizon_len: int | None = None) -> dict:
    """The control experiment (rule C1): score the SAME rollout predictions against SHUFFLED true
    directions. The result must sit at the honest null (``_null_score``: a biased model's null is its
    majority agreement, not 50 %) — anything clearly higher means the scoring leaks, and any model claim
    built on it is void. ``horizon_len`` is free, as in rollout_metrics."""
    h = dirs.shape[1] if horizon_len is None else int(horizon_len)
    if h < 1 or h > dirs.shape[1]:
        raise ValueError(f"horizon_len {h} outside the data's horizon 1..{dirs.shape[1]}")
    dirs = dirs[:, :h]
    rng = np.random.default_rng(seed)
    model.eval()
    pred_dirs = []
    with torch.no_grad():
        for _, ret, _ in _batched(model, X, dirs, device, 1024, h):
            pred_dirs.append(np.sign(ret))
    pred_dirs = np.concatenate(pred_dirs)
    scores = _shuffle_scores(pred_dirs, dirs, rng, n_shuffles)
    return {"shuffled_direction_accuracy": float(np.mean(scores)),
            "shuffled_scores": [float(s) for s in scores],
            "expected": _null_score(pred_dirs, dirs),
            "always_up_accuracy": always_up_accuracy(dirs)}


def panel_direction_control(model, X: np.ndarray, dirs: np.ndarray, device: str = "cpu", n_shuffles: int = 5,
                            seed: int = 0, horizon_len: int | None = None) -> dict:
    """The panel twin of ``direction_control``: X (windows, coins, context, F), dirs (windows, coins,
    horizon); all coins' steps are pooled after the panel-wide rollout. ``horizon_len`` is free."""
    from nig_timenet_torch.model.panel_ar_candle import rollout as panel_rollout
    h = dirs.shape[2] if horizon_len is None else int(horizon_len)
    if h < 1 or h > dirs.shape[2]:
        raise ValueError(f"horizon_len {h} outside the data's horizon 1..{dirs.shape[2]}")
    dirs = dirs[:, :, :h]
    flat = lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])  # noqa: E731
    dirs = flat(dirs)
    rng = np.random.default_rng(seed)
    model.eval()
    pred_dirs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.as_tensor(X[i:i + 256], dtype=torch.float32, device=device)
            out = panel_rollout(model, xb, h)
            pred_dirs.append(np.sign(out["mu"][:, :, :, 0].detach().cpu().numpy()))
    pred_dirs = flat(np.concatenate(pred_dirs))
    scores = _shuffle_scores(pred_dirs, dirs, rng, n_shuffles)
    return {"shuffled_direction_accuracy": float(np.mean(scores)),
            "shuffled_scores": [float(s) for s in scores],
            "expected": _null_score(pred_dirs, dirs),
            "always_up_accuracy": always_up_accuracy(dirs)}
