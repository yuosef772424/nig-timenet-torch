"""Training the autoregressive candle transformer (``model/ar_candle_transformer.py``).

The loop trains the model the way it will be used: candle by candle over the horizon, appending each
step's input as it goes (``_rollout_step``). With probability ``p`` (a per-epoch curriculum from
``ss_start`` to ``ss_end``) the appended candle is the model's OWN prediction, detached — the model
learns to consume its own outputs (like a text model in deployment) but the noise they contain never
becomes a learning signal, because the gradient stops at the detach and the loss is always scored
against the TRUE candle.

The loss is deliberate about sudden market moves (the "aware trainer"):
* the magnitude term is Huber with ``huber_delta``, so a shock moves the gradient linearly, not quadratically;
* each horizon step carries an adaptive weight ``1 / (1 + (|ret| / (outlier_softness * ret_scale))^2)``
  that shrinks as the true move becomes an outlier — the model is not punished hard for shocks;
* the DIRECTION term keeps its full weight (the sign of a shock is still a lesson), so the model is
  encouraged to get the direction right first.

Selection (rule C3) is fixed in code: the best epoch is the one with the highest validation direction
accuracy of the full autoregressive rollout — the quantity this model exists for.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.ar_candle_transformer import ArCandleTransformer, rollout

__all__ = ["ArStreamSettings", "build_ar_stream_model", "step_weights", "robust_candle_loss",
           "fit_ar_stream", "rollout_direction_accuracy"]


@dataclass(frozen=True)
class ArStreamSettings:
    """Every knob of the autoregressive stream run. ``horizon_len`` is free: 4, 8, 16, ... (the window
    is ``context_len + horizon_len`` candles, 48 by default). ``encoder`` picks the trunk:
    "transformer" (the causal GPT-style default) or "gru" (the ANTI_MEMORIZATION preset's cheap trunk)."""
    context_len: int = 32
    horizon_len: int = 16
    encoder: str = "transformer"
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ss_start: float = 0.0
    ss_end: float = 0.5
    huber_delta: float = 1.0
    w_dir: float = 1.0
    w_sigma: float = 0.1
    outlier_softness: float = 8.0
    patience: int = 4
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True


def build_ar_stream_model(settings: ArStreamSettings, n_features: int) -> ArCandleTransformer:
    """The model of this run, from the settings' architecture fields (transformer or gru trunk)."""
    return ArCandleTransformer(n_features=n_features, context_len=settings.context_len,
                               horizon_len=settings.horizon_len, d_model=settings.d_model,
                               num_layers=settings.num_layers, num_heads=settings.num_heads,
                               num_kv_heads=settings.num_kv_heads, dropout=settings.dropout,
                               encoder=settings.encoder)


def step_weights(y_rets: torch.Tensor, ret_scale: float, outlier_softness: float) -> torch.Tensor:
    """(B, H) magnitude weights: 1 for typical moves, shrinking towards 0 as |ret| grows past
    ``outlier_softness * ret_scale`` — a sudden market move is down-weighted, not deleted."""
    scale = max(float(ret_scale), 1e-8)
    ratio = y_rets.abs() / (float(outlier_softness) * scale)
    return 1.0 / (1.0 + ratio * ratio)


def robust_candle_loss(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, dir_logit: torch.Tensor,
                       dir_true: torch.Tensor, weight: torch.Tensor, s: ArStreamSettings) -> tuple:
    """(loss, stats). Magnitude: Huber per feature, mean over features, scaled by the adaptive weight.
    Direction: BCE-with-logits at FULL weight (the sign of a shock is still a lesson). Uncertainty: the
    predicted log sigma is pulled towards log(1 + |error|) with the error detached, so an honest large
    sigma on a shock is cheap while a small sigma on a large error is not."""
    err = y - mu
    a = err.abs()
    q = torch.clamp_max(a, s.huber_delta)
    huber = 0.5 * q.square() + s.huber_delta * (a - q)
    mag = weight * huber.mean(dim=-1)
    dirn = F.binary_cross_entropy_with_logits(dir_logit, (dir_true > 0).to(dir_logit.dtype), reduction="none")
    target = torch.log1p(a.detach().mean(dim=-1))
    sig = (log_sigma.mean(dim=-1) - target).abs()
    total = (mag + s.w_dir * dirn + s.w_sigma * sig).mean()
    stats = {"magnitude": mag.mean().detach(), "direction": dirn.mean().detach(), "sigma": sig.mean().detach(),
             "mae": a.mean().detach()}
    return total, stats


def rollout_direction_accuracy(model: ArCandleTransformer, X: torch.Tensor, dirs: torch.Tensor,
                               device: str = "cpu", batch_size: int = 1024) -> float:
    """Direction accuracy of the full autoregressive rollout (own predictions only): the share of
    (window, horizon step) pairs whose predicted next-candle direction matches the true sign."""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb, db = X[i:i + batch_size].to(device), dirs[i:i + batch_size].to(device)
            out = rollout(model, xb)
            up = torch.sign(out["mu"][:, :, 0]) > 0
            keep = db != 0  # a flat true move teaches nothing about direction
            hits += int(((up == (db > 0)) & keep).sum())
            total += int(keep.sum())
    return hits / max(total, 1)


def _rollout_step(model: ArCandleTransformer, xb: torch.Tensor, yb: torch.Tensor, p_own: float):
    """One training batch, the deployment way: for every horizon step, predict the next candle, score it
    against the truth, then append the model's own prediction (probability ``p_own``, DETACHED) or the
    true candle as the next input. Gradients flow only through each step's prediction."""
    b, h = xb.shape[0], yb.shape[1]
    seq = xb
    mus, log_sigmas, dir_logits = [], [], []
    own = torch.rand(b, device=xb.device) < p_own
    for t in range(h):
        out = model(seq)
        mus.append(out["mu"][:, -1])
        log_sigmas.append(out["log_sigma"][:, -1])
        dir_logits.append(out["dir_logit"][:, -1])
        mu_t = out["mu"][:, -1].detach()
        nxt = torch.where(own[:, None], mu_t, yb[:, t])
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return (torch.stack(mus, dim=1), torch.stack(log_sigmas, dim=1), torch.stack(dir_logits, dim=1))


def _train_epoch(model, opt, X, Y, dirs, w, s, p_own, device, epoch: int) -> dict:
    model.train()
    n = len(X)
    sums = {"loss": 0.0, "magnitude": 0.0, "direction": 0.0, "sigma": 0.0}
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(int(s.seed) + epoch))
    for i in range(0, n, s.batch_size):
        rows = idx[i:i + s.batch_size].to(device)
        xb, yb, db, wb = X[rows], Y[rows], dirs[rows], w[rows]
        mu, ls, dl = _rollout_step(model, xb, yb, p_own)
        loss, stats = robust_candle_loss(mu, ls, yb, dl, db, wb, s)
        opt.zero_grad()
        loss.backward()
        if s.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
        opt.step()
        sums["loss"] += float(loss.detach())
        for k in ("magnitude", "direction", "sigma"):
            sums[k] += float(stats[k])
    steps = max(1, math.ceil(n / s.batch_size))
    return {k: v / steps for k, v in sums.items()}


def fit_ar_stream(model: ArCandleTransformer, X: np.ndarray, Y: np.ndarray, dirs: np.ndarray,
                  vX: np.ndarray, vY: np.ndarray, vdirs: np.ndarray,
                  settings: ArStreamSettings) -> dict:
    """Train on (X, Y, dirs), select on the validation rollout direction accuracy, early stop after
    ``patience`` epochs without improvement and reload the best weights. Returns the history dict."""
    if X.shape[1] != settings.context_len or Y.shape[1] < settings.horizon_len:
        raise ValueError(f"windows {X.shape[1]}+{Y.shape[1]} do not fit settings "
                         f"{settings.context_len}+{settings.horizon_len}")
    # the data may carry a LONGER future than this run trains on (a free eval horizon): train on the
    # first horizon_len steps only
    Y, vY, dirs, vdirs = (a[:, :settings.horizon_len] for a in (Y, vY, dirs, vdirs))
    seed_everything(settings.seed)
    device = settings.device
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    X, Y, dirs, vX, vY, vdirs = t(X), t(Y), t(dirs), t(vX), t(vY), t(vdirs)
    ret_scale = float(torch.median(Y[:, :, 0].abs()))
    w = step_weights(Y[:, :, 0], ret_scale, settings.outlier_softness)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    model.to(device)
    ramp = max(settings.epochs - 1, 1)
    hist = {"train_loss": [], "magnitude": [], "direction": [], "sigma": [], "val_dir_acc": [], "p_own": []}
    best, best_state, bad = -1.0, None, 0
    for epoch in range(settings.epochs):
        p_own = settings.ss_start + (settings.ss_end - settings.ss_start) * epoch / ramp
        row = _train_epoch(model, opt, X, Y, dirs, w, settings, p_own, device, epoch)
        acc = rollout_direction_accuracy(model, vX, vdirs, device)
        for k, v in (("train_loss", row["loss"]), ("magnitude", row["magnitude"]),
                     ("direction", row["direction"]), ("sigma", row["sigma"]),
                     ("val_dir_acc", acc), ("p_own", p_own)):
            hist[k].append(v)
        if settings.verbose:
            print(f"epoch {epoch + 1}/{settings.epochs}  loss {row['loss']:.4f}  "
                  f"dir {row['direction']:.4f}  val_dir_acc {acc:.4f}  p_own {p_own:.2f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= settings.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_dir_acc"], hist["best_epoch"] = best, int(np.argmax(hist["val_dir_acc"]))
    return hist
