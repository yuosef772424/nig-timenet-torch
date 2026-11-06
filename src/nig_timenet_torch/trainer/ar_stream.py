"""Training the autoregressive candle transformer (``model/ar_candle_transformer.py``).

The loop trains the model the way it will be used: candle by candle over the horizon, appending each
step's input as it goes (``_rollout_step``). With probability ``p`` (a per-epoch curriculum from
``ss_start`` to ``ss_end``) the appended candle is the model's OWN prediction, detached — the model
learns to consume its own outputs (like a text model in deployment) but the noise they contain never
becomes a learning signal, because the gradient stops at the detach and the loss is always scored
against the TRUE candle.

The loss is deliberate about markets whose moves are mostly UNPREDICTABLE (a headline, a thin book, one
whale order) and about the cowardice those markets teach:
* the RETURN channel is scored by the Composite Zero-Agnostic Return loss (CZAR, Pfeffer et al. 2026,
  arXiv:2609.36061) on the per-window LOCAL volatility: every step is normalized by its own market
  regime, so a shock is bounded by the Cauchy kernel instead of exploding the gradient; a prediction
  shrunk towards zero on a real move keeps a near-LINEAR penalty — the anti-cowardice term that makes
  "predict ~0 always" the expensive option, not the safe one; and near-zero truths stop teaching noise
  (the zero-agnostic floor counts, but does not differentiate, the zero-prediction loss);
* the bounded ratio channels (wick/body/volume) keep a Huber term, down-weighted on shock steps;
* the DIRECTION term EXCLUDES flat truths (the sign of a flat candle is noise) and ramps its weight up
  with the move size in local-vol units — a shock's SIGN is the lesson, its magnitude is not.

Selection (rule C3) is fixed in code: the best epoch is the one with the highest validation direction
accuracy of the full autoregressive rollout — the quantity this model exists for, and a scale-free
metric, so the shrinkage that a symmetric validation loss would reintroduce cannot come back through
model selection either.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.ar_candle_transformer import ArCandleTransformer, rollout

__all__ = ["ArStreamSettings", "build_ar_stream_model", "czar_loss", "czar_softening", "local_vol",
           "step_weights", "robust_candle_loss", "fit_ar_stream", "rollout_direction_accuracy"]


@dataclass(frozen=True)
class ArStreamSettings:
    """Every knob of the autoregressive stream run. ``horizon_len`` is free: 4, 8, 16, ... (the window
    is ``context_len + horizon_len`` candles, 48 by default). ``encoder`` picks the trunk:
    "transformer" (the causal GPT-style default) or "gru" (the ANTI_MEMORIZATION preset's cheap trunk)."""
    context_len: int = 32
    horizon_len: int = 16
    encoder: str = "transformer"
    loss: str = "czar"  # "czar" (anti-cowardice, shock-bounded) or "huber" (the v1 robust loss)
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
    czar_alpha: float = 1.0
    czar_epsilon: float = 1.0
    czar_tau: float = 0.05
    w_dir: float = 1.0
    w_mag: float = 0.15
    w_move: float = 0.5
    w_sigma: float = 0.1
    outlier_softness: float = 8.0
    vol_floor: float = 0.25
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
    """(B, H) magnitude weights for the "huber" loss kind: 1 for typical moves, shrinking towards 0 as
    |ret| grows past ``outlier_softness * ret_scale`` — a sudden market move is down-weighted, not deleted."""
    scale = max(float(ret_scale), 1e-8)
    ratio = y_rets.abs() / (float(outlier_softness) * scale)
    return 1.0 / (1.0 + ratio * ratio)


def local_vol(X: torch.Tensor, ret_scale: float, floor_frac: float) -> torch.Tensor:
    """(B,) the per-window volatility of the CONTEXT returns (feature 0), floored at ``floor_frac`` of
    the train-head median |return| — every window's future is then scored in its own market regime, and
    a dead-quiet stretch can never divide by ~0."""
    vol = X[:, :, 0].std(dim=1)
    return vol.clamp_min(floor_frac * max(float(ret_scale), 1e-8))


def _czar_base(z_true: torch.Tensor, z_pred: torch.Tensor, alpha: float) -> torch.Tensor:
    """The CZAR base loss (Pfeffer et al. 2026, arXiv:2609.36061), vectorized. Three regions in the
    (z_true, z_pred) plane, z = return / local volatility:
    * Region A (wrong sign OR zero): ``0.5·h1·z_pred² − s·z_pred + c`` — the penalty keeps a LINEAR
      piece ``−s·z_pred``, so the gradient never stops pushing the prediction AWAY from zero: the
      anti-cowardice term that makes "always predict ~0" expensive instead of safe;
    * Region B (right sign, undershoot): a Cauchy-kernel difference — smooth, and it keeps growing
      (unlike Huber) until the prediction catches up with the move;
    * Region C (right sign, overshoot): the SMALL double-derivative (``min(h1, h3)``) — boldness is
      cheap, so the model is never taught to shrink."""
    s = torch.where(z_true == 0, 1.0, torch.sign(z_true))
    s_pred = torch.where(z_pred == 0, 1.0, torch.sign(z_pred))
    u = s * z_pred
    a = z_true.abs()
    delta = alpha / math.sqrt(3.0)
    d2 = 1.0 + delta * delta
    d_true = z_true + s * delta
    d_pred = z_pred + s_pred * delta
    h1 = torch.as_tensor(d2 * 2.0 * abs(delta) / (d2 * d2))
    h3 = d2 * 2.0 * d_true.abs() / (1.0 + d_true * d_true).square()
    c = s * d2 * (torch.atan(d_true) - torch.atan(s * delta))
    l1 = 0.5 * h1 * z_pred * z_pred - s * z_pred + c
    l2 = s * d2 * (torch.atan(d_true) - torch.atan(d_pred))
    dz = z_pred - z_true
    l3 = 0.5 * torch.minimum(h3, h1) * dz * dz + s * d2 / (1.0 + d_true * d_true) * dz
    return torch.where(u <= 0, l1, torch.where(u <= a, l2, l3))


def czar_softening(z_true: torch.Tensor, alpha: float, epsilon: float, tau: float) -> torch.Tensor:
    """The zero-agnostic floor (arXiv:2609.36061 §4): a function of the TRUE z only — no gradient flows
    through it. Near-zero truths contribute the loss of the ZERO prediction against an
    ``eps_eff = arctan(epsilon)``-scaled truth (so fitting flat noise cannot pay), decaying smoothly to
    0 as the truth grows past the ``epsilon``-band; ``tau`` smooths the clamp."""
    if epsilon <= 0.0:
        return torch.zeros_like(z_true)
    delta = alpha / math.sqrt(3.0)
    d2 = 1.0 + delta * delta
    beta = delta / d2
    if abs(delta) < 1e-12:  # alpha == 0: the pure Cauchy kernel
        eps_eff, denom = math.atan(epsilon), epsilon
    else:
        big = d2 * (math.atan(epsilon + delta) - math.atan(delta))
        eps_eff = (-1.0 + math.sqrt(1.0 + 4.0 * beta * big)) / (2.0 * beta)
        denom = epsilon + beta * epsilon * epsilon
    floor = _czar_base(torch.full_like(z_true, eps_eff), torch.zeros_like(z_true), alpha)
    num = d2 * (torch.atan(z_true.abs() + delta) - math.atan(torch.as_tensor(delta)))
    norm = 1.0 - num / denom
    tau_eff = max(tau * abs(1.0 - d2 * (math.pi / 2 - math.atan(delta)) / denom), 1e-8)
    norm = norm - tau_eff * F.softplus((norm - 1.0) / tau_eff)  # a smooth clamp to <= 1
    return floor * torch.clamp(norm, min=0.0)


def czar_loss(z_true: torch.Tensor, z_pred: torch.Tensor, alpha: float = 1.0, epsilon: float = 1.0,
              tau: float = 0.05) -> torch.Tensor:
    """``_czar_base`` + the zero-agnostic floor — the full Composite Zero-Agnostic Return loss."""
    return _czar_base(z_true, z_pred, alpha) + czar_softening(z_true, alpha, epsilon, tau)


def _czar_candle_loss(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, dir_logit: torch.Tensor,
                      dir_true: torch.Tensor, vol: torch.Tensor, _w: torch.Tensor,
                      s: ArStreamSettings) -> tuple:
    """(loss, stats). The RETURN channel is scored by CZAR in z units (return / local volatility) but
    PAID in return units (multiplied by that local vol): the loss is regime-FAIR in value while its
    gradient wrt the prediction stays bounded ~O(1) in every regime — a calm window is never dwarfed
    by a wild one, and the magnitude lesson never drowns the direction lesson. A shock is bounded by
    the Cauchy kernel instead of exploding the gradient, a shrunk-to-zero prediction on a real move
    stays expensive (anti-cowardice), and near-zero truths stop teaching noise (the zero-agnostic
    floor). The bounded ratio channels (wick/body/volume) keep a Huber term, down-weighted on shock
    steps. The DIRECTION BCE EXCLUDES flat truths (the sign of a flat candle is noise) and ramps its
    weight up with the move size in local-vol units — a shock's sign is the lesson, its magnitude is
    not. Sigma is pulled towards log1p(|error| / local vol), the error detached: an honest large sigma
    on a shock is cheap, a small one is not."""
    vol = vol[:, None] if vol.dim() == 1 else vol  # (B,) broadcasts over the horizon
    y_ret, mu_ret = y[..., 0], mu[..., 0]
    z_true, z_pred = y_ret / vol, mu_ret / vol
    mag_ret = vol * czar_loss(z_true, z_pred, s.czar_alpha, s.czar_epsilon, s.czar_tau)
    a = (y[..., 1:] - mu[..., 1:]).abs()
    q = torch.clamp_max(a, s.huber_delta)
    huber = 0.5 * q.square() + s.huber_delta * (a - q)
    shock = 1.0 / (1.0 + (z_true.abs() / s.outlier_softness).square())
    mag = s.w_mag * (mag_ret + shock * huber.mean(dim=-1))
    keep = dir_true != 0
    move = torch.tanh(z_true.abs())
    bce = F.binary_cross_entropy_with_logits(dir_logit, (dir_true > 0).to(dir_logit.dtype), reduction="none")
    dirn = torch.where(keep, (1.0 + s.w_move * move) * bce, torch.zeros_like(bce))
    dirn = dirn.sum() / keep.to(dirn.dtype).sum().clamp_min(1.0)
    sig = (log_sigma.mean(dim=-1) - torch.log1p((y_ret - mu_ret).abs().detach() / vol)).abs()
    total = mag.mean() + s.w_dir * dirn + s.w_sigma * sig.mean()
    stats = {"magnitude": mag.mean().detach(), "direction": dirn.detach(), "sigma": sig.mean().detach(),
             "mae": (y - mu).abs().mean().detach()}
    return total, stats


def _huber_candle_loss(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, dir_logit: torch.Tensor,
                       dir_true: torch.Tensor, _vol: torch.Tensor, weight: torch.Tensor,
                       s: ArStreamSettings) -> tuple:
    """(loss, stats). The v1 robust loss, selectable with ``loss="huber"``: Huber-capped magnitude with the
    adaptive per-step outlier weight (a sudden move is down-weighted, never punished quadratically),
    plain full-weight direction BCE, sigma pulled towards log1p(|error|) detached."""
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


_LOSS_KINDS = {"czar": _czar_candle_loss, "huber": _huber_candle_loss}


def robust_candle_loss(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, dir_logit: torch.Tensor,
                       dir_true: torch.Tensor, vol: torch.Tensor, weight: torch.Tensor,
                       s: ArStreamSettings) -> tuple:
    """(loss, stats). The selected loss kind (``settings.loss``): ``"czar"`` (the default - the Composite
    Zero-Agnostic Return loss, see ``_czar_candle_loss``) or ``"huber"`` (the v1 robust loss, see
    ``_huber_candle_loss``). Both scale tensors are always passed; each kind reads the ones it needs."""
    try:
        kind = _LOSS_KINDS[s.loss]
    except KeyError:
        raise ValueError(f"unknown loss kind {s.loss!r}; known: {sorted(_LOSS_KINDS)}") from None
    return kind(mu, log_sigma, y, dir_logit, dir_true, vol, weight, s)


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


def _train_epoch(model, opt, X, Y, dirs, vol, w, s, p_own, device, epoch: int) -> dict:
    model.train()
    n = len(X)
    sums = {"loss": 0.0, "magnitude": 0.0, "direction": 0.0, "sigma": 0.0}
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(int(s.seed) + epoch))
    for i in range(0, n, s.batch_size):
        rows = idx[i:i + s.batch_size].to(device)
        xb, yb, db = X[rows], Y[rows], dirs[rows]
        mu, ls, dl = _rollout_step(model, xb, yb, p_own)
        loss, stats = robust_candle_loss(mu, ls, yb, dl, db, vol[rows], w[rows], s)
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
    vol = local_vol(X, ret_scale, settings.vol_floor)
    w = step_weights(Y[:, :, 0], ret_scale, settings.outlier_softness)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    model.to(device)
    ramp = max(settings.epochs - 1, 1)
    hist = {"train_loss": [], "magnitude": [], "direction": [], "sigma": [], "val_dir_acc": [], "p_own": []}
    best, best_state, bad = -1.0, None, 0
    for epoch in range(settings.epochs):
        p_own = settings.ss_start + (settings.ss_end - settings.ss_start) * epoch / ramp
        row = _train_epoch(model, opt, X, Y, dirs, vol, w, settings, p_own, device, epoch)
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
