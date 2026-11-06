"""Training the PANEL autoregressive candle transformer (``model/panel_ar_candle.py``).

Same recipe as the single-coin ``trainer/ar_stream.py`` - candle-by-candle prediction over the horizon,
scheduled sampling on DETACHED own predictions, Huber-capped magnitude loss with the adaptive
outlier weight (a sudden market move is down-weighted, never allowed to explode the gradient) and a
full-weight direction term - applied to a (windows, coins) grid: every step predicts ALL coins' next
candle and appends each coin's own prediction as its next input. Selection is the validation direction
accuracy of the panel-wide rollout.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.panel_ar_candle import PanelArCandleTransformer, rollout
from nig_timenet_torch.trainer.ar_stream import ArStreamSettings, robust_candle_loss, step_weights

__all__ = ["PanelArStreamSettings", "build_panel_ar_stream_model", "fit_panel_ar_stream",
           "panel_rollout_direction_accuracy"]


@dataclass(frozen=True)
class PanelArStreamSettings(ArStreamSettings):
    """The panel run's knobs: the single-coin knobs plus the cross-coin attention depth."""
    cross_layers: int = 1


def build_panel_ar_stream_model(settings: PanelArStreamSettings, n_features: int, n_coins: int) \
        -> PanelArCandleTransformer:
    """The panel model of this run, from the settings' architecture fields (transformer or gru trunk)."""
    return PanelArCandleTransformer(n_features=n_features, n_coins=n_coins,
                                    context_len=settings.context_len, horizon_len=settings.horizon_len,
                                    d_model=settings.d_model, num_layers=settings.num_layers,
                                    num_heads=settings.num_heads, num_kv_heads=settings.num_kv_heads,
                                    cross_layers=settings.cross_layers, dropout=settings.dropout,
                                    encoder=settings.encoder)


def panel_rollout_direction_accuracy(model: PanelArCandleTransformer, X: torch.Tensor, dirs: torch.Tensor,
                                     device: str = "cpu", batch_size: int = 256) -> float:
    """Direction accuracy of the panel-wide rollout, all coins pooled: the share of (window, coin, step)
    triples whose predicted next-candle direction matches the true sign (flat true moves excluded)."""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb, db = X[i:i + batch_size].to(device), dirs[i:i + batch_size].to(device)
            out = rollout(model, xb)
            up = torch.sign(out["mu"][:, :, :, 0]) > 0
            keep = db != 0
            hits += int(((up == (db > 0)) & keep).sum())
            total += int(keep.sum())
    return hits / max(total, 1)


def _panel_rollout_step(model, xb, yb, p_own):
    """One panel batch the deployment way: every horizon step predicts all coins' next candle, scores it
    against the truth, then appends each coin's own prediction (probability ``p_own`` per (window, coin),
    DETACHED) or the true candle as that coin's next input."""
    b, n, h = xb.shape[0], xb.shape[1], yb.shape[2]
    seq = xb
    mus, log_sigmas, dir_logits = [], [], []
    own = torch.rand(b, n, device=xb.device) < p_own
    for t in range(h):
        out = model(seq)
        mus.append(out["mu"][:, :, -1])
        log_sigmas.append(out["log_sigma"][:, :, -1])
        dir_logits.append(out["dir_logit"][:, :, -1])
        mu_t = out["mu"][:, :, -1].detach()
        nxt = torch.where(own[:, :, None], mu_t, yb[:, :, t])
        seq = torch.cat([seq, nxt[:, :, None]], dim=2)
    return (torch.stack(mus, dim=2), torch.stack(log_sigmas, dim=2), torch.stack(dir_logits, dim=2))


def _flatten(mu, ls, dl, yb, db, wb):
    """(B, N, ...) -> (B*N, ...) so the single-coin loss scores every coin's steps alike; 4-D (B, N, H, F)
    becomes (B*N, H, F) and 3-D (B, N, H) becomes (B*N, H), over the SAME (window, coin) order."""
    flat = lambda a: (a.reshape(-1, a.shape[-2], a.shape[-1]) if a.ndim == 4  # noqa: E731
                     else a.reshape(-1, a.shape[-1]))
    return flat(mu), flat(ls), flat(dl), flat(yb), flat(db), flat(wb)


def _train_epoch(model, opt, X, Y, dirs, w, s, p_own, device, epoch) -> dict:
    model.train()
    n = len(X)
    sums = {"loss": 0.0, "magnitude": 0.0, "direction": 0.0, "sigma": 0.0}
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(int(s.seed) + epoch))
    for i in range(0, n, s.batch_size):
        rows = idx[i:i + s.batch_size].to(device)
        mu, ls, dl = _panel_rollout_step(model, X[rows], Y[rows], p_own)
        fmu, fls, fdl, fy, fd, fw = _flatten(mu, ls, dl, Y[rows], dirs[rows], w[rows])
        loss, stats = robust_candle_loss(fmu, fls, fy, fdl, fd, fw, s)
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


def fit_panel_ar_stream(model: PanelArCandleTransformer, X: np.ndarray, Y: np.ndarray, dirs: np.ndarray,
                        vX: np.ndarray, vY: np.ndarray, vdirs: np.ndarray,
                        settings: PanelArStreamSettings) -> dict:
    """Train the panel grid (X, Y, dirs are (windows, coins, ...)), select on the validation panel-rollout
    direction accuracy, early stop after ``patience`` epochs without improvement, reload the best weights."""
    if X.shape[2] != settings.context_len or Y.shape[2] < settings.horizon_len:
        raise ValueError(f"panel windows {X.shape[2]}+{Y.shape[2]} do not fit settings "
                         f"{settings.context_len}+{settings.horizon_len}")
    Y, vY, dirs, vdirs = (a[:, :, :settings.horizon_len] for a in (Y, vY, dirs, vdirs))
    seed_everything(settings.seed)
    device = settings.device
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    X, Y, dirs, vX, vY, vdirs = t(X), t(Y), t(dirs), t(vX), t(vY), t(vdirs)
    ret_scale = float(torch.median(Y[:, :, :, 0].abs()))
    w = step_weights(Y[:, :, :, 0], ret_scale, settings.outlier_softness)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    model.to(device)
    ramp = max(settings.epochs - 1, 1)
    hist = {"train_loss": [], "magnitude": [], "direction": [], "sigma": [], "val_dir_acc": [], "p_own": []}
    best, best_state, bad = -1.0, None, 0
    for epoch in range(settings.epochs):
        p_own = settings.ss_start + (settings.ss_end - settings.ss_start) * epoch / ramp
        row = _train_epoch(model, opt, X, Y, dirs, w, settings, p_own, device, epoch)
        acc = panel_rollout_direction_accuracy(model, vX, vdirs, device)
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
