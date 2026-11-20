"""PURPOSE: Day-balanced, failure-weighted training loss (owner 2026-11-20, off by default): every UTC day of the
    training rows carries the same total weight, and a day whose loss is high (the model fails on it) is weighted up.
TAGS: day weights, sample weights, group dro, hard days, failure weighting, balanced days, robust training, loss
    reweighting, generalisation over days
PITFALLS: the weights act through the task losses' weighted mean INSIDE a batch (``tasks._wmean``), so they only
    matter when a batch mixes days: ``make_day_weighter`` refuses grouped batches (``batch_by`` != "rows", a
    ``batch_order`` that batches timestamps, the cross-section group channel), where every row of a batch shares
    one day and the weights would cancel. The day loss is measured in eval mode at the START of an epoch (epoch 0:
    no measurement yet, balance only) on the model of the end of the previous epoch; the per-row loss comes from the
    gradient of the weighted loss with respect to the row weights (exact for every term that is a weighted mean;
    the NIG calibration term's diversity penalty is not, so there it is an influence approximation). A day with a
    high loss can be a day the model cannot learn (pure noise); ``max_ratio`` and ``shrink_rows`` keep such days
    from dominating. Validation and the selection metric stay unweighted (rule C3).

Weights (one measurement, days d with n_d training rows):
    L_d    mean per-row loss of day d, shrunk to the mean over all rows by ``shrink_rows`` pseudo-rows
    E_d    exponential moving average of L_d over the measurements (``momentum``; the first one is L_d)
    z_d    (E_d - median(E)) / (1.4826 MAD(E)), clipped to [-3, 3]   (robust: one crash day does not set the scale)
    q_d    exp(strength * z_d), divided by its mean over days            (strength 0 = no failure weighting)
    w_i    q_d(i) * N / (D * n_d(i)) with ``balance`` (every day the same total), else q_d(i); then the row weights
           are divided by their mean and clipped to [1 / max_ratio, max_ratio] (a few rounds; the mean ends near 1,
           and only ratios matter: the task losses divide by the batch's weight sum).
"""

from dataclasses import dataclass

import torch

__all__ = ["DayWeightSettings", "DayWeighter", "make_day_weighter", "row_losses", "day_weights_from_losses"]

_MAD_TO_SD = 1.4826
_Z_CLIP = 3.0


@dataclass(frozen=True)
class DayWeightSettings:
    """``FitSettings.day_weights`` (None = every row weight 1, as before)."""
    balance: bool = True        # every UTC day carries the same total weight (rows of busy days count less)
    strength: float = 1.0       # exponent on the robust z-score of the day loss (0 = balance only)
    momentum: float = 0.5       # EMA of the day loss across measurements (0 = the last measurement only)
    max_ratio: float = 5.0      # a row weight stays within [1 / max_ratio, max_ratio] x the mean row weight
    shrink_rows: int = 20       # pseudo-rows at the overall mean loss added to every day (small days are noisy)
    every: int = 1              # measure the day losses at the start of every ``every``-th epoch (epoch >= 1)
    measure_chunk: int = 8192   # rows per forward pass of the measurement

    def __post_init__(self):
        if self.strength < 0 or not 0.0 <= self.momentum < 1.0:
            raise ValueError("day_weights.strength must be >= 0 and momentum in [0, 1)")
        if not self.max_ratio >= 1.0:
            raise ValueError(f"day_weights.max_ratio must be >= 1, got {self.max_ratio!r}")
        for name in ("shrink_rows", "every", "measure_chunk"):
            v = getattr(self, name)
            low = 0 if name == "shrink_rows" else 1
            if isinstance(v, bool) or not isinstance(v, int) or v < low:
                raise ValueError(f"day_weights.{name} must be an integer >= {low}, got {v!r}")


def row_losses(trainer, x, y, chunk: int) -> torch.Tensor:
    """The per-row value of the trainer's plain objective (NIG / class losses with their loss weights and Kendall
    factors; no cross-section / direction term), up to ONE constant shared by every row (the Kendall log-variance
    terms and the L2 term), in eval mode, float64 on the CPU.

    With row weights w (all 1) the objective is sum_t c_t sum_i w_i l_ti / sum_i w_i + K, so
    n * d(objective)/d(w_i) + objective = sum_t c_t l_ti + K: the loss of row i alone."""
    model, dev = trainer.model, trainer._device()
    was_training = model.training
    model.eval()
    out = []
    try:
        for s in range(0, len(x), chunk):
            xb = x[s:s + chunk].to(dev)
            yb = {k: v[s:s + chunk].to(dev) for k, v in y.items()}
            with torch.no_grad():
                outputs = model(xb)
            w = torch.ones(len(xb), device=dev, dtype=torch.float32, requires_grad=True)
            with torch.enable_grad():
                total = trainer.compute_losses(None, yb, sample_weight=w, outputs=outputs, plain=True)[0]
                (grad,) = torch.autograd.grad(total, w)
            out.append((grad.double() * len(xb) + total.detach().double()).cpu())
    finally:
        model.train(was_training)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


def _robust_z(e: torch.Tensor) -> torch.Tensor:
    med = e.median()
    mad = (e - med).abs().median() * _MAD_TO_SD
    if not mad > 0:
        mad = e.std(unbiased=False) if len(e) > 1 and e.std(unbiased=False) > 0 else torch.ones((), dtype=e.dtype)
    return ((e - med) / mad).clamp(-_Z_CLIP, _Z_CLIP)


def day_weights_from_losses(ema: torch.Tensor | None, counts: torch.Tensor, inverse: torch.Tensor,
                            cfg: DayWeightSettings) -> torch.Tensor:
    """Row weights (float32, mean ~1; the scale cancels in the batch's weighted mean) from the per-day loss EMA
    (None = not measured yet: balance only) and the day of every row (``inverse``: index into ``counts``)."""
    n, n_days = len(inverse), len(counts)
    q = torch.ones(n_days, dtype=torch.float64)
    if ema is not None and cfg.strength > 0:
        q = torch.exp(cfg.strength * _robust_z(ema))
        q = q / q.mean()
    w = q[inverse]
    if cfg.balance:
        w = w * (n / (n_days * counts.double()))[inverse]
    w = w / w.mean()
    for _ in range(5):  # clip and renormalise a few rounds, then a last clip: the bounds hold, the mean is ~1
        w = w.clamp(1.0 / cfg.max_ratio, cfg.max_ratio)
        w = w / w.mean()
    return w.clamp(1.0 / cfg.max_ratio, cfg.max_ratio).to(torch.float32)


class DayWeighter:
    """The per-row weights of one ``fit``: the day of every training row, the per-day loss EMA (checkpointed) and the
    weights of the current epoch."""

    def __init__(self, cfg: DayWeightSettings, day_ids):
        days = torch.as_tensor(day_ids, dtype=torch.int64).reshape(-1)
        self.cfg = cfg
        self.days, self.inverse, self.counts = torch.unique(days, return_inverse=True, return_counts=True)
        self.ema: torch.Tensor | None = None       # float64 per day; None until the first measurement
        self.last: torch.Tensor | None = None      # the last measured (shrunk) day losses, for the table
        self.weights = day_weights_from_losses(None, self.counts, self.inverse, cfg)

    def epoch_weights(self, trainer, x, y, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Measure the day losses when ``epoch`` is due (epoch >= 1, every ``cfg.every``) and return the row weights
        of this epoch plus their summary logs."""
        if epoch >= 1 and epoch % self.cfg.every == 0:
            self.update(row_losses(trainer, x, y, self.cfg.measure_chunk))
        return self.weights, self.summary()

    def update(self, losses: torch.Tensor) -> None:
        """Fold one measurement (per-row losses aligned to the training rows) into the EMA and the weights."""
        losses = torch.as_tensor(losses, dtype=torch.float64).reshape(-1)
        if len(losses) != len(self.inverse):
            raise ValueError(f"{len(losses)} row losses for {len(self.inverse)} training rows")
        sums = torch.zeros(len(self.counts), dtype=torch.float64).index_add_(0, self.inverse, losses)
        k = float(self.cfg.shrink_rows)
        day = (sums + k * losses.mean()) / (self.counts.double() + k)
        m = self.cfg.momentum
        self.ema = day if self.ema is None else m * self.ema + (1.0 - m) * day
        self.last = day
        self.weights = day_weights_from_losses(self.ema, self.counts, self.inverse, self.cfg)

    def day_totals(self) -> torch.Tensor:
        """The total weight of every day (sum of its row weights; float64)."""
        return torch.zeros(len(self.counts), dtype=torch.float64).index_add_(0, self.inverse, self.weights.double())

    def summary(self) -> dict[str, float]:
        """Log-only: the largest row weight (mean 1), the share of the total weight on the heaviest 10 % of days, and
        the effective number of days as a share of the days."""
        tot = self.day_totals()
        top = max(1, int(round(0.1 * len(tot))))
        ess = float(tot.sum() ** 2 / (tot.square().sum() * len(tot)))
        return {"day_w_max": float(self.weights.max()), "day_w_top10_share": float(tot.topk(top).values.sum()
                                                                                    / tot.sum()),
                "day_w_ess_share": ess}

    def table(self) -> list[dict]:
        """One record per day: day (UTC day number), rows, the last measured day loss, its EMA, the day's total
        weight relative to an equal share (1 = the weight every day would have with balance and strength 0)."""
        tot = self.day_totals()
        rel = tot / tot.mean()
        ema = [None] * len(tot) if self.ema is None else self.ema.tolist()
        last = [None] * len(tot) if self.last is None else self.last.tolist()
        return [{"day": int(d), "rows": int(c), "loss_last": lo, "loss_ema": e, "weight_rel": float(r)}
                for d, c, lo, e, r in zip(self.days.tolist(), self.counts.tolist(), last, ema, rel.tolist(),
                                          strict=True)]

    def state_dict(self) -> dict:
        return {"ema": self.ema, "last": self.last}

    def load_state_dict(self, state: dict) -> None:
        self.ema, self.last = state["ema"], state["last"]
        if self.ema is not None and len(self.ema) != len(self.counts):
            raise ValueError(f"the checkpoint's day weights hold {len(self.ema)} days, the training rows "
                             f"{len(self.counts)}")
        self.weights = day_weights_from_losses(self.ema, self.counts, self.inverse, self.cfg)


def make_day_weighter(cfg: DayWeightSettings | None, day_ids, n_rows: int, *, grouped_batches: bool
                      ) -> DayWeighter | None:
    """The ``DayWeighter`` of a fit (None when ``cfg`` is None). ``grouped_batches``: the fit batches whole groups
    (timestamps / days), where the in-batch weighted mean would cancel the weights: refused."""
    if cfg is None:
        if day_ids is not None:
            raise ValueError("day_groups is given but FitSettings.day_weights is None")
        return None
    if grouped_batches:
        raise ValueError("day_weights needs batches of shuffled rows (batch_by 'rows', batch_order 'rows', no "
                         "cross-section group channel): inside a batch of one day / timestamp the weights cancel")
    if day_ids is None:
        raise ValueError("FitSettings.day_weights needs day_groups (the UTC day of every training row)")
    weighter = DayWeighter(cfg, day_ids)
    if len(weighter.inverse) != n_rows:
        raise ValueError(f"day_groups has {len(weighter.inverse)} ids for {n_rows} training rows")
    return weighter
