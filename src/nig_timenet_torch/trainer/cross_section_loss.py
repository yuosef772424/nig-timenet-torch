"""PURPOSE: The cross-section loss term of a multi-coin run (D-032, off by default): 1 - the mean over timestamps of
    the Pearson correlation, inside the timestamp, of every NIG mean with its label (kind "ic") or with the label's
    rank inside the timestamp (kind "rank_ic").
TAGS: cross section loss, ic loss, rank ic loss, relative move, timestamp groups, auxiliary loss, multi coin, panel
PITFALLS: needs batches of WHOLE timestamps (``SplitSettings.cross_section``; the trainer raises without the group
    channel); a timestamp with fewer than ``MIN_ROWS`` rows or a constant label is left out; the term is invariant to
    a shift and a positive scale of the mean inside the timestamp, so it never fits the market's own move or the size
    of the mean: the NIG loss still does that. The rank is taken on the LABEL side only (the mean stays
    differentiable).

Why (docs/audit/2026-11-16-panel, reports B and C): 40-60 % of the close label's variance is the move all coins of a
timestamp share; a loss summed over all rows spends most of its gradient on that part, which no input predicts. This
term reads only what a coin does AGAINST its timestamp, the part the day metrics trade.
"""

import torch

from nig_timenet_torch.trainer.tasks import TASK_REGISTRY

__all__ = ["CROSS_SECTION_LOSSES", "cross_section_loss", "group_correlation", "group_ranks", "MIN_ROWS"]

MIN_ROWS = 3


def _groups(gid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, inv = torch.unique(gid, return_inverse=True)
    return inv, torch.bincount(inv)


def _demean(v: torch.Tensor, inv: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    mean = v.new_zeros(len(counts)).index_add_(0, inv, v) / counts.to(v.dtype)
    return v - mean[inv]


def group_ranks(y: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    """The rank of each row's ``y`` inside its group, as (position + 0.5) / rows - 0.5 (in (-0.5, 0.5); ties broken
    by row order)."""
    inv, counts = _groups(gid)
    order = torch.argsort(y, stable=True)
    order = order[torch.argsort(inv[order], stable=True)]              # sorted by group, then by y
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.empty_like(inv)
    pos[order] = torch.arange(len(y), device=y.device) - starts[inv[order]]
    return (pos.to(y.dtype) + 0.5) / counts[inv].to(y.dtype) - 0.5


def group_correlation(a: torch.Tensor, b: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    """Mean over the groups with >= ``MIN_ROWS`` rows and a non-constant ``b`` of the Pearson correlation of ``a``
    with ``b`` inside the group (a constant ``a`` counts as 0); 0 when no group qualifies."""
    inv, counts = _groups(gid)
    da, db = _demean(a, inv, counts), _demean(b, inv, counts)
    zeros = a.new_zeros(len(counts))
    cov, va, vb = (zeros.clone().index_add_(0, inv, v) for v in (da * db, da * da, db * db))
    ok = (counts >= MIN_ROWS) & (vb > 0)
    if not bool(ok.any()):
        return a.sum() * 0.0
    return (cov[ok] / torch.sqrt(va[ok] * vb[ok] + 1e-12)).mean()


def _ic(mu, y, gid):
    return 1.0 - group_correlation(mu, y, gid)


def _rank_ic(mu, y, gid):
    return 1.0 - group_correlation(mu, group_ranks(y, gid), gid)


#: loss ``cross_section`` -> fn(mean (n,), label (n,), group id (n,)) -> scalar (None = no term)
CROSS_SECTION_LOSSES = {"none": None, "ic": _ic, "rank_ic": _rank_ic}


def cross_section_loss(kind: str, outputs: dict, y: dict, targets_cfg: dict, gid: torch.Tensor) -> torch.Tensor:
    """The mean of the ``kind`` term over every evidential (NIG) target of the RESOLVED ``targets_cfg``."""
    fn = CROSS_SECTION_LOSSES[kind]
    terms = [fn(outputs[c["output_keys"]["mu"]].reshape(-1).to(torch.float32),
                y[c["true_key"]].reshape(-1).to(torch.float32), gid)
             for c in targets_cfg.values() if TASK_REGISTRY[c["task_type"]]["fn"] is TASK_REGISTRY["evidential"]["fn"]]
    if not terms:
        raise ValueError("the cross-section loss needs at least one evidential (NIG) target")
    return torch.stack(terms).mean()
