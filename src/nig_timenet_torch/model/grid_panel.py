"""The cross-asset model with a two-dimensional interface: days x coin slots in, days x coin slots out.

``PanelModel`` (model/panel.py) takes M coin rows tagged with (day, pos) and returns (M, K); callers had to flatten
the day x slot grid before predicting and scatter the answer back. ``GridPanelModel`` is the adapter that removes
that: one batch is ``{"x": (B, N, T, F) | {tf: (B, N, T_tf, F_tf)}, "mask": (B, N)}`` (B days, N slots, ``mask`` True
for a real coin) and the output is ``{"logit": (B, N, K), "mu": (B, N, K), "mask": (B, N)}``, so targets, losses and
metrics keep the same (days, slots) shape end to end. Padding never reaches the shared encoder (only real rows are
encoded, so batch statistics and compute are not spent on padding) and never acts as an attention key.
"""

import torch
from torch import nn

from nig_timenet_torch.model.panel import PanelModel

__all__ = ["GridPanelModel"]


class GridPanelModel(nn.Module):
    def __init__(self, core: PanelModel):
        super().__init__()
        self.core = core

    @property
    def targets(self) -> tuple[str, ...]:
        return self.core.targets

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x, mask = batch["x"], torch.as_tensor(batch["mask"]).bool()
        rows = {k: v[mask] for k, v in x.items()} if isinstance(x, dict) else x[mask]
        if not bool(mask.any()):
            raise ValueError("a panel batch needs at least one real coin (mask is all False)")
        h = self.core.encode(rows)
        if hasattr(self.core, "enc_proj"):
            h = self.core.enc_proj(h)
        grid = h.new_zeros((*mask.shape, self.core.d_model)).masked_scatter(mask[..., None], h)
        if self.core.n_cross_layers:
            grid = self.core.mix(grid, mask)
        return {**self.core.heads(grid), "mask": mask}
