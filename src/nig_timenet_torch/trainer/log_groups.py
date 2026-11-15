"""PURPOSE: Device-side training logs and their grouped read-back (Keras steps_per_execution): the logs of N steps stay
    0-d tensors and are moved to the host together, one sync per group instead of one per step.
TAGS: steps per execution, training logs, host sync, grouped logs, epoch means, tensor logs
PITFALLS: speed only: the optimizer steps are not touched, so the weights equal those of one step per execution; the
    epoch mean is weighted by batch size exactly as the per-step version; ``flush`` must be called after the last
    step (a partial group is read like a full one).
"""

from collections import defaultdict

import torch

__all__ = ["tensor_logs", "GroupedLogs"]


def tensor_logs(total, raw, per_target, stats) -> dict[str, torch.Tensor]:
    """The scalar logs of one step as detached 0-d tensors (same keys as ``Trainer._logs``)."""
    return {"total_loss": total.detach(), "raw_loss": raw.detach(),
            **{f"{t}_loss": v.detach() for t, v in per_target.items()}, **{k: v.detach() for k, v in stats.items()}}


class GroupedLogs:
    """Batch-size weighted sums of the logs of one epoch, read from the device ``group`` steps at a time."""

    def __init__(self, group: int):
        self.group, self.sums = group, defaultdict(float)
        self._pending: list[tuple[int, dict[str, torch.Tensor]]] = []

    def add(self, n_rows: int, logs: dict[str, torch.Tensor]) -> None:
        self._pending.append((n_rows, logs))
        if len(self._pending) >= self.group:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        for k in self._pending[0][1]:
            vals = torch.stack([logs[k].float() for _, logs in self._pending]).tolist()   # the group's one sync
            for (n, _), v in zip(self._pending, vals, strict=True):
                self.sums[k] += v * n
        self._pending.clear()

    def means(self, n_rows: int) -> dict[str, float]:
        self.flush()
        return {k: v / n_rows for k, v in self.sums.items()}
