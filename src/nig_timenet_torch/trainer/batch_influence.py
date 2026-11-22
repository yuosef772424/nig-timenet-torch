"""PURPOSE: In-run batch influence during ``Trainer.fit``: the exact change of fixed probe losses (validation, training
    rows) after every step, its split-half check and, every k steps, the first-order g_probe . dtheta by module.
TAGS: batch influence, data attribution, in-run data shapley, tracin, loss change per step, probe set, split half
    reliability, first order, update direction, harmful batches, training monitor
PITFALLS: the probe loss is the trainer's RAW loss with the loss schedules frozen at their START values (so the number
    means the same thing in every epoch and the step deltas telescope: their sum = last probe loss - first); the exact
    delta also holds the BatchNorm moving-statistics update of the batch, the first-order estimate only the parameter
    update; the monitor never draws from torch's RNG and never touches the optimizer, so training is unchanged (tested
    bitwise on CPU); with a group channel the probe holds whole timestamps; group-level TracIn costs one backward per
    group every ``group_every`` steps (keep the groups few: months, or the largest coins).

Method (chosen 2026-11-17, owner task "the effect of each batch"; the candidates are compared in
docs/INFLUENCE.md): the exact one-step change of the probe loss, L_probe(theta_{t+1}) -
L_probe(theta_t), is measured after EVERY step (one forward of the probe per step, since L before step t+1 is L after
step t). It is the ground truth of what that batch's update did to the probe, in the order it really happened
(first-order In-Run Data Shapley / TracIn-Ideal is its linear approximation). The probe is split into two random halves:
the correlation across steps of the two halves' deltas is the split-half reliability (a delta that only one half sees is
probe noise, not a batch effect). With ``first_order_every = k`` every k-th step also stores g_probe(theta_t) .
(theta_{t+1} - theta_t) (the real AdamW update, not the raw gradient, so it is right to first order for any optimizer),
that sum per top-level module (which part of the model the batch moved toward or away from the probe), the cosine
between the probe gradient and the batch gradient, and the update norm. The analysis (reliability, harmful share, which
coins / times hurt, with a permutation null) is ``eval/influence_report.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.trainer.groups import row_groups

__all__ = ["InfluenceSettings", "BatchInfluence", "probe_rows"]

_TAGS = {"loss": "delta", "a": "delta_a", "b": "delta_b"}   # column of a probe measurement's change


@dataclass(frozen=True)
class InfluenceSettings:
    """``probe_rows``: rows per probe (both halves together); ``first_order_every``: 0 = exact deltas only, k = also
    the first-order terms on every k-th step (one probe forward + backward each); ``seed``: the probe draw (numpy,
    never torch's RNG); ``keep_rows``: store every batch's row indices (for the composition analysis);
    ``group_every``: 0 = off, k = on every k-th step the gradient of every row GROUP (``BatchInfluence(groups=...)``,
    e.g. a coin or a month of training rows) is compared with the primary probe's gradient (group-level TracIn: one
    backward per group); ``group_rows``: rows drawn per group for that gradient."""
    probe_rows: int = 512
    first_order_every: int = 0
    seed: int = 0
    keep_rows: bool = True
    group_every: int = 0
    group_rows: int = 256

    def __post_init__(self):
        if self.probe_rows < 4 or self.first_order_every < 0 or self.group_every < 0 or self.group_rows < 4:
            raise ValueError("probe_rows >= 4, group_rows >= 4, first_order_every >= 0 and group_every >= 0")


def probe_rows(n: int, size: int, seed: int, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(half A, half B): two disjoint random row sets of about ``size`` rows together, drawn from ``n`` rows; with
    ``groups`` (a timestamp id per row) whole groups are drawn and the halves split by group."""
    rng = np.random.default_rng(seed)
    if groups is None:
        pick = rng.permutation(n)[:min(size, n)]
        return np.sort(pick[0::2]), np.sort(pick[1::2])
    groups = np.asarray(groups)
    uniq = rng.permutation(np.unique(groups))
    counts = pd.Series(groups).value_counts().reindex(uniq).to_numpy()
    take = uniq[:max(2, int(np.searchsorted(np.cumsum(counts), size)) + 1)]
    halves = (take[0::2], take[1::2])
    return tuple(np.flatnonzero(np.isin(groups, h)) for h in halves)


class _Probe:
    """One named probe: its two halves on the model's device (rows contiguous by group when grouped)."""

    def __init__(self, name: str, x: torch.Tensor, y: dict, size: int, seed: int, grouped: bool):
        groups = row_groups(x).numpy() if grouped else None
        self.name, self.grouped = name, grouped
        self.halves = []
        for rows in probe_rows(len(x), size, seed, groups):
            rows = torch.as_tensor(rows, dtype=torch.long)
            self.halves.append((x[rows], {k: v[rows] for k, v in y.items()}))
        if min(len(h[0]) for h in self.halves) < 2:
            raise ValueError(f"probe {name!r}: each half needs >= 2 rows")

    def to(self, dev: torch.device) -> "_Probe":
        self.halves = [(x.to(dev), {k: v.to(dev) for k, v in y.items()}) for x, y in self.halves]
        return self

    def losses(self, trainer, grad: bool = False):
        """([raw loss of half A, of half B], [per-target dict of A, of B]) in eval mode; tensors keep their graph with
        ``grad``. The schedules are the trainer's frozen START values (set by the caller)."""
        raws, per = [], []
        for x, y in self.halves:
            with torch.set_grad_enabled(grad):
                out = trainer.model(x)
                _, raw, per_target, _ = trainer.compute_losses(
                    None, y, outputs=out, groups=row_groups(x) if self.grouped else None)
            raws.append(raw)
            per.append(per_target)
        return raws, per


class BatchInfluence:
    """The monitor ``Trainer.fit(..., batch_monitor=...)`` calls: ``start`` once, then ``before_step`` / ``after_step``
    around every optimizer step. ``probes``: {name: (x, y)} (inputs and labels as the trainer gets them); the FIRST
    probe is the primary one (first-order terms, the report's headline), normally validation rows."""

    def __init__(self, probes: dict, settings: InfluenceSettings | None = None, *, group_channel: bool = False,
                 groups: dict | None = None):
        if not probes:
            raise ValueError("at least one probe (name -> (x, y)) is needed")
        self.settings = s = settings or InfluenceSettings()
        self.probes = [_Probe(n, x, y, s.probe_rows, s.seed + salt, group_channel)
                       for salt, (n, (x, y)) in enumerate(probes.items())]
        #: ``groups``: {name: (x, y)} training rows of each group (group-level TracIn on every group_every-th step)
        self.groups = [_Probe(str(n), x, y, s.group_rows, s.seed + 1000 + i, group_channel)
                       for i, (n, (x, y)) in enumerate((groups or {}).items())]
        if s.group_every and not self.groups:
            raise ValueError("group_every needs groups={name: (x, y)} of training rows")
        self.group_records: list[dict] = []
        self.records: list[dict] = []
        self.batch_rows: list[np.ndarray] = []
        self._last: dict | None = None     # probe losses after the last step (= before the next one)
        self._fo: dict | None = None       # first-order state of the current step
        self._vars: dict | None = None
        self._step = 0

    # ------------------------------------------------------------------------------------------------ measuring
    def _measure(self, trainer) -> dict:
        """{probe: {"a", "b", "loss", <target>...}} of the current weights (no grad)."""
        res = {}
        live = trainer.scheduled_vars
        trainer.scheduled_vars = self._vars
        try:
            for p in self.probes:
                raws, per = p.losses(trainer)
                row = {"a": float(raws[0]), "b": float(raws[1])}
                row["loss"] = 0.5 * (row["a"] + row["b"])
                row.update({t: 0.5 * float(per[0][t] + per[1][t]) for t in per[0]})
                res[p.name] = row
        finally:
            trainer.scheduled_vars = live
        return res

    def start(self, trainer) -> None:
        """Fix the probe's loss definition (the schedules' start values) and measure the starting point."""
        dev = trainer.device
        self.probes = [p.to(dev) for p in self.probes]
        self.groups = [g.to(dev) for g in self.groups]
        self._vars = {n: float(s.start) for n, s in trainer.schedules.items()}
        was = trainer.model.training
        trainer.model.eval()
        self._last = self._measure(trainer)
        self.start_losses = {k: v["loss"] for k, v in self._last.items()}
        trainer.model.train(was)

    def _due(self, k: int) -> bool:
        return k > 0 and self._step % k == 0

    def _grad(self, trainer, probe: _Probe, named) -> list[torch.Tensor]:
        """The gradient of ``probe``'s loss (mean of its halves) on the parameters ``named``, eval mode, schedules
        frozen; the parameters' .grad is left empty."""
        trainer.optimizer.zero_grad(set_to_none=True)
        live = trainer.scheduled_vars
        trainer.scheduled_vars = self._vars
        try:   # eval-mode backward: cuDNN RNNs refuse it on the GPU (PRs #21-#23), so cuDNN is off for this pass
            with torch.backends.cudnn.flags(enabled=False):
                raws, _ = probe.losses(trainer, grad=True)
                (0.5 * (raws[0] + raws[1])).backward()
        finally:
            trainer.scheduled_vars = live
        grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for _, p in named]
        trainer.optimizer.zero_grad(set_to_none=True)
        return grads

    def _group_alignment(self, trainer, named, g_probe: list[torch.Tensor]) -> None:
        """Group-level TracIn: per group, g_probe . g_group (> 0 = a descent step on the group lowers the probe
        loss: the group helps) and their cosine."""
        gp_sq = sum(float((g * g).sum()) for g in g_probe)
        for grp in self.groups:
            gg = self._grad(trainer, grp, named)
            dot = sum(float((a * b).sum()) for a, b in zip(g_probe, gg, strict=True))
            gg_sq = sum(float((b * b).sum()) for b in gg)
            den = (gp_sq * gg_sq) ** 0.5
            self.group_records.append({"epoch": self._cur["epoch"], "step": self._step, "group": grp.name,
                                       "dot": dot, "cos": dot / den if den > 0 else float("nan")})

    def before_step(self, trainer, epoch: int, ix: torch.Tensor) -> None:
        """Remember the batch; on a first-order step, the primary probe's gradient and the weights before the step."""
        if self._last is None:
            raise RuntimeError("BatchInfluence.start(trainer) was not called")
        self._cur = {"epoch": int(epoch) + 1, "step": self._step, "n_rows": int(len(ix))}
        if self.settings.keep_rows:
            self.batch_rows.append(np.asarray(ix, dtype=np.int64))
        self._fo = None
        fo_due, group_due = self._due(self.settings.first_order_every), self._due(self.settings.group_every)
        if not (fo_due or group_due):
            return
        named = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
        trainer.model.eval()
        g_probe = self._grad(trainer, self.probes[0], named)
        if group_due:
            self._group_alignment(trainer, named, g_probe)
        if fo_due:
            self._fo = {"names": [n for n, _ in named], "grad": g_probe,
                        "before": [p.detach().clone() for _, p in named]}

    def after_step(self, trainer, logs: dict) -> None:
        """Measure every probe after the step; store the exact deltas (and the first-order terms when due)."""
        params = dict(trainer.model.named_parameters())
        rec = dict(self._cur)
        if self._fo is not None:
            rec.update(self._first_order(params))
        trainer.model.eval()
        now = self._measure(trainer)
        trainer.model.train()
        rec["train_loss"] = float(logs["raw_loss"])
        for name, row in now.items():
            before = self._last[name]
            pre = "" if name == self.probes[0].name else f"{name}_"
            rec[f"{pre}probe_before"] = before["loss"]
            for key in row:
                tag = _TAGS[key] if key in _TAGS else f"delta_{key}"
                rec[f"{pre}{tag}"] = row[key] - before[key]
        self.records.append(rec)
        self._last = now
        self._step += 1

    def _first_order(self, params: dict) -> dict:
        fo = self._fo
        out, pred, gtrain_dot, gtrain_sq, gprobe_sq, upd_sq = {}, 0.0, 0.0, 0.0, 0.0, 0.0
        per_module: dict[str, float] = {}
        for name, g, before in zip(fo["names"], fo["grad"], fo["before"], strict=True):
            p = params[name]
            d = p.detach() - before
            term = float((g * d).sum())
            pred += term
            module = name.split(".")[0]
            per_module[module] = per_module[module] + term if module in per_module else term
            upd_sq += float((d * d).sum())
            gprobe_sq += float((g * g).sum())
            if p.grad is not None:   # the batch's own gradient (still on the parameters after the step)
                gtrain_dot += float((g * p.grad).sum())
                gtrain_sq += float((p.grad * p.grad).sum())
        den = (gprobe_sq * gtrain_sq) ** 0.5
        out["fo_delta"] = pred
        out["grad_alignment"] = gtrain_dot / den if den > 0 else float("nan")
        out["update_norm"] = upd_sq ** 0.5
        out.update({f"fo_{m}": v for m, v in per_module.items()})
        return out

    # ----------------------------------------------------------------------------------------------- outputs
    def frame(self) -> pd.DataFrame:
        """One row per optimizer step (columns: epoch, step, n_rows, train_loss, probe_before, delta, delta_a,
        delta_b, delta_<target>, <probe>_delta..., and fo_* / grad_alignment / update_norm on first-order steps)."""
        return pd.DataFrame(self.records)

    def group_frame(self) -> pd.DataFrame:
        """Group-level TracIn: one row per (group step, group): epoch, step, group, dot, cos."""
        return pd.DataFrame(self.group_records, columns=["epoch", "step", "group", "dot", "cos"])

    def rows_of_steps(self) -> tuple[np.ndarray, np.ndarray]:
        """(step id per stored row index, row indices into the training rows): every batch's composition."""
        if not self.batch_rows:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        steps = np.concatenate([np.full(len(r), i, dtype=np.int64) for i, r in enumerate(self.batch_rows)])
        return steps, np.concatenate(self.batch_rows)

    def save(self, out) -> None:
        """``batch_influence.csv`` (the frame), ``batch_rows.npz`` (step, row) and, with groups,
        ``group_alignment.csv`` under the folder ``out``."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(out / "batch_influence.csv", index=False)
        steps, rows = self.rows_of_steps()
        np.savez_compressed(out / "batch_rows.npz", step=steps, row=rows)
        if self.group_records:
            self.group_frame().to_csv(out / "group_alignment.csv", index=False)
