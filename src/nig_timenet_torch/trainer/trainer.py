"""PURPOSE: Trainer: loss weighting (loss_weight, Kendall for the non-excluded task types), train / eval steps and the
    ``fit`` epoch loop; its config is validated by ``trainer.targets`` first and its schedules are parsed once.
TAGS: trainer, fit, compute_losses, kendall uncertainty weighting, scheduled vars, selection metric, resume, config
PITFALLS: ``Trainer.config`` is the dict as given (it is hashed into the RunIdentity); the resolved copies are
    ``targets_cfg`` and ``schedules``. Kendall weights ONLY the task types NOT in
    ``uncertainty_weighting_exclude_task_types`` (default: evidential); the excluded ones are added unweighted.
    A selection metric that is not ``selectable`` (the high/low direction AUCs) raises at construction.

Trainer + compute_losses (source: crypto-signal-prediction trainer/trainer.py @ aeafc564, GenericTrainer
__init__ / _compute_losses / train_step / test_step; BUILD_PLAN.md P18, P22). ``Trainer.fit`` (ROADMAP T1) adds the
epoch loop; its pieces live in schedules.py, tracker.py and checkpoint.py.

L2 kernel regularisation: torch has no per-layer kernel_regularizer, so the model exposes
``regularization_loss()`` (``l2 * sum(w^2)``, no 0.5 factor, NOTES.md #6) and compute_losses adds it once to both
total and raw loss, as the TF trainer adds ``model.losses``.
"""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.cross_section_loss import CROSS_SECTION_LOSSES, cross_section_loss
from nig_timenet_torch.trainer.direction_balance import direction_balance_loss
from nig_timenet_torch.trainer.groups import group_batches, group_chunks, row_groups
from nig_timenet_torch.trainer.identity import RunIdentity, SeedMismatchError, check_resume
from nig_timenet_torch.trainer.log_groups import GroupedLogs, tensor_logs
from nig_timenet_torch.trainer.loss_balance import make_balancer
from nig_timenet_torch.trainer.schedules import LrSettings, ParamSchedule, apply_schedules, lr_at
from nig_timenet_torch.trainer.selection_metric import (
    GROUP_KEY,
    SELECTION_METRICS,
    check_selectable,
    compute_selection_metric,
)
from nig_timenet_torch.trainer.switches import INITS, EmaWeights, apply_init
from nig_timenet_torch.trainer.targets import resolve_loss, resolve_targets, validate_trainer_config
from nig_timenet_torch.trainer.task_weighting import UncertaintyWeightedLoss
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY
from nig_timenet_torch.trainer.trace import epoch_trace, epoch_trace_mean
from nig_timenet_torch.trainer.tracker import BestTracker, EarlyStopSettings

__all__ = ["Trainer", "compute_losses", "weight_for_target", "FitSettings", "FitResult"]


@dataclass(frozen=True)
class FitSettings:
    """What ``Trainer.fit`` needs besides the data. ``epochs`` has no default (and is not part of the RunIdentity
    hash, so a resumed run may be given more epochs); ``lr=None`` keeps the optimizer's own learning rate."""
    epochs: int
    batch_size: int = 64
    seed: int = 0
    early_stop: EarlyStopSettings = field(default_factory=EarlyStopSettings)
    lr: LrSettings | None = None
    trace: bool = False            # T4: add the trainer trace (trainer/trace.py) to every history row; log-only
    # D-021 switches (TR1); the defaults reproduce the trainer before D-021 and are left out of the run identity
    grad_clip_norm: float | None = None   # clip the global gradient norm per step (TF used 1.0, B-018)
    ema_decay: float | None = None        # evaluate / select / keep an EMA copy of the weights (TF use_ema, B-018)
    init: str = "torch"                   # trainer.switches.INITS, applied at the start of a fresh fit (B-007)
    # Keras ``steps_per_execution``: the logs of this many consecutive steps are read back from the device together
    # (one host sync per group instead of one per step). Speed only: the updates are the same ones, in the same order.
    steps_per_execution: int = 1

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if isinstance(self.steps_per_execution, bool) or not isinstance(self.steps_per_execution, int) \
                or self.steps_per_execution < 1:
            raise ValueError(f"steps_per_execution must be an integer >= 1, got {self.steps_per_execution!r}")
        if self.grad_clip_norm is not None and not self.grad_clip_norm > 0:
            raise ValueError("grad_clip_norm must be None or > 0")
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be None or in (0, 1)")
        INITS.entry(self.init)  # unknown init names raise now


@dataclass
class FitResult:
    history: list[dict]            # one dict per finished epoch: epoch, lr, train_*, val_*, selection_metric
    epochs_done: int
    best_epoch: int | None
    best_value: float | None       # the smoothed selection metric at best_epoch
    stopped_early: bool
    restored_best: bool            # the model now holds the weights of best_epoch


def weight_for_target(sample_weight, target_name: str, tcfg: dict):
    """sample_weight is one tensor for every target, or a mapping keyed by target name or true_key."""
    if sample_weight is None:
        return None
    if isinstance(sample_weight, Mapping):
        for key in (target_name, tcfg["true_key"]):
            if key in sample_weight:
                return sample_weight[key]
        return None
    return sample_weight


def compute_losses(model: torch.nn.Module, x, y, targets_cfg: dict, scheduled_vars: dict[str, float],
                   loss_weights: dict[str, float], uncertainty_layer: UncertaintyWeightedLoss | None,
                   kendall_task_names: list[str], fixed_task_names: list[str], sample_weight=None,
                   training: bool = False, outputs: dict | None = None, multipliers: dict | None = None):
    """-> (total_loss, raw_loss, per_target_losses, per_stat_values, per_constraint_values).
    raw_loss = sum of loss * loss_weight (no Kendall terms) + regularisation; total applies Kendall to
    kendall_task_names and adds fixed_task_names unweighted. ``training`` only gates flooding (the model's own
    train()/eval() mode gates dropout). ``outputs``: precomputed model(x) (then x is not used). ``targets_cfg`` must be
    RESOLVED (``targets.resolve_targets``: base and task defaults filled). ``multipliers``: the loss balancer's
    per-target factors (FX2, ``trainer.loss_balance``; None = 1.0), applied with ``loss_weight``."""
    if outputs is None:
        outputs = model(x)
    per_target, per_stat = {}, {}
    for t, tcfg in targets_cfg.items():
        entry = TASK_REGISTRY[tcfg["task_type"]]
        w_t = weight_for_target(sample_weight, t, tcfg)
        if w_t is not None:
            if not entry["supports_sample_weight"]:
                raise ValueError(f"target {t!r} (task type {tcfg['task_type']!r}) does not support sample_weight")
            loss, stats = entry["fn"](y[tcfg["true_key"]], outputs, tcfg, scheduled_vars, sample_weight=w_t)
        else:
            loss, stats = entry["fn"](y[tcfg["true_key"]], outputs, tcfg, scheduled_vars)
        loss = loss.to(torch.float32)
        flood = tcfg["flood_level"]
        if training and flood is not None:
            loss = (loss - float(flood)).abs() + float(flood)
        per_target[t] = loss * (float(loss_weights[t]) * (1.0 if multipliers is None else multipliers[t]))
        for k, v in stats.items():
            per_stat[f"{t}_{k}"] = torch.as_tensor(v, dtype=torch.float32)

    raw = torch.stack(list(per_target.values())).sum()
    if uncertainty_layer is not None:
        total = uncertainty_layer({t: per_target[t] for t in kendall_task_names})
        if fixed_task_names:
            total = total + torch.stack([per_target[t] for t in fixed_task_names]).sum()
    else:
        total = raw
    reg = getattr(model, "regularization_loss", None)
    if reg is not None:
        extra = reg()
        total, raw = total + extra, raw + extra
    return total, raw, per_target, per_stat, {}


class Trainer:
    """GenericTrainer: scheduled_vars (at their schedule ``start`` until ``fit`` sets them per epoch), loss_weights,
    Kendall split, train/eval steps, and (T1) ``fit``: the epoch loop with schedules, early stopping, best weights,
    checkpoint + resume.

    ``tasks_cfg`` is the trainer config dict ({"targets", "loss", optional "constraints"}). ``optimizer`` may be
    None (losses only), a torch Optimizer (the Kendall log-variances are added to it as an extra param group if
    missing), or a callable ``params -> Optimizer``. ``selection_metric`` must be a SELECTION_METRICS name (unknown
    names raise UnknownNameError here, at construction)."""

    def __init__(self, model: torch.nn.Module, tasks_cfg: dict, optimizer=None, selection_metric: str = "val_raw_loss"):
        validate_trainer_config(tasks_cfg)
        check_selectable(selection_metric)
        self.selection_metric = selection_metric
        #: True = the last input channel is a group (timestamp) id: batches and evaluation chunks hold whole groups
        self.group_channel = False
        config = tasks_cfg  # kept as given: it is hashed into the RunIdentity (defaults are never written into it)
        self.model, self.config = model, config
        self.targets_cfg = resolve_targets(config)
        self.target_names = list(self.targets_cfg)
        loss_cfg = resolve_loss(config)
        self.schedules = {n: from_dict(ParamSchedule, c) for n, c in loss_cfg["schedules"].items()}  # parsed once
        self.scheduled_vars = {n: float(s.start) for n, s in self.schedules.items()}
        self.loss_weights = {t: float(c["loss_weight"]) for t, c in self.targets_cfg.items()}
        use_kendall = bool(loss_cfg["use_uncertainty_weighting"])
        excluded = set(loss_cfg["uncertainty_weighting_exclude_task_types"])
        self.kendall_task_names = [t for t, c in self.targets_cfg.items()
                                   if use_kendall and c["task_type"] not in excluded]
        self.fixed_task_names = [t for t in self.target_names if t not in self.kendall_task_names]
        self.uncertainty_layer = (UncertaintyWeightedLoss(self.kendall_task_names, loss_cfg["kendall_init_logvar"])
                                  if self.kendall_task_names else None)
        self.balancer, self._shared = make_balancer(loss_cfg["balance"], self.target_names,
                                                    self.kendall_task_names), None
        if self.balancer.needs_shared:   # FX2: h of the last forward (the model names its shared module)
            dict(model.named_modules())[model.balance_point].register_forward_hook(
                lambda _m, _i, out: setattr(self, "_shared", out))
        first = next(iter(model.parameters()), None)
        if self.uncertainty_layer is not None and first is not None:  # Kendall log-vars live on the model's device
            self.uncertainty_layer.to(first.device)
        if loss_cfg["all_constraints"]:
            raise NotImplementedError("output constraints are not ported yet")
        #: D-032: the cross-section term ("none" = no term) and its weight; needs ``group_channel``
        self.cross_section, self.cross_section_weight = loss_cfg["cross_section"], float(
            loss_cfg["cross_section_weight"])
        if self.cross_section not in CROSS_SECTION_LOSSES:
            raise ValueError(f"loss 'cross_section' must be one of {tuple(CROSS_SECTION_LOSSES)}, "
                             f"not {self.cross_section!r}")
        #: D-032: weight of the direction-balance term (0 = no term) and the two-sided targets it reads
        self.direction_balance = float(loss_cfg["direction_balance"])
        self.balance_targets = list(loss_cfg["direction_balance_targets"])
        unknown = sorted(set(self.balance_targets) - set(self.targets_cfg))
        if self.direction_balance < 0 or unknown or (self.direction_balance > 0 and not self.balance_targets):
            raise ValueError(f"loss 'direction_balance' must be >= 0 and name targets of the config "
                             f"('direction_balance_targets'; unknown: {unknown})")
        self.optimizer = self._attach_optimizer(optimizer)

    def _attach_optimizer(self, optimizer):
        if optimizer is None or isinstance(optimizer, torch.optim.Optimizer):
            if optimizer is not None and self.uncertainty_layer is not None:
                have = {id(p) for g in optimizer.param_groups for p in g["params"]}
                extra = [p for p in self.uncertainty_layer.parameters() if id(p) not in have]
                if extra:
                    optimizer.add_param_group({"params": extra})
            return optimizer
        return optimizer(list(self.parameters()))

    def parameters(self):
        yield from self.model.parameters()
        if self.uncertainty_layer is not None:
            yield from self.uncertainty_layer.parameters()

    def compute_losses(self, x, y, sample_weight=None, training: bool = False, outputs: dict | None = None,
                       groups: torch.Tensor | None = None):
        """``compute_losses`` with this trainer's config; the D-032 terms that are switched on (the cross-section
        term, the direction-balance term) are added to the total and the raw loss and logged as the stats
        ``cross_section_loss`` / ``direction_balance_loss``. ``groups``: the rows' timestamp ids when ``x`` is not
        given (``evaluate``); else they are read from the group channel of ``x``."""
        cross = CROSS_SECTION_LOSSES[self.cross_section] is not None
        if not cross and not self.direction_balance:
            return compute_losses(self.model, x, y, self.targets_cfg, self.scheduled_vars, self.loss_weights,
                                  self.uncertainty_layer, self.kendall_task_names, self.fixed_task_names,
                                  sample_weight=sample_weight, training=training, outputs=outputs,
                                  multipliers=self.balancer.multipliers())
        if cross and groups is None and not (self.group_channel and x is not None):
            raise ValueError(f"the cross-section loss {self.cross_section!r} needs whole-timestamp batches: run with "
                             f"SplitSettings.cross_section (the group channel)")
        outputs = self.model(x) if outputs is None else outputs
        total, raw, per_target, stats, cons = compute_losses(
            self.model, x, y, self.targets_cfg, self.scheduled_vars, self.loss_weights, self.uncertainty_layer,
            self.kendall_task_names, self.fixed_task_names, sample_weight=sample_weight, training=training,
            outputs=outputs, multipliers=self.balancer.multipliers())
        extra = []
        if cross:
            term = cross_section_loss(self.cross_section, outputs, y, self.targets_cfg,
                                      row_groups(x) if groups is None else groups)
            stats["cross_section_loss"] = term.detach().to(torch.float32)
            extra.append(self.cross_section_weight * term)
        if self.direction_balance:
            term = direction_balance_loss(outputs, self.targets_cfg, self.balance_targets)
            stats["direction_balance_loss"] = term.detach().to(torch.float32)
            extra.append(self.direction_balance * term)
        add = torch.stack(extra).sum()
        return total + add, raw + add, per_target, stats, cons

    @staticmethod
    def batch_indices(n: int, batch_size: int, shuffle: bool = True) -> list[torch.Tensor]:
        """Mini-batch index tensors over n rows; the shuffle draws from torch's global RNG, so it derives from the
        seed_everything(seed) call made before the Trainer is built."""
        order = torch.randperm(n) if shuffle else torch.arange(n)
        return list(order.split(batch_size))

    @staticmethod
    def _logs(total, raw, per_target, stats) -> dict[str, float]:
        logs = {"total_loss": float(total.detach()), "raw_loss": float(raw.detach())}
        logs.update({f"{t}_loss": float(v.detach()) for t, v in per_target.items()})
        logs.update({k: float(v.detach()) for k, v in stats.items()})
        return logs

    def train_step(self, x, y, sample_weight=None, grad_clip_norm: float | None = None,
                   as_tensors: bool = False) -> dict[str, float]:
        """One optimizer step on (x, y) with the model in train() mode; returns scalar logs. ``grad_clip_norm``
        clips the global norm of every trainer parameter's gradient before the step (None = no clipping).
        ``as_tensors``: the logs stay 0-d device tensors (no host sync; ``_train_epoch`` reads them per group)."""
        if self.optimizer is None:
            raise RuntimeError("Trainer has no optimizer")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, raw, per_target, stats, _ = self.compute_losses(x, y, sample_weight=sample_weight, training=True)
        self.balancer.update(per_target, self._shared)
        total.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.parameters()), grad_clip_norm)
        self.optimizer.step()
        if as_tensors:
            return tensor_logs(total, raw, per_target, stats)
        return self._logs(total, raw, per_target, stats)

    @torch.no_grad()
    def eval_step(self, x, y, sample_weight=None) -> dict[str, float]:
        """No-grad eval() pass; scalar logs + ``selection_metric`` (the value of ``self.selection_metric``)."""
        self.model.eval()
        outputs = self.model(x)
        total, raw, per_target, stats, _ = self.compute_losses(x, y, sample_weight=sample_weight, outputs=outputs)
        logs = self._logs(total, raw, per_target, stats)
        logs["selection_metric"] = compute_selection_metric(self.selection_metric, outputs, y, logs)
        return logs

    # ------------------------------------------------------------------------------------------------- fit (T1)
    @torch.no_grad()
    def evaluate(self, x, y, batch_size: int) -> dict[str, float]:
        """eval() pass over ALL rows in batches; losses and the selection metric are computed once on the
        concatenated outputs (an AUC must not be averaged over batches)."""
        self.model.eval()
        dev = self._device()
        gid = row_groups(x) if self.group_channel else None
        if self.group_channel:  # whole timestamps per chunk (rows are time-major: workflow cross_section)
            parts = [self.model(x[c].to(dev)) for c in group_chunks(gid, batch_size)]
        else:
            parts = [self.model(xb.to(dev)) for xb in x.split(batch_size)]
        outputs = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        y = {k: v.to(dev) for k, v in y.items()}
        total, raw, per_target, stats, _ = self.compute_losses(None, y, outputs=outputs,
                                                               groups=None if gid is None else gid.to(dev))
        logs = self._logs(total, raw, per_target, stats)
        labels = y if gid is None else {**y, GROUP_KEY: gid}  # a rank metric reads the timestamps
        logs["selection_metric"] = compute_selection_metric(self.selection_metric, outputs, labels, logs)
        return logs

    def _device(self) -> torch.device:
        """The model's device: data may stay on the CPU (a big dataset) and is moved batch by batch (B-019)."""
        first = next(iter(self.model.parameters()), None)
        return torch.device("cpu") if first is None else first.device

    def _train_epoch(self, x, y, settings: FitSettings, ema: EmaWeights | None = None, monitor=None,
                     epoch: int = 0) -> dict[str, float]:
        """One shuffled pass; the mean of every log, weighted by batch size; ``ema`` is updated after every step.
        The logs of ``settings.steps_per_execution`` steps are read back together (one host sync per group).
        ``monitor`` (a ``trainer.batch_influence.BatchInfluence``, log-only) is called around every step."""
        logs, dev = GroupedLogs(settings.steps_per_execution), self._device()
        batches = (group_batches(row_groups(x), settings.batch_size) if self.group_channel
                   else self.batch_indices(len(x), settings.batch_size))
        for ix in batches:
            if monitor is not None:
                monitor.before_step(self, epoch, ix)
            step = self.train_step(x[ix].to(dev), {k: v[ix].to(dev) for k, v in y.items()},
                                   grad_clip_norm=settings.grad_clip_norm, as_tensors=True)
            if monitor is not None:
                monitor.after_step(self, step)
            if ema is not None:
                ema.update(self.model)
            logs.add(len(ix), step)
        return logs.means(len(x))

    def _begin_epoch(self, epoch: int, settings: FitSettings, schedules: dict[str, ParamSchedule]) -> float:
        """Everything that must be a pure function of (settings, epoch) so that resume equals an uninterrupted
        run: lambda schedules, learning rate, and the RNG (dropout masks, shuffle order) reseeded per epoch."""
        apply_schedules(self.scheduled_vars, schedules, epoch)
        if settings.lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr_at(settings.lr, epoch)
        torch.manual_seed(settings.seed * 1_000_003 + epoch)
        return self.optimizer.param_groups[0]["lr"]

    def _state(self, epochs_done: int, tracker: BestTracker, history: list[dict], identity: RunIdentity,
               ema: EmaWeights | None = None) -> dict:
        state = {"epochs_done": epochs_done, "model": self.model.state_dict(), "tracker": tracker.get_state(),
                 "optimizer": self.optimizer.state_dict(), "history": history,
                 "kendall": None if self.uncertainty_layer is None else self.uncertainty_layer.state_dict(),
                 "identity": json.dumps(identity.to_json(), sort_keys=True)}
        if self.balancer.needs_shared:  # FX2 balancer state; a default checkpoint keeps its keys
            state["balance"] = self.balancer.state_dict()
        if ema is not None:  # only with the switch on: a default checkpoint keeps its keys
            state["ema"] = ema.state_dict()
        return state

    def _restore(self, ckpt: dict, tracker: BestTracker, identity: RunIdentity, ema: EmaWeights | None = None
                 ) -> list[dict]:
        check_resume(RunIdentity.from_json(json.loads(ckpt["identity"])), identity)
        self.model.load_state_dict(ckpt["model"])
        if ema is not None:
            ema.load_state_dict(ckpt["ema"])
        if self.uncertainty_layer is not None:
            self.uncertainty_layer.load_state_dict(ckpt["kendall"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.balancer.needs_shared:
            self.balancer.load_state_dict(ckpt["balance"])
        tracker.load_state(ckpt["tracker"])
        return list(ckpt["history"])

    def _trace(self, x, y, batch_size: int, batches) -> dict[str, float]:
        """The trace row of the epoch: on the first ``batch_size`` training rows, or the mean over ``batches``."""
        dev = self._device()
        if batches is None:
            return epoch_trace(self, x[:batch_size].to(dev), {k: v[:batch_size].to(dev) for k, v in y.items()})
        ix = [torch.as_tensor(b, dtype=torch.long) for b in batches]
        return epoch_trace_mean(self, [(x[i].to(dev), {k: v[i].to(dev) for k, v in y.items()}) for i in ix])

    def _evaluate_with(self, weights: dict | None, x, y, batch_size: int) -> dict[str, float]:
        """``evaluate`` with ``weights`` loaded for the pass (the live weights are put back after); None = live."""
        if weights is None:
            return self.evaluate(x, y, batch_size)
        live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(weights)
        try:
            return self.evaluate(x, y, batch_size)
        finally:
            self.model.load_state_dict(live)

    def fit(self, x_train, y_train, x_val, y_val, settings: FitSettings, *, run_dir=None,
            identity: RunIdentity | None = None, resume: bool = True,
            deadline: float | None = None, on_epoch=None, trace_batches=None, batch_monitor=None) -> FitResult:
        """Epoch loop with lambda / LR schedules, early stopping and best weights chosen by ``selection_metric``.

        With ``run_dir`` the state is saved after every epoch (checkpoint.py) and a run folder that already holds a
        checkpoint is RESUMED (``resume=False`` raises instead of overwriting); resume refuses a changed
        ``identity`` (ResumeMismatchError). An ``identity`` whose seed differs from ``settings.seed`` raises
        SeedMismatchError before anything runs (ROADMAP T2). Calling fit again with more ``epochs`` continues the
        run; the result equals an uninterrupted run (bitwise on CPU). At the end the model holds the best weights if
        ``early_stop.restore_best_weights``; the checkpoint keeps the LAST epoch's training state. ``deadline`` (a
        ``time.monotonic()`` value, default None = none) stops the loop before an epoch that would not finish by then
        (judged by the longest epoch of this call; at least one epoch always runs): the per-epoch checkpoint is the
        resume point of the next call. ``trace_batches`` (with ``settings.trace``): index arrays into the training
        rows (``trainer.batch_set.representative_batches``); the trace is then the mean over them instead of the first
        ``batch_size`` rows (log-only either way). D-021 switches: ``settings.init`` re-initialises the model at the
        start of a fresh fit (not on resume); ``grad_clip_norm`` clips every step; with ``ema_decay`` the EMA copy is
        what validation, selection, the best weights and the model at the end hold (the checkpoint keeps both).
        ``batch_monitor`` (default None = none): a ``trainer.batch_influence.BatchInfluence`` that measures every
        step's effect on its probe sets; log-only, the training is the same with and without it."""
        if self.optimizer is None:
            raise RuntimeError("Trainer has no optimizer")
        if run_dir is not None and identity is None:
            raise ValueError("run_dir needs an identity (RunIdentity) so that a resume can be checked")
        if identity is not None and identity.seed != settings.seed:
            raise SeedMismatchError(f"identity.seed ({identity.seed}) != settings.seed ({settings.seed}): the identity "
                                    f"would describe a different run than the one that is trained")
        seed_everything(settings.seed)
        ema = None
        higher = SELECTION_METRICS.entry(self.selection_metric).meta["higher_is_better"]
        tracker = BestTracker(settings.early_stop, higher)
        history: list[dict] = []
        best_weights = None
        ckpt = None if run_dir is None else checkpoint.load_checkpoint(run_dir)
        if ckpt is not None:
            if not resume:
                raise FileExistsError(f"{run_dir} already holds a checkpoint; resume=False refuses to overwrite it")
            ema = None if settings.ema_decay is None else EmaWeights(self.model, settings.ema_decay)
            history = self._restore(ckpt, tracker, identity, ema)
            best_weights = None if tracker.best is None else checkpoint.load_best(run_dir)
        else:
            apply_init(settings.init, self.model)       # "torch" (default) changes nothing
            ema = None if settings.ema_decay is None else EmaWeights(self.model, settings.ema_decay)
        longest = 0.0
        if batch_monitor is not None:
            batch_monitor.start(self)
        for epoch in range(len(history), settings.epochs):
            if tracker.should_stop():
                break
            if deadline is not None and longest and time.monotonic() + longest > deadline:
                break
            t0 = time.monotonic()
            lr = self._begin_epoch(epoch, settings, self.schedules)
            row = {"epoch": epoch + 1, "lr": lr}
            train_logs = self._train_epoch(x_train, y_train, settings, ema, batch_monitor, epoch)
            row.update({f"train_{k}": v for k, v in train_logs.items()})
            val = self._evaluate_with(None if ema is None else ema.shadow, x_val, y_val, settings.batch_size)
            row.update({f"val_{k}": v for k, v in val.items()})
            if settings.trace:
                row.update(self._trace(x_train, y_train, settings.batch_size, trace_batches))
            longest = max(longest, time.monotonic() - t0)
            if settings.trace:
                row["trace_epoch_seconds"] = time.monotonic() - t0
            history.append(row)
            if on_epoch is not None:  # progress display only (e.g. a print); it cannot change training
                on_epoch(dict(row))
            if tracker.update(epoch + 1, val["selection_metric"]):
                best_weights = ({k: v.detach().clone() for k, v in self.model.state_dict().items()} if ema is None
                                else ema.weights())
                if run_dir is not None:
                    checkpoint.save_best(run_dir, best_weights, {"best": tracker.best, "best_epoch": tracker.best_epoch,
                                                                 "metric": self.selection_metric})
            if run_dir is not None:
                checkpoint.save_checkpoint(run_dir, self._state(epoch + 1, tracker, history, identity, ema))
        restored = settings.early_stop.restore_best_weights and best_weights is not None
        if restored:
            self.model.load_state_dict(best_weights)
        elif ema is not None:
            self.model.load_state_dict(ema.weights())
        return FitResult(history, len(history), tracker.best_epoch, tracker.best, tracker.should_stop(), restored)
