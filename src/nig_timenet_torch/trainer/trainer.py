"""Trainer + compute_losses (source: crypto-signal-prediction trainer/trainer.py @ aeafc564, GenericTrainer
__init__ / _compute_losses / train_step / test_step; BUILD_PLAN.md P18, P22). ``Trainer.fit`` (ROADMAP T1) adds the
epoch loop; its pieces live in schedules.py, tracker.py and checkpoint.py.

L2 kernel regularisation: torch has no per-layer kernel_regularizer, so the model exposes
``regularization_loss()`` (``l2 * sum(w^2)``, no 0.5 factor, NOTES.md #6) and compute_losses adds it once to both
total and raw loss, as the TF trainer adds ``model.losses``.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.identity import RunIdentity, SeedMismatchError, check_resume
from nig_timenet_torch.trainer.schedules import LrSettings, ParamSchedule, apply_schedules, lr_at
from nig_timenet_torch.trainer.selection_metric import SELECTION_METRICS, compute_selection_metric
from nig_timenet_torch.trainer.task_weighting import UncertaintyWeightedLoss
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY
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

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")


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
                   training: bool = False, outputs: dict | None = None):
    """-> (total_loss, raw_loss, per_target_losses, per_stat_values, per_constraint_values).
    raw_loss = sum of loss * loss_weight (no Kendall terms) + regularisation; total applies Kendall to
    kendall_task_names and adds fixed_task_names unweighted. ``training`` only gates flooding (the model's own
    train()/eval() mode gates dropout). ``outputs``: precomputed model(x) (then x is not used)."""
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
        flood = tcfg.get("flood_level")
        if training and flood is not None:
            loss = (loss - float(flood)).abs() + float(flood)
        per_target[t] = loss * float(loss_weights[t])
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
        SELECTION_METRICS.entry(selection_metric)
        self.selection_metric = selection_metric
        config = tasks_cfg
        self.model, self.config = model, config
        self.target_names = list(config["targets"])
        loss_cfg = config["loss"]
        self.scheduled_vars = {n: float(c.get("start", 0.0)) for n, c in loss_cfg.get("schedules", {}).items()}
        self.loss_weights = {t: float(c.get("loss_weight", 1.0)) for t, c in config["targets"].items()}
        use_kendall = bool(loss_cfg.get("use_uncertainty_weighting", False))
        excluded = set(loss_cfg.get("uncertainty_weighting_exclude_task_types", ["evidential"]))
        self.kendall_task_names = [t for t, c in config["targets"].items()
                                   if use_kendall and c["task_type"] not in excluded]
        self.fixed_task_names = [t for t in self.target_names if t not in self.kendall_task_names]
        self.uncertainty_layer = UncertaintyWeightedLoss(self.kendall_task_names) if self.kendall_task_names else None
        first = next(iter(model.parameters()), None)
        if self.uncertainty_layer is not None and first is not None:  # Kendall log-vars live on the model's device
            self.uncertainty_layer.to(first.device)
        constraints = list(loss_cfg.get("constraints", [])) + list(config.get("constraints", []))
        if constraints:
            raise NotImplementedError("output constraints are not ported yet")
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

    def compute_losses(self, x, y, sample_weight=None, training: bool = False, outputs: dict | None = None):
        return compute_losses(self.model, x, y, self.config["targets"], self.scheduled_vars, self.loss_weights,
                              self.uncertainty_layer, self.kendall_task_names, self.fixed_task_names,
                              sample_weight=sample_weight, training=training, outputs=outputs)

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

    def train_step(self, x, y, sample_weight=None) -> dict[str, float]:
        """One optimizer step on (x, y) with the model in train() mode; returns scalar logs."""
        if self.optimizer is None:
            raise RuntimeError("Trainer has no optimizer")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, raw, per_target, stats, _ = self.compute_losses(x, y, sample_weight=sample_weight, training=True)
        total.backward()
        self.optimizer.step()
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
        parts = [self.model(xb) for xb in x.split(batch_size)]
        outputs = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        total, raw, per_target, stats, _ = self.compute_losses(None, y, outputs=outputs)
        logs = self._logs(total, raw, per_target, stats)
        logs["selection_metric"] = compute_selection_metric(self.selection_metric, outputs, y, logs)
        return logs

    def _train_epoch(self, x, y, batch_size: int) -> dict[str, float]:
        """One shuffled pass; the mean of every log, weighted by batch size."""
        sums: dict[str, float] = {}
        for ix in self.batch_indices(len(x), batch_size):
            logs = self.train_step(x[ix], {k: v[ix] for k, v in y.items()})
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v * len(ix)
        return {k: v / len(x) for k, v in sums.items()}

    def _begin_epoch(self, epoch: int, settings: FitSettings, schedules: dict[str, ParamSchedule]) -> float:
        """Everything that must be a pure function of (settings, epoch) so that resume equals an uninterrupted
        run: lambda schedules, learning rate, and the RNG (dropout masks, shuffle order) reseeded per epoch."""
        apply_schedules(self.scheduled_vars, schedules, epoch)
        if settings.lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr_at(settings.lr, epoch)
        torch.manual_seed(settings.seed * 1_000_003 + epoch)
        return self.optimizer.param_groups[0]["lr"]

    def _state(self, epochs_done: int, tracker: BestTracker, history: list[dict], identity: RunIdentity) -> dict:
        return {"epochs_done": epochs_done, "model": self.model.state_dict(), "tracker": tracker.get_state(),
                "optimizer": self.optimizer.state_dict(), "history": history,
                "kendall": None if self.uncertainty_layer is None else self.uncertainty_layer.state_dict(),
                "identity": json.dumps(identity.to_json(), sort_keys=True)}

    def _restore(self, ckpt: dict, tracker: BestTracker, identity: RunIdentity) -> list[dict]:
        check_resume(RunIdentity.from_json(json.loads(ckpt["identity"])), identity)
        self.model.load_state_dict(ckpt["model"])
        if self.uncertainty_layer is not None:
            self.uncertainty_layer.load_state_dict(ckpt["kendall"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        tracker.load_state(ckpt["tracker"])
        return list(ckpt["history"])

    def fit(self, x_train, y_train, x_val, y_val, settings: FitSettings, *, run_dir=None,
            identity: RunIdentity | None = None, resume: bool = True) -> FitResult:
        """Epoch loop with lambda / LR schedules, early stopping and best weights chosen by ``selection_metric``.

        With ``run_dir`` the state is saved after every epoch (checkpoint.py) and a run folder that already holds a
        checkpoint is RESUMED (``resume=False`` raises instead of overwriting); resume refuses a changed
        ``identity`` (ResumeMismatchError). An ``identity`` whose seed differs from ``settings.seed`` raises
        SeedMismatchError before anything runs (ROADMAP T2). Calling fit again with more ``epochs`` continues the
        run; the result equals an uninterrupted run (bitwise on CPU). At the end the model holds the best weights if
        ``early_stop.restore_best_weights``; the checkpoint keeps the LAST epoch's training state."""
        if self.optimizer is None:
            raise RuntimeError("Trainer has no optimizer")
        if run_dir is not None and identity is None:
            raise ValueError("run_dir needs an identity (RunIdentity) so that a resume can be checked")
        if identity is not None and identity.seed != settings.seed:
            raise SeedMismatchError(f"identity.seed ({identity.seed}) != settings.seed ({settings.seed}): the identity "
                                    f"would describe a different run than the one that is trained")
        seed_everything(settings.seed)
        schedules = {n: from_dict(ParamSchedule, c) for n, c in self.config["loss"].get("schedules", {}).items()}
        higher = SELECTION_METRICS.entry(self.selection_metric).meta["higher_is_better"]
        tracker = BestTracker(settings.early_stop, higher)
        history: list[dict] = []
        best_weights = None
        ckpt = None if run_dir is None else checkpoint.load_checkpoint(run_dir)
        if ckpt is not None:
            if not resume:
                raise FileExistsError(f"{run_dir} already holds a checkpoint; resume=False refuses to overwrite it")
            history = self._restore(ckpt, tracker, identity)
            best_weights = None if tracker.best is None else checkpoint.load_best(run_dir)
        for epoch in range(len(history), settings.epochs):
            if tracker.should_stop():
                break
            lr = self._begin_epoch(epoch, settings, schedules)
            row = {"epoch": epoch + 1, "lr": lr}
            row.update({f"train_{k}": v for k, v in self._train_epoch(x_train, y_train, settings.batch_size).items()})
            val = self.evaluate(x_val, y_val, settings.batch_size)
            row.update({f"val_{k}": v for k, v in val.items()})
            history.append(row)
            if tracker.update(epoch + 1, val["selection_metric"]):
                best_weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if run_dir is not None:
                    checkpoint.save_best(run_dir, best_weights, {"best": tracker.best, "best_epoch": tracker.best_epoch,
                                                                 "metric": self.selection_metric})
            if run_dir is not None:
                checkpoint.save_checkpoint(run_dir, self._state(epoch + 1, tracker, history, identity))
        restored = settings.early_stop.restore_best_weights and best_weights is not None
        if restored:
            self.model.load_state_dict(best_weights)
        return FitResult(history, len(history), tracker.best_epoch, tracker.best, tracker.should_stop(), restored)
