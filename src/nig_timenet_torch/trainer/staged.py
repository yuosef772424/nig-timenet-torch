"""PURPOSE: Staged training (ROADMAP ST1, card F): train on part of the rows until the train probe and the fixed
    validation show memorisation, restore the best weights, add rows, lower the lr; plus B1 / B2 / B3 plans. A
    step loop of the ONE Trainer.
TAGS: staged training, memorisation sign, growth arms, train probe, stage table, lr decay, equal budget baselines, st1
PITFALLS: not a second trainer: every update goes through ``Trainer.train_step`` (G5). The scalers are fitted ONCE on
    the whole training split before this runs (``run.build``) and never again. The sign needs val_raw_loss flat AND the
    eval-mode train probe falling; a flat probe means "finished learning" and never ends a stage. Selection and the
    stage's best weights use the RAW selection value of the FIXED validation rows (N-13), never a smoothed one. The
    lambda schedules advance with gstep // (steps of one pass over the FULL split), so one stage over all rows follows
    the plain epoch schedule exactly. A cross-section trainer, ``ema_decay`` and ``FitSettings.lr`` are refused.

Card: docs/audit/2026-11-14-model-first/cards/F_iterative_training.md (definitions section 2, loop section 3, baselines
section 5). With one stage over all rows, ``gamma`` 1 and no sign (``StageSettings(start_share=1.0, gamma=1.0)`` with
``eval_every`` = steps of one epoch and ``step_cap_epochs`` = the epochs) the updates are bit-for-bit those of
``Trainer.fit`` (same reseeding per pass, same permutation, same lambda schedule, same selection).
"""

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from nig_timenet_torch.core.config import to_dict
from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.core.seeds import bn_refresh_seed, epoch_seed, seed_everything, stage_probe_seed
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer import checkpoint, fit_state
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.log_groups import GroupedLogs
from nig_timenet_torch.trainer.schedules import apply_schedules
from nig_timenet_torch.trainer.selection_metric import SELECTION_METRICS
from nig_timenet_torch.trainer.switches import apply_init
from nig_timenet_torch.trainer.tracker import BestTracker, EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer

__all__ = ["GROWTH", "GrowthContext", "SignDetector", "StageSettings", "StageSpec", "StagedResult", "plan_stages",
           "baseline_stages", "staged_fit", "default_eval_every"]

_NEVER = 10 ** 9          # patience that never stops: the staged loop ends stages itself
GROWTH = Registry("growth", Contract("growth", required_meta=("doc",)))


@dataclass(frozen=True)
class GrowthContext:
    """What a growth arm may look at, per TRAIN row (positions 0..n-1 of the training tensors)."""
    coin: np.ndarray        # (n,) integer coin id of every row
    time: np.ndarray        # (n,) int64 time of every row
    seed: int = 0

    def __post_init__(self):
        if len(self.coin) != len(self.time):
            raise ValueError("GrowthContext: coin and time must have one entry per training row")


def _stride_rows(ctx: GrowthContext, share: float) -> np.ndarray:
    """Every k-th row, k = 1 / share (a power of two, so that the stages are nested): all coins, all dates."""
    k = 1.0 / share
    if abs(k - round(k)) > 1e-9 or int(round(k)) < 1 or int(round(k)) & (int(round(k)) - 1):
        raise ValueError(f"growth 'stride' needs shares 1 / 2^j, got share {share!r} (stride {k!r})")
    return np.arange(0, len(ctx.coin), int(round(k)), dtype=np.int64)


def _coin_rows(ctx: GrowthContext, share: float) -> np.ndarray:
    """All rows of the first ceil(share * coins) coins of a fixed random coin order (nested: a prefix)."""
    coins = np.unique(ctx.coin)
    order = coins[np.random.default_rng(ctx.seed).permutation(len(coins))]
    keep = order[:max(1, math.ceil(share * len(coins) - 1e-9))]
    return np.flatnonzero(np.isin(ctx.coin, keep)).astype(np.int64)


def _time_rows(ctx: GrowthContext, share: float) -> np.ndarray:
    """The OLDEST ceil(share * rows) rows by time (nested: a prefix of the time order)."""
    n = len(ctx.time)
    first = np.argsort(ctx.time, kind="stable")[:max(1, math.ceil(share * n - 1e-9))]
    return np.sort(first).astype(np.int64)


GROWTH.register("stride", _stride_rows, version=1, doc="every 1/share-th row, all coins and dates (card arm G1)")
GROWTH.register("coins", _coin_rows, version=1, doc="the rows of a growing share of the coins (card arm G2)")
GROWTH.register("time", _time_rows, version=1, doc="the oldest share of the rows, extended forward (card arm G3)")


@dataclass(frozen=True)
class StageSettings:
    """Staged training (card F section 2-4). Every field is read by ``plan_stages`` / ``staged_fit``."""
    growth: str = "stride"             # GROWTH: stride | coins | time
    start_share: float = 1 / 32        # share of the training rows of stage 0
    grow: float = 2.0                  # the share is multiplied by this every stage (capped at 1 = the whole split)
    gamma: float = 0.5                 # lr of stage s = max(lr_{s-1} * gamma, lr_floor)
    lr_floor: float = 1e-6
    eval_every: int | None = None      # optimizer steps between evaluations; None = max(100, 1/4 of the stage's pass)
    k: int = 3                         # evaluations without improvement before the sign
    min_delta: float = 1e-3            # improvement threshold of the sign (val plateau AND train probe fall)
    smooth_points: int = 3             # the sign reads the mean of this many evaluations of val_raw_loss
    step_cap_epochs: float = 20.0      # a stage ends at this many passes over its rows even without the sign
    warmup_steps: int = 100            # lr warm-up after a stage boundary (stage 0 has none)
    new_weight: int = 1                # the rows added at a stage are drawn this many times per pass (union kept)
    probe_rows: int = 2000             # eval-mode train probe, drawn once per stage from the stage's rows
    val_max_rows: int = 60_000         # the fixed validation rows: evenly strided to at most this many
    bn_refresh_steps: int = 50         # forward-only train-mode steps after rows are added (BatchNorm statistics)
    restore_stage_best: bool = True    # improvement 1: restore the stage's best weights before adding data
    reset_optimizer: bool = True       # improvement 3: reset Adam's moments at a stage boundary (with the warm-up)

    def __post_init__(self):
        GROWTH.entry(self.growth)  # unknown arm names raise
        ints = {"k": self.k, "smooth_points": self.smooth_points, "new_weight": self.new_weight,
                "probe_rows": self.probe_rows, "val_max_rows": self.val_max_rows}
        for name, v in ints.items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"StageSettings.{name} must be an integer >= 1, got {v!r}")
        for name in ("warmup_steps", "bn_refresh_steps"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"StageSettings.{name} must be an integer >= 0, got {v!r}")
        if self.eval_every is not None and (isinstance(self.eval_every, bool) or self.eval_every < 1):
            raise ValueError(f"StageSettings.eval_every must be None or an integer >= 1, got {self.eval_every!r}")
        if not (0.0 < self.start_share <= 1.0 and self.grow > 1.0 and 0.0 < self.gamma <= 1.0):
            raise ValueError("need 0 < start_share <= 1, grow > 1 and 0 < gamma <= 1")
        if self.lr_floor < 0 or self.min_delta < 0 or self.step_cap_epochs <= 0:
            raise ValueError("need lr_floor >= 0, min_delta >= 0 and step_cap_epochs > 0")


@dataclass(frozen=True)
class StageSpec:
    """One stage of a plan: the rows trained on (the union so far), the rows new at this stage, lr, caps."""
    rows: np.ndarray                   # sorted positions into the training tensors
    new_rows: np.ndarray               # the part of ``rows`` that is new at this stage
    lr: float
    step_cap: int
    eval_every: int
    stop_at_sign: bool = True
    restart: bool = True               # optimizer reset + warm-up + BN refresh at the start (never stage 0)
    lr_at: tuple = ()                  # ((step_in_stage, lr), ...): later entries replace ``lr`` (baseline B2)


class SignDetector:
    """The sign of memorisation (card F section 2): the ``smooth``-point mean of the validation loss has not improved
    by ``min_delta`` for ``k`` evaluations while the eval-mode train probe loss fell by more than ``min_delta`` over
    the same ``k`` evaluations. Pure numbers; ``update`` returns True at the first evaluation that shows it."""

    def __init__(self, k: int, smooth: int, min_delta: float):
        self.k, self.smooth, self.min_delta = k, smooth, min_delta
        self.val: list[float] = []
        self.probe: list[float] = []

    def update(self, val_loss: float, probe_loss: float) -> bool:
        self.val.append(float(val_loss))
        self.probe.append(float(probe_loss))
        m = [float(np.mean(self.val[i - self.smooth + 1:i + 1])) for i in range(self.smooth - 1, len(self.val))]
        if len(m) < self.k + 1:
            return False
        stalled = min(m[-self.k:]) > min(m[:-self.k]) - self.min_delta
        fell = self.probe[-self.k - 1] - self.probe[-1] > self.min_delta
        return bool(stalled and fell)

    def get_state(self) -> dict:
        return {"val": list(self.val), "probe": list(self.probe)}

    def load_state(self, state: dict) -> None:
        self.val, self.probe = list(state["val"]), list(state["probe"])


def default_eval_every(n_rows: int, batch_size: int) -> int:
    """Card F section 2: 1/4 of the stage's pass, at least 100 optimizer steps."""
    return max(100, math.ceil(math.ceil(n_rows / batch_size) / 4))


def _cap(settings: StageSettings, pool_rows: int, batch_size: int) -> int:
    return max(1, math.ceil(settings.step_cap_epochs * math.ceil(pool_rows / batch_size)))


def _pool_rows(rows: np.ndarray, new_rows: np.ndarray, weight: int) -> int:
    return len(rows) + (weight - 1) * len(new_rows)


def plan_stages(settings: StageSettings, ctx: GrowthContext, batch_size: int, base_lr: float) -> list[StageSpec]:
    """The stages of the staged arm: nested row sets (shares start_share, x grow, ... , 1), lr x gamma per stage
    (never below ``lr_floor``, never above the previous lr), the cap and the evaluation cadence of each."""
    grow_rows = GROWTH[settings.growth]
    shares, share = [], settings.start_share
    while True:
        shares.append(1.0 if share >= 1.0 - 1e-12 else share)
        if shares[-1] >= 1.0:
            break
        share *= settings.grow
    n = len(ctx.coin)
    stages: list[StageSpec] = []
    prev, lr = np.array([], dtype=np.int64), base_lr
    for sh in shares:
        rows = np.arange(n, dtype=np.int64) if sh >= 1.0 else np.sort(grow_rows(ctx, sh)).astype(np.int64)
        s = len(stages)
        if s and len(rows) == len(prev):     # a share too small to add a coin / row: no new stage
            continue
        if s and not np.isin(prev, rows).all():
            raise ValueError(f"growth {settings.growth!r}: the rows of stage {s} do not contain stage {s - 1}")
        new = np.setdiff1d(rows, prev)
        lr = lr if s == 0 else min(lr, max(lr * settings.gamma, settings.lr_floor))
        pool = _pool_rows(rows, new if s else rows[:0], settings.new_weight)
        every = settings.eval_every or default_eval_every(pool, batch_size)
        cap = _cap(settings, pool, batch_size)
        stages.append(StageSpec(rows, new if s else rows[:0], lr, cap, min(every, cap), True, s > 0))
        prev = rows
    return stages


def baseline_stages(kind: str, staged: list[StageSpec], result: "StagedResult", settings: StageSettings,
                    batch_size: int) -> list[StageSpec]:
    """The equal-budget baselines of card F section 5, one stage each, run for ``result.total_steps`` steps:
    ``b1`` all rows, constant lr_0; ``b2`` all rows, the staged run's lr at the staged run's stage boundary steps;
    ``b3`` the rows of stage 0 only, constant lr_0. None stops at a sign and none restarts."""
    if kind not in ("b1", "b2", "b3"):
        raise ValueError(f"baseline kind must be b1, b2 or b3, got {kind!r}")
    total = result.total_steps
    rows = staged[0].rows if kind == "b3" else staged[-1].rows
    lr_at: tuple = ()
    if kind == "b2":   # the staged run's lr from the step each of its stages began at
        begins = np.concatenate([[0], np.cumsum([r["steps"] for r in result.stages])[:-1]]).astype(int)
        lr_at = tuple((int(a), staged[i].lr) for i, a in enumerate(begins) if i)
    every = min(settings.eval_every or default_eval_every(len(rows), batch_size), total)
    return [StageSpec(rows, rows[:0], staged[0].lr, total, max(1, every), False, False, lr_at)]


@dataclass
class StagedResult:
    history: list[dict]            # one row per evaluation point
    stages: list[dict]             # the per-stage table (card F improvement 7)
    total_steps: int
    best_value: float | None       # best raw selection value over ALL stages
    best_eval: int | None          # 1-based evaluation index of that value
    best_step: int | None
    restored_best: bool            # the model now holds those weights
    finished: bool                 # every stage ended (False: stopped by the deadline, resume continues)
    val_rows: int = 0
    extra: dict = field(default_factory=dict)


def _strided(n: int, max_rows: int) -> np.ndarray | None:
    if n <= max_rows:
        return None
    return np.unique(np.linspace(0, n - 1, max_rows).astype(np.int64))


class _Loop:
    """The step-level loop of one ``staged_fit`` call; the pieces are small methods so that each fits a screen."""

    def __init__(self, trainer: Trainer, data: tuple, stages: list[StageSpec], st: StageSettings, fit: FitSettings,
                 run_dir, identity, resume: bool, deadline, on_eval):
        fit_state.require_optimizer(trainer)
        if trainer.group_channel:
            raise ValueError("staged training does not support a cross-section trainer (stages cut whole rows)")
        if fit.ema_decay is not None or fit.lr is not None:
            raise ValueError("staged training sets the lr itself and keeps no EMA: FitSettings.lr and ema_decay "
                             "must be None")
        fit_state.check_run_args(run_dir, identity, fit.seed)
        self.t, self.stages, self.st, self.fit = trainer, stages, st, fit
        self.run_dir, self.identity, self.resume = run_dir, identity, resume
        self.deadline, self.on_eval = deadline, on_eval
        self.x, self.y, xv, yv = data[0], data[1], data[2], data[3]
        pick = _strided(len(xv), st.val_max_rows)       # the SAME validation rows in every stage and baseline
        self.xv, self.yv = (xv, yv) if pick is None else (xv[torch.as_tensor(pick)],
                                                           {k: v[torch.as_tensor(pick)] for k, v in yv.items()})
        self.dev, self.bs = trainer.device, fit.batch_size
        self.full_spe = math.ceil(len(self.x) / self.bs)
        es = fit.early_stop
        self.tracker = BestTracker(EarlyStopSettings(patience=_NEVER, min_delta=es.min_delta, smoothing="none",
                                                     restore_best_weights=es.restore_best_weights),
                                   SELECTION_METRICS.entry(trainer.selection_metric).meta["higher_is_better"])
        self.s, self.started, self.sis, self.gstep, self.pass_id, self.b = 0, False, 0, 0, 0, 0
        self.perm, self.history, self.records = None, [], []
        self.best_w, self.stage_best_w, self.stage_tracker, self.detector = None, None, None, None
        self.logs, self.interval_rows, self.lr_now = GroupedLogs(fit.steps_per_execution), 0, 0.0

    # ---------------------------------------------------------------------------------------------- one stage
    def _setup_stage_data(self) -> None:
        """Everything derived from (settings, stage index): pool, probe rows, fresh sign detector and stage tracker."""
        spec, st = self.stages[self.s], self.st
        pool = np.concatenate([spec.rows] + [spec.new_rows] * (st.new_weight - 1))
        self.pool_t, self.n_pool = torch.as_tensor(pool), len(pool)
        # own generator: the global RNG is untouched
        g = torch.Generator().manual_seed(stage_probe_seed(self.fit.seed, self.s))
        pick = torch.randperm(len(spec.rows), generator=g)[:st.probe_rows]
        probe = torch.as_tensor(np.sort(spec.rows[pick.numpy()]))
        self.xp, self.yp = self.x[probe], {k: v[probe] for k, v in self.y.items()}
        self.stage_tracker = BestTracker(EarlyStopSettings(patience=_NEVER, min_delta=self.fit.early_stop.min_delta,
                                                           smoothing="none"), self.tracker.higher_is_better)
        self.detector = SignDetector(st.k, st.smooth_points, st.min_delta)
        self.stage_best_w = None

    def _enter_stage(self) -> None:
        spec, st = self.stages[self.s], self.st
        self._setup_stage_data()
        if self.s and self.b > 0:
            self.pass_id += 1
        self.b, self.perm, self.sis = 0, None, 0
        if self.s and spec.restart:
            if st.reset_optimizer:
                self.t.optimizer.state.clear()
            self._refresh_bn(spec)
        self.started = True

    @torch.no_grad()
    def _refresh_bn(self, spec: StageSpec) -> None:
        """Forward-only train-mode steps on the stage's rows so that BatchNorm moving statistics see the new rows."""
        rows = spec.new_rows if len(spec.new_rows) else spec.rows
        g = torch.Generator().manual_seed(bn_refresh_seed(self.fit.seed, self.s))
        self.t.model.train()
        for _ in range(self.st.bn_refresh_steps):
            ix = torch.as_tensor(rows)[torch.randint(len(rows), (min(self.bs, len(rows)),), generator=g)]
            self.t.model(self.x[ix].to(self.dev))

    def _lr(self, spec: StageSpec) -> float:
        lr = spec.lr
        for frm, v in spec.lr_at:
            if self.sis >= frm:
                lr = v
        if spec.restart and self.s and self.sis < self.st.warmup_steps:
            lr *= (self.sis + 1) / self.st.warmup_steps
        return lr

    def _one_step(self, spec: StageSpec) -> None:
        if self.perm is None:
            torch.manual_seed(epoch_seed(self.fit.seed, self.pass_id))   # as Trainer._begin_epoch
            self.perm = torch.randperm(self.n_pool)
        ix = self.pool_t[self.perm[self.b * self.bs:(self.b + 1) * self.bs]]
        self.lr_now = self._lr(spec)
        for group in self.t.optimizer.param_groups:
            group["lr"] = self.lr_now
        apply_schedules(self.t.scheduled_vars, self.t.schedules, self.gstep // self.full_spe)
        logs = self.t.train_step(self.x[ix].to(self.dev), {k: v[ix].to(self.dev) for k, v in self.y.items()},
                                 grad_clip_norm=self.fit.grad_clip_norm, as_tensors=True)
        self.logs.add(len(ix), logs)
        self.interval_rows += len(ix)
        self.sis, self.gstep, self.b = self.sis + 1, self.gstep + 1, self.b + 1
        if self.b * self.bs >= self.n_pool:
            self.pass_id, self.b, self.perm = self.pass_id + 1, 0, None

    def _evaluate(self, spec: StageSpec) -> dict:
        val = self.t.evaluate(self.xv, self.yv, self.bs)
        probe = self.t.evaluate(self.xp, self.yp, self.bs)
        row = {"eval": len(self.history) + 1, "step": self.gstep, "stage": self.s, "stage_step": self.sis,
               "rows": int(len(spec.rows)), "lr": self.lr_now}
        row.update({f"train_{k}": v for k, v in self.logs.means(self.interval_rows).items()})
        row.update({f"val_{k}": v for k, v in val.items()})
        row.update(probe_raw_loss=probe["raw_loss"], gap=val["raw_loss"] - probe["raw_loss"])
        row["sign"] = self.detector.update(val["raw_loss"], probe["raw_loss"])
        self.logs, self.interval_rows = GroupedLogs(self.fit.steps_per_execution), 0
        if self.stage_tracker.update(row["eval"], val["selection_metric"]):
            self.stage_best_w = fit_state.clone_state(self.t.model)
        if self.tracker.update(row["eval"], val["selection_metric"]):
            self.best_w = fit_state.clone_state(self.t.model)
            fit_state.save_best_weights(self.run_dir, self.best_w, self.tracker, self.t.selection_metric)
        self.history.append(row)
        return row

    @torch.no_grad()
    def _rank_ics(self) -> dict:
        """Rank IC of the NIG mean of every price head on the fixed validation rows (the model's current weights)."""
        names = {t: f"val_rank_ic_{t}_reg" for t in TARGET_COLUMNS
                 if head_spec("nig_regression", t).label_key in self.yv}
        live = self.t.selection_metric
        try:
            out = {}
            for t, name in names.items():
                self.t.selection_metric = name
                out[t] = self.t.evaluate(self.xv, self.yv, self.bs)["selection_metric"]
            return out
        finally:
            self.t.selection_metric = live

    def _end_stage(self, spec: StageSpec, reason: str, row: dict) -> None:
        if self.st.restore_stage_best and self.stage_best_w is not None:
            self.t.model.load_state_dict(self.stage_best_w)
        st = self.stage_tracker
        self.records.append({
            "stage": self.s, "rows": int(len(spec.rows)), "new_rows": int(len(spec.new_rows)), "lr": spec.lr,
            "steps": self.sis, "steps_to_sign": self.sis if reason == "sign" else None, "ended_by": reason,
            "best_value": st.best,
            "best_step": None if st.best_epoch is None else self.history[st.best_epoch - 1]["step"],
            "val_raw_at_end": row["val_raw_loss"], "probe_raw_at_end": row["probe_raw_loss"], "gap_at_end": row["gap"],
            "rank_ic": self._rank_ics()})
        self.s, self.started = self.s + 1, False

    # ------------------------------------------------------------------------------------------ checkpoint / resume
    def _signature(self) -> str:
        plan = [[len(s.rows), len(s.new_rows), s.lr, s.step_cap, s.eval_every, s.stop_at_sign, s.restart,
                 [list(x) for x in s.lr_at]] for s in self.stages]
        return json.dumps({"settings": to_dict(self.st), "plan": plan, "batch_size": self.bs,
                           "val_rows": len(self.xv)}, sort_keys=True)

    def _save(self) -> None:
        if self.run_dir is None:
            return
        staged = {"s": self.s, "started": self.started, "sis": self.sis, "gstep": self.gstep,
                  "pass_id": self.pass_id, "b": self.b, "rng": torch.get_rng_state(),
                  "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                  "records": self.records, "signature": self._signature(), "lr_now": self.lr_now,
                  "stage_tracker": None if self.stage_tracker is None else self.stage_tracker.get_state(),
                  "detector": None if self.detector is None else self.detector.get_state(),
                  "stage_best": self.stage_best_w}
        checkpoint.save_checkpoint(self.run_dir, fit_state.base_state(
            self.t, len(self.history), self.tracker, self.history, self.identity, staged=staged))

    def _load(self, ckpt: dict) -> None:
        # the identity first: a flat run's checkpoint has no "staged" part (D-042: staged settings are in it)
        fit_state.check_checkpoint_identity(ckpt, self.identity)
        sg = ckpt["staged"]
        if sg["signature"] != self._signature():
            raise ValueError(f"{self.run_dir}: the staged plan or settings changed since the checkpoint was written")
        self.history = fit_state.restore_base(self.t, ckpt, self.tracker)
        self.records = list(sg["records"])
        self.s, self.started, self.sis, self.gstep = sg["s"], sg["started"], sg["sis"], sg["gstep"]
        self.pass_id, self.b, self.lr_now = sg["pass_id"], sg["b"], sg["lr_now"]
        self.best_w = None if self.tracker.best is None else checkpoint.load_best(self.run_dir)
        if self.started:
            self._setup_stage_data()
            self.stage_tracker.load_state(sg["stage_tracker"])
            self.detector.load_state(sg["detector"])
            self.stage_best_w = sg["stage_best"]
            if self.b > 0:     # mid-pass: the same permutation, then the dropout stream where it stopped
                torch.manual_seed(epoch_seed(self.fit.seed, self.pass_id))
                self.perm = torch.randperm(self.n_pool)
                torch.set_rng_state(sg["rng"])
                if sg["cuda_rng"]:
                    torch.cuda.set_rng_state_all(sg["cuda_rng"])

    # ------------------------------------------------------------------------------------------------------- run
    def run(self) -> StagedResult:
        seed_everything(self.fit.seed)
        ckpt = fit_state.load_resume_checkpoint(self.run_dir, self.resume)
        if ckpt is not None:
            self._load(ckpt)
        else:
            apply_init(self.fit.init, self.t.model)
        longest, t0 = 0.0, time.monotonic()
        while self.s < len(self.stages):
            if not self.started:
                self._enter_stage()
            spec = self.stages[self.s]
            target = min(spec.step_cap, (self.sis // spec.eval_every + 1) * spec.eval_every)
            while self.sis < target:
                self._one_step(spec)
            row = self._evaluate(spec)
            reason = "sign" if (row["sign"] and spec.stop_at_sign) else ("cap" if self.sis >= spec.step_cap else None)
            if reason:
                self._end_stage(spec, reason, row)
            self._save()
            if self.on_eval is not None:     # progress display only; it cannot change training
                self.on_eval(dict(row))
            longest = max(longest, time.monotonic() - t0)
            t0 = time.monotonic()
            if self.deadline is not None and time.monotonic() + longest > self.deadline and self.s < len(self.stages):
                break
        return self._finish()

    def _finish(self) -> StagedResult:
        restored = fit_state.finish_weights(self.t.model, self.best_w, self.fit.early_stop.restore_best_weights)
        tr = self.tracker
        return StagedResult(self.history, self.records, self.gstep, tr.best, tr.best_epoch,
                            None if tr.best_epoch is None else self.history[tr.best_epoch - 1]["step"], restored,
                            self.s >= len(self.stages), len(self.xv))


def staged_fit(trainer: Trainer, x_train, y_train, x_val, y_val, stages: list[StageSpec], settings: StageSettings,
               fit: FitSettings, *, run_dir=None, identity: RunIdentity | None = None, resume: bool = True,
               deadline: float | None = None, on_eval=None) -> StagedResult:
    """Run ``stages`` (``plan_stages`` or ``baseline_stages``) on the ONE ``trainer``: evaluations every
    ``eval_every`` steps on the fixed validation rows and a train probe, the sign of memorisation, the stage's best
    weights restored at the end of each stage, the best weights over all stages kept at the end (when
    ``fit.early_stop.restore_best_weights``). With ``run_dir`` the state is saved at every evaluation and a folder
    that holds a checkpoint is RESUMED (equal to an uninterrupted run, bitwise on CPU); the layout is that of
    ``Trainer.fit`` (``checkpoint.pt``, ``best.pt``) so ``workflow.run.restore`` reads it. ``deadline``: a
    ``time.monotonic()`` value; the loop stops after an evaluation that leaves no time for the next interval."""
    if not stages:
        raise ValueError("staged_fit needs at least one stage")
    return _Loop(trainer, (x_train, y_train, x_val, y_val), stages, settings, fit, run_dir, identity, resume,
                 deadline, on_eval).run()
