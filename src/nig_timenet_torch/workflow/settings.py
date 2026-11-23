"""PURPOSE: Settings of the workflow layer: the strict RunSettings tree of one flat single-split run, and the rolling
    walk-forward settings (RollingSettings) that workflow/rolling.py (folds) and rolling_run.py (runs) read.
TAGS: run settings, strict settings tree, split settings, train settings, evaluation settings, rolling settings
PITFALLS: RunSettings has no rolling field: a rolling study is run_rolling(RunSettings, RollingSettings), one Run per
    fold; every field is read by workflow code; RollingSettings has NO defaults for the spans (the experiment card
    decides them). A new field's default must reproduce the baseline job (tests/jobs/test_baseline_pin.py).

``RunSettings`` is everything ``workflow.run`` needs for ONE run on ONE chronological split: the data, the split, the
model, the training and the evaluation (health report, test report, battery). The jobs (``jobs/baseline.py``) and the
notebooks build it (``from_dict`` / ``updated``) and call the steps; nothing else configures a run. ``RollingSettings``
is the walk-forward setting (BUILD_PLAN.md P37, ROADMAP F2) used by ``workflow/rolling.py`` and ``rolling_run.py``.
"""

from dataclasses import dataclass, field

import pandas as pd

from nig_timenet_torch.core.costs import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.eval.battery_scorecard import TRIM_KEYS
from nig_timenet_torch.eval.verification_trading import STOP_DISTANCE
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings
from nig_timenet_torch.trainer.cross_section_loss import CROSS_SECTION_LOSSES
from nig_timenet_torch.trainer.day_weights import DayWeightSettings
from nig_timenet_torch.trainer.groups import BATCH_ORDERS
from nig_timenet_torch.trainer.schedules import LR_KINDS
from nig_timenet_torch.trainer.staged import StageSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import BATCHINGS

__all__ = ["RollingSettings", "RunSettings", "ROLLING_MODES", "SplitSettings", "TrainSettings", "EvalSettings",
           "BatterySettings", "VerificationSettings", "default_model_settings", "DEFAULT_MODEL_OVERRIDES", "model_for"]

ROLLING_MODES = ("expanding", "sliding")


@dataclass(frozen=True, kw_only=True)
class RollingSettings:
    mode: str = "expanding"            # "expanding" | "sliding"
    train_span: pd.Timedelta           # expanding: span of the FIRST fold's train window; sliding: every fold's
    test_span: pd.Timedelta
    step: pd.Timedelta                 # how far train_end advances per fold
    embargo: pd.Timedelta
    max_folds: int | None = None
    retrain_every_k: int = 1           # retrain on fold i iff i % k == 0, else reuse the last trained model
    seeds: tuple[int, ...] = (0,)
    # only these fold indices (None = every fold): several processes share one study, each on its folds (owner
    # 2026-11-20); needs retrain_every_k == 1 (a reused model comes from an earlier fold)
    fold_ids: tuple[int, ...] | None = None
    # D-031: a fold has no validation set of its own; the last ``val_share`` of its train window (by time, purged by
    # label end + embargo) is cut off as the validation of epoch selection and early stopping. The test window is
    # never read by training or selection.
    val_share: float = 0.15

    def __post_init__(self):
        if not 0.0 < self.val_share < 1.0:
            raise ValueError(f"val_share must be in (0, 1), got {self.val_share!r}")
        if self.mode not in ROLLING_MODES:
            raise ValueError(f"mode must be one of {ROLLING_MODES}, not {self.mode!r}")
        for name in ("train_span", "test_span", "step", "embargo"):
            object.__setattr__(self, name, pd.Timedelta(getattr(self, name)))
        if min(self.train_span, self.test_span, self.step) <= pd.Timedelta(0) or self.embargo < pd.Timedelta(0):
            raise ValueError("train_span, test_span, step must be > 0 and embargo >= 0")
        if self.retrain_every_k < 1 or (self.max_folds is not None and self.max_folds < 1):
            raise ValueError("retrain_every_k and max_folds must be >= 1")
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        if self.fold_ids is not None:
            ids = tuple(int(i) for i in self.fold_ids)
            if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
                raise ValueError(f"fold_ids must be distinct indices >= 0, got {self.fold_ids!r}")
            if self.retrain_every_k != 1:
                raise ValueError("fold_ids needs retrain_every_k == 1 (a reused model comes from an earlier fold)")
            object.__setattr__(self, "fold_ids", ids)


@dataclass(frozen=True)
class SplitSettings:
    """Train / validation / test by time. ``train_end`` / ``val_end`` (both, ISO instants) win; else the artifact's own
    ``split_dates``; else the ``train_share`` / ``train_share + val_share`` quantiles of the window times."""
    train_end: str = ""
    val_end: str = ""
    train_share: float = 0.80            # TF data/defaults.py train_pct
    val_share: float = 0.10              # TF data/defaults.py val_pct
    embargo_hours: float = 0.0           # 0: the split already purges by label end
    max_rows: int | None = None          # smoke mode (rule E5): at most this many rows per split, evenly strided
    exclude_coins: tuple[str, ...] = ()  # coins whose rows leave train, val AND test (after the time split; names as in
                                         # asset_bounds, with or without USDT, any case); () = keep every coin
    # cross-coin runs (model/cross_coin.py): rows of each split in time-major order, ``max_rows`` keeps WHOLE
    # timestamps (evenly strided over the dates), the inputs get the timestamp-id channel and the trainer cuts
    # batches and evaluation chunks at timestamp borders
    cross_section: bool = False
    # keep only the ``top_coins`` coins with the most TRAIN rows (the longest history; the artifact holds no volume
    # to rank liquidity by), in every split. None = all coins.
    top_coins: int | None = None
    # snapshot panel model (model/snapshot.py, owner 2026-11-18): every row also carries its coin id (the dataset's
    # asset index) as one more input channel, after the features and before the timestamp channel; needs cross_section
    coin_channel: bool = False

    def __post_init__(self):
        coins = (self.exclude_coins,) if isinstance(self.exclude_coins, str) else tuple(self.exclude_coins)
        if not all(isinstance(c, str) and c.strip() for c in coins):
            raise ValueError(f"exclude_coins must be coin names (non-empty strings), got {self.exclude_coins!r}")
        object.__setattr__(self, "exclude_coins", coins)
        if bool(self.train_end) != bool(self.val_end):
            raise ValueError("train_end and val_end are given together or not at all")
        if not (0.0 < self.train_share and 0.0 < self.val_share and self.train_share + self.val_share < 1.0):
            raise ValueError("train_share and val_share must be > 0 and sum to less than 1")
        if self.embargo_hours < 0:
            raise ValueError("embargo_hours must be >= 0")
        if self.top_coins is not None and (isinstance(self.top_coins, bool) or self.top_coins < 1):
            raise ValueError(f"top_coins must be None or an integer >= 1, got {self.top_coins!r}")
        if self.max_rows is not None and (isinstance(self.max_rows, bool) or self.max_rows < 1):
            raise ValueError(f"max_rows must be None or an integer >= 1, got {self.max_rows!r}")
        if self.coin_channel and not self.cross_section:
            raise ValueError("split.coin_channel needs split.cross_section (the snapshot model reads whole timestamps)")


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 5
    batch_size: int = 256
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1                           # the classification heads
    selection_metric: str = "val_raw_loss"
    early_stop: EarlyStopSettings = field(default_factory=EarlyStopSettings)
    device: str = "auto"                                   # "auto" = cuda when available, else cpu
    trace_representative: bool = False                     # trainer trace on the representative batch set (D-019)
    # D-021 switches (TR1 + the B-060 / B-062 candidates); every default reproduces the run before D-021
    steps_per_execution: int = 1                           # Keras steps_per_execution (speed only)
    grad_clip_norm: float | None = None                    # B-018 (TF 1.0)
    ema_decay: float | None = None                         # B-018 (TF use_ema)
    init: str = "torch"                                    # B-007: trainer.switches.INITS ("keras")
    weight_decay_groups: str = "all_params"                # B-018: trainer.switches.WEIGHT_DECAY_GROUPS
    uncertainty_weighting: bool = True                     # B-060: Kendall weights of the class heads
    class_loss_weight: float = 1.0                         # B-060: loss_weight of every classification head
    lambda_reg_end: float = 0.05                           # B-062: end value of the NIG evidence regulariser
    lambda_calib_end: float = 0.1                          # end value of the NIG calibration term
    beta_nll: float = 0.0                                  # FX1 / B-062: NIG NLL x stop_grad(w_St^2)^beta_nll
    beta_nll_normalize: bool = False                       # FX1: beta-NLL weights / their batch mean
    # D-023 switches (FX2, B-060 / gate A5); defaults reproduce the run before D-023
    loss_balance: str = "none"                             # trainer.loss_balance.LOSS_BALANCERS
    kendall_init_logvar: float = 0.0                       # start value of the Kendall log-variances
    # ST1 / D-030: None = today's epoch loop (steps.train); a StageSettings = staged training (steps.train_staged)
    staged: StageSettings | None = None
    # D-032 (panel audit 2026-11-16): a loss term on the NIG means INSIDE each timestamp, on top of the NIG loss
    # (trainer.cross_section_loss.CROSS_SECTION_LOSSES: "none" | "ic" | "rank_ic"); needs split.cross_section
    cross_section_loss: str = "none"
    cross_section_weight: float = 1.0                      # weight of that term (read only when it is on)
    # D-032 (owner idea 2026-11-16): weight of a term that penalises a batch whose close predictions lean long or
    # short (trainer.direction_balance; NIG mean and class probability of the close head); 0 = off
    direction_balance: float = 0.0
    # D-034 (owner 2026-11-17): what one batch is: "rows" = shuffled rows in batch_size chunks (as before),
    # "timestamp" / "day" = ALL training rows of one timestamp / one UTC day form one batch and the batches (not the
    # rows) are shuffled (trainer.trainer.BATCHINGS); batch_size is then not used; not with split.cross_section
    batch_by: str = "rows"
    # owner question 2026-11-18 (D-035, experiment arm, flat model only): "rows" = random rows per batch (today);
    # "time_blocks" = each batch holds whole blocks of ``time_block_bars`` consecutive timestamps (all coins of them),
    # rows inside in time order, the batches shuffled (trainer.groups.BATCH_ORDERS). No extra input channel.
    batch_order: str = "rows"
    time_block_bars: int = 1                               # read only by "time_blocks"; >= 1
    # owner 2026-11-20 (D-039): None = every training row weighs 1 (as before); a DayWeightSettings = every UTC day
    # carries the same total weight and the days the model fails on (high loss, measured each epoch) weigh more
    # (trainer.day_weights); flat epoch loop with row batches only
    day_weights: DayWeightSettings | None = None
    # LS1 (owner 2026-11-19): the losses of the auxiliary close heads (ModelSettings override ``aux_heads``:
    # side_payoff / upper_quantile; read only when the model has them): their fixed loss weight (they are never
    # Kendall-weighted) and the quantile the upper_quantile head learns. The side payoffs use the battery's cost.
    aux_loss_weight: float = 1.0
    aux_quantile: float = 0.9
    # LS1: the side-payoff rows weighted by (close NIG predictive variance)^-this (0 = plain squared error; 1 = WLS)
    aux_variance_weight: float = 0.0
    # LR1 (owner 2026-11-19, "a gap in the settings or the training"): the learning-rate schedule over ``epochs``
    # (trainer.schedules.LR_KINDS; "constant" = lr every epoch, as every run before; "cosine" = from lr down to
    # lr * lr_min_ratio at the last epoch). The run path never passed a schedule before, so every run was constant.
    lr_schedule: str = "constant"
    lr_min_ratio: float = 0.1                              # read only by a non-constant schedule; in (0, 1]

    def __post_init__(self):
        if self.batch_by not in BATCHINGS:
            raise ValueError(f"TrainSettings.batch_by must be one of {sorted(BATCHINGS)}, not {self.batch_by!r}")
        BATCH_ORDERS.entry(self.batch_order)               # unknown names raise now
        for name in ("epochs", "batch_size", "steps_per_execution", "time_block_bars"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"TrainSettings.{name} must be an integer >= 1, got {v!r}")
        if self.lr <= 0 or self.weight_decay < 0 or not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("lr > 0, weight_decay >= 0, 0 <= label_smoothing < 1")
        if min(self.class_loss_weight, self.lambda_reg_end, self.lambda_calib_end, self.beta_nll,
               self.cross_section_weight, self.direction_balance, self.aux_loss_weight, self.aux_variance_weight) < 0:
            raise ValueError("class_loss_weight, lambda_reg_end, lambda_calib_end, beta_nll, cross_section_weight, "
                             "direction_balance, aux_loss_weight and aux_variance_weight must be >= 0")
        if self.lr_schedule not in LR_KINDS:
            raise ValueError(f"TrainSettings.lr_schedule must be one of {sorted(LR_KINDS)}, not {self.lr_schedule!r}")
        if not 0.0 < self.lr_min_ratio <= 1.0:
            raise ValueError(f"TrainSettings.lr_min_ratio must be in (0, 1], got {self.lr_min_ratio!r}")
        if not 0.0 < self.aux_quantile < 1.0:
            raise ValueError(f"TrainSettings.aux_quantile must be in (0, 1), got {self.aux_quantile!r}")


@dataclass(frozen=True)
class BatterySettings:
    """The evaluation battery (eval/battery.py) on the val and test stage of the run (owner goal: profit)."""
    enabled: bool = True
    cost_per_side: float = DEFAULT_COST_PER_SIDE   # charged on EVERY side of a trade: a round trip pays twice this
    n_variants: int = 1                  # variants tried so far (widens the confidence intervals)
    # D-032: also write the day metrics of a multi-coin run (winning days of the sign portfolio and of the top /
    # bottom book against their baselines, uncertainty use): workflow/panel_report.py; read even when ``enabled`` is
    # False (the day metrics have their own files)
    day_metrics: bool = False
    # N8 (audit 2026-11-21): the scorecard verdict's trim (eval.battery_scorecard.TRIM_KEYS: "best" drops the best
    # 5 % of days, "symmetric" the best AND worst), whether every section drops the flat rows (dir_true == 0, chosen
    # by the outcome: E5), and the account's sub-books (0 = read the label horizon from the table, eval.account)
    verdict_trim: str = "best"
    drop_flat: bool = True
    account_books: int = 0

    def __post_init__(self):
        if self.cost_per_side < 0 or self.n_variants < 1:
            raise ValueError("cost_per_side >= 0 and n_variants >= 1")
        if self.verdict_trim not in TRIM_KEYS:
            raise ValueError(f"verdict_trim must be one of {sorted(TRIM_KEYS)}, got {self.verdict_trim!r}")
        if self.account_books < 0:
            raise ValueError("account_books >= 0 (0 = read the horizon from the table)")


@dataclass(frozen=True)
class VerificationSettings:
    """The integrated verification (eval/verification.py, the old TF section 7-d) of a finished run: one verdict table
    over data integrity, baselines, confidence / R:R trades and the candle shape. The cost is the battery's."""
    enabled: bool = True                 # False = ``verification_step`` writes and returns nothing
    target_acc: float = 0.65             # the confident slice must reach this direction accuracy on TEST
    rr: float = 2.0                      # reward : risk of the trades (target at rr x the stop distance)
    stop_mode: str = "nig"               # stop distance: "nig" (NIG std) | "model_levels" (predicted high / low)
    n_boot: int = 500                    # day-block bootstrap draws of every interval
    run_candle: bool = True              # also the candle-shape baseline (fits sklearn models on up to 300 000 rows)
    latest_n: int = 1                    # last test samples per coin in the per-target latest table (0 = no table)
    latest_coins: int = 3                # how many coins (the ones with most test rows) the latest table shows

    def __post_init__(self):
        if not 0.0 < self.target_acc < 1.0 or self.rr <= 0 or self.n_boot < 1:
            raise ValueError("verification: target_acc in (0, 1), rr > 0 and n_boot >= 1")
        if self.latest_n < 0 or self.latest_coins < 1:
            raise ValueError("verification: latest_n >= 0 and latest_coins >= 1")
        if self.stop_mode not in STOP_DISTANCE:
            raise ValueError(f"verification.stop_mode must be one of {sorted(STOP_DISTANCE)}, got {self.stop_mode!r}")


@dataclass(frozen=True)
class EvalSettings:
    chunk: int = 20_000                  # rows per predict chunk (a big split must fit the device)
    health_max_rows: int = 100_000       # evenly strided subsample of the validation rows for the health report
    battery: BatterySettings = field(default_factory=BatterySettings)
    verification: VerificationSettings = field(default_factory=VerificationSettings)

    def __post_init__(self):
        if self.chunk < 1 or self.health_max_rows < 1:
            raise ValueError("chunk and health_max_rows must be >= 1")


#: What every run's model gets on top of ``ANTI_MEMORIZATION_CONFIG`` (the TF ``main.ipynb`` preset, unchanged for
#: the builder's own defaults). D-023 (owner 2026-11-09, FX1 / FX2 switches the E1 evidence showed useful) and D-026
#: (model-first audit 2026-11-14, N-01..N-04: blocking gate fails 7.7 -> 2.7 at 3 epochs), one line per key:
DEFAULT_MODEL_OVERRIDES = {
    "bn_warmup": True,              # BatchNorm moving statistics follow the data from the first step (D-023, N-01)
    "order_detach_close": True,     # high / low offsets no longer drag the close mean (D-023, N-02)
    "nig_scale_fit": "train_std",   # NIG heads in train-fitted label units (D-023), not ``nig_scale`` (D-026)
    "nig_trunk_grad": 0.3,          # share of the NIG heads' gradient that reaches the trunk (D-023)
    "input_scaler": "train_std",    # inputs standardised with train statistics (D-026, N-03)
}


def default_model_settings() -> ModelSettings:
    """The model every run builds unless told otherwise: ``nig_timenet_v2`` with the anti-memorization preset (the TF
    ``main.ipynb`` model, BUILD_PLAN.md V1) plus ``DEFAULT_MODEL_OVERRIDES``. The model before D-023 / D-026 is
    ``ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG))``."""
    return ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, **DEFAULT_MODEL_OVERRIDES})


def model_for(architecture: str | None, overrides: dict | None = None, *, preset: bool = True) -> ModelSettings:
    """The ModelSettings of ``architecture`` (None = the default model's) with ``overrides``: they merge over the
    default model (preset + switches) only for its own architecture and with ``preset``; any other architecture
    starts from ITS registered default config."""
    base = default_model_settings()
    arch = base.architecture if architecture is None else architecture
    keep = preset and arch == base.architecture
    return ModelSettings(arch, {**(base.overrides if keep else {}), **(overrides or {})})


@dataclass(frozen=True)
class RunSettings:
    """Everything one flat single-split run needs, in sections. FROZEN: a field is never assigned (a
    ``FrozenInstanceError`` says so); change it with ``from nig_timenet_torch.core.config import updated`` ->
    ``s = updated(s, {"data.name": path, "data.allow_unchecked": True})``, a copy (dotted keys reach the nested
    sections, an unknown key raises)."""
    run_dir: str                                           # checkpoint and outputs (a Drive folder: a cut run resumes)
    code_sha: str = ""                                     # recorded in the run identity; "" = git HEAD of the package
    data: LoaderSettings = field(default_factory=LoaderSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    model: ModelSettings = field(default_factory=default_model_settings)
    train: TrainSettings = field(default_factory=TrainSettings)
    evaluation: EvalSettings = field(default_factory=EvalSettings)

    def __post_init__(self):
        if not isinstance(self.run_dir, str) or not self.run_dir:
            raise ValueError("RunSettings.run_dir must be a non-empty string")
        if BATCHINGS[self.train.batch_by] and (self.split.cross_section or self.train.staged is not None):
            raise ValueError(f"train.batch_by={self.train.batch_by!r} needs the flat epoch loop: not with "
                             f"split.cross_section (it batches whole timestamps already) or train.staged")
        whole_timestamps = (self.split.cross_section or BATCHINGS[self.train.batch_by]
                            or BATCH_ORDERS.entry(self.train.batch_order).meta["needs_times"])
        if CROSS_SECTION_LOSSES.get(self.train.cross_section_loss) is not None and not whole_timestamps:   # (C-08)
            raise ValueError(f"train.cross_section_loss={self.train.cross_section_loss!r} ranks the rows of one "
                             f"timestamp: it needs whole-timestamp batches (split.cross_section, a grouping "
                             f"train.batch_by or a timestamp train.batch_order such as 'time_blocks')")
        if self.split.cross_section and BATCH_ORDERS.entry(self.train.batch_order).meta["needs_times"]:
            raise ValueError(f"train.batch_order {self.train.batch_order!r} groups rows by timestamp; the panel path "
                             f"(split.cross_section = True) already batches whole timestamps: use batch_order 'rows'")
        if self.train.day_weights is not None and (
                self.split.cross_section or self.train.staged is not None or BATCHINGS[self.train.batch_by]
                or BATCH_ORDERS.entry(self.train.batch_order).meta["needs_times"]):
            raise ValueError("train.day_weights needs the flat epoch loop with batches of shuffled rows: not with "
                             "split.cross_section, train.staged, a grouping batch_by or a timestamp batch_order")
        if self.train.staged is not None and BATCH_ORDERS.entry(self.train.batch_order).meta["needs_times"]:
            raise NotImplementedError(f"train.batch_order {self.train.batch_order!r} is not implemented for staged "
                                      f"training (it samples its own batches from the stage rows)")
