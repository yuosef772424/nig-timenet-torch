"""Workflow settings (BUILD_PLAN.md P37): the RunSettings-equivalent section that carries the rolling seam.

``RunSettings.rolling`` defaults to None = today's single global split (P25), unchanged. RollingSettings has NO
defaults for the spans: the plan leaves them to the experiment card (E-roll-001, out of scope), so building one
without them raises instead of inventing values.
"""

from dataclasses import dataclass, field

import pandas as pd

from nig_timenet_torch.data.panel import PanelSettings
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings

__all__ = ["RollingSettings", "RunSettings", "ROLLING_MODES", "ROLLING_INITS", "SplitSettings", "TargetRunSettings",
           "TrainSettings", "PanelNetSettings", "PanelLossSettings", "EvalSettings", "RUN_KINDS"]

RUN_KINDS = ("flat", "panel")

ROLLING_MODES = ("expanding", "sliding")
ROLLING_INITS = ("fresh", "warm")


@dataclass(frozen=True, kw_only=True)
class RollingSettings:
    mode: str = "expanding"            # "expanding" | "sliding"
    train_span: pd.Timedelta           # expanding: span of the FIRST fold's train window; sliding: every fold's
    test_span: pd.Timedelta
    step: pd.Timedelta                 # how far train_end advances per fold
    embargo: pd.Timedelta
    max_folds: int | None = None
    retrain_every_k: int = 1           # retrain on fold i iff i % k == 0, else reuse the last trained model
    val_span: pd.Timedelta = pd.Timedelta(0)   # > 0: the last val_span before the cut is the fold's validation set
    init: str = "fresh"                # a retrain starts from "fresh" weights or "warm" (the previous fold's model)
    warm_epochs: int = 0               # init="warm": epochs of a later retrain; 0 = the same as the first fold
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self):
        if self.mode not in ROLLING_MODES:
            raise ValueError(f"mode must be one of {ROLLING_MODES}, not {self.mode!r}")
        for name in ("train_span", "test_span", "step", "embargo", "val_span"):
            object.__setattr__(self, name, pd.Timedelta(getattr(self, name)))
        if min(self.train_span, self.test_span, self.step) <= pd.Timedelta(0) or self.embargo < pd.Timedelta(0):
            raise ValueError("train_span, test_span, step must be > 0 and embargo >= 0")
        if self.val_span < pd.Timedelta(0) or self.val_span >= self.train_span:
            raise ValueError("val_span must be >= 0 and shorter than train_span")
        if self.init not in ROLLING_INITS or self.warm_epochs < 0:
            raise ValueError(f"init must be one of {ROLLING_INITS} and warm_epochs >= 0")
        if self.retrain_every_k < 1 or (self.max_folds is not None and self.max_folds < 1):
            raise ValueError("retrain_every_k and max_folds must be >= 1")
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))


@dataclass(frozen=True)
class SplitSettings:
    """Train / validation / test by time. ``train_end`` / ``val_end`` (ISO instants) win; else the artifact's own
    ``split_dates``; else the ``train_share`` / ``train_share + val_share`` quantiles of the window times."""
    train_end: str = ""
    val_end: str = ""
    train_share: float = 0.80
    val_share: float = 0.10
    embargo_hours: float = 0.0           # 0: the split already purges by label end

    def __post_init__(self):
        if not (0.0 < self.train_share and 0.0 < self.val_share and self.train_share + self.val_share < 1.0):
            raise ValueError("train_share and val_share must be > 0 and sum to less than 1")
        if self.embargo_hours < 0:
            raise ValueError("embargo_hours must be >= 0")


@dataclass(frozen=True)
class TargetRunSettings:
    """Retarget the artifact before training (data.targets.retarget): "" keeps the artifact's targets."""
    mode: str = ""
    clip: float = 0.0                    # 0 = the mode's own clip
    center: str = "median"               # "+relative" modes: the day centre
    group_freq: str = ""                 # "+relative" modes: time bucket of a day ("" = the exact timestamp)


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 30
    batch_size: int = 256
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1                           # flat model's classification heads
    selection_metric: str = "val_raw_loss"
    early_stop: EarlyStopSettings = field(default_factory=EarlyStopSettings)
    device: str = "auto"                                   # "auto" = cuda when available, else cpu
    verbose: bool = True

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("epochs, batch_size >= 1, lr > 0, weight_decay >= 0")


@dataclass(frozen=True)
class PanelNetSettings:
    """The cross-asset network around the shared per-coin encoder (model/panel.py)."""
    d_model: int = 64
    n_cross_layers: int = 1
    num_heads: int = 4
    cross_attention: bool = True
    dropout: float = 0.25
    attn_dropout: float = 0.1
    encoder: dict = field(default_factory=dict)            # overrides of the encoder's builder defaults


@dataclass(frozen=True)
class PanelLossSettings:
    w_cls: float = 1.0                   # weight of the "beats the day centre" logit loss
    w_reg: float = 0.5                   # weight of the scaled relative-return loss
    w_ic: float = 0.5                    # weight of the daily rank-IC loss (0 = off)
    label_smoothing: float = 0.05
    loss_fn: str = "huber"               # "mse" | "mae" | "huber"
    huber_delta: float = 1.0
    ic_output: str = "logit"             # the output the IC loss ranks: "logit" | "mu"


@dataclass(frozen=True)
class EvalSettings:
    batch_size: int = 1024               # flat model: rows per prediction batch
    batch_days: int = 8                  # panel model: days per prediction batch
    max_rows: int = 0                    # 0 = score every test row; else an evenly strided subsample


@dataclass(frozen=True)
class RunSettings:
    """Everything a run needs, in sections; the notebooks change it with ``settings.updated({...})``."""
    kind: str = "flat"                                     # "flat" (one row per coin window) | "panel" (days x coins)
    run_dir: str = ""                                      # checkpoints and outputs; "" = nothing is written
    code_sha: str = ""                                     # recorded in the run identity; "" = git HEAD of the checkout
    data: LoaderSettings = field(default_factory=LoaderSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    targets: TargetRunSettings = field(default_factory=TargetRunSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    panel: PanelSettings = field(default_factory=PanelSettings)
    panel_net: PanelNetSettings = field(default_factory=PanelNetSettings)
    panel_loss: PanelLossSettings = field(default_factory=PanelLossSettings)
    evaluation: EvalSettings = field(default_factory=EvalSettings)
    rolling: RollingSettings | None = None                 # None = the single global split (P25), unchanged

    def __post_init__(self):
        if self.kind not in RUN_KINDS:
            raise ValueError(f"kind must be one of {RUN_KINDS}, not {self.kind!r}")
