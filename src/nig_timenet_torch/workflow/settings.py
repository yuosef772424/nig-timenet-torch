"""PURPOSE: Settings of the workflow layer: the strict RunSettings tree of one flat single-split run, and the rolling
    seam (RollingSettings) that the I4 package will connect to the run steps.
TAGS: run settings, strict settings tree, split settings, train settings, evaluation settings, rolling seam
PITFALLS: RunSettings has no rolling field (rolling and panel runs are package I4); every field is read by
    workflow/run*.py; RollingSettings has NO defaults for the spans (the experiment card decides them). A new field's
    default must reproduce the baseline job (tests/jobs/test_baseline_pin.py).

``RunSettings`` is everything ``workflow.run`` needs for ONE run on ONE chronological split: the data, the split, the
model, the training and the evaluation (health report, test report, battery). The jobs (``jobs/baseline.py``) and the
notebooks build it (``from_dict`` / ``updated``) and call the steps; nothing else configures a run. ``RollingSettings``
is the walk-forward seam (BUILD_PLAN.md P37) used by ``workflow/rolling.py``.
"""

from dataclasses import dataclass, field

import pandas as pd

from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings

__all__ = ["RollingSettings", "RunSettings", "ROLLING_MODES", "SplitSettings", "TrainSettings", "EvalSettings",
           "BatterySettings", "default_model_settings"]

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

    def __post_init__(self):
        if self.mode not in ROLLING_MODES:
            raise ValueError(f"mode must be one of {ROLLING_MODES}, not {self.mode!r}")
        for name in ("train_span", "test_span", "step", "embargo"):
            object.__setattr__(self, name, pd.Timedelta(getattr(self, name)))
        if min(self.train_span, self.test_span, self.step) <= pd.Timedelta(0) or self.embargo < pd.Timedelta(0):
            raise ValueError("train_span, test_span, step must be > 0 and embargo >= 0")
        if self.retrain_every_k < 1 or (self.max_folds is not None and self.max_folds < 1):
            raise ValueError("retrain_every_k and max_folds must be >= 1")
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))


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

    def __post_init__(self):
        if bool(self.train_end) != bool(self.val_end):
            raise ValueError("train_end and val_end are given together or not at all")
        if not (0.0 < self.train_share and 0.0 < self.val_share and self.train_share + self.val_share < 1.0):
            raise ValueError("train_share and val_share must be > 0 and sum to less than 1")
        if self.embargo_hours < 0:
            raise ValueError("embargo_hours must be >= 0")
        if self.max_rows is not None and (isinstance(self.max_rows, bool) or self.max_rows < 1):
            raise ValueError(f"max_rows must be None or an integer >= 1, got {self.max_rows!r}")


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

    def __post_init__(self):
        for name in ("epochs", "batch_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"TrainSettings.{name} must be an integer >= 1, got {v!r}")
        if self.lr <= 0 or self.weight_decay < 0 or not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("lr > 0, weight_decay >= 0, 0 <= label_smoothing < 1")


@dataclass(frozen=True)
class BatterySettings:
    """The evaluation battery (eval/battery.py) on the val and test stage of the run (owner goal: profit)."""
    enabled: bool = True
    cost_per_side: float = 0.0008        # charged on EVERY side of a trade: a round trip pays twice this
    n_variants: int = 1                  # variants tried so far (widens the confidence intervals)

    def __post_init__(self):
        if self.cost_per_side < 0 or self.n_variants < 1:
            raise ValueError("cost_per_side >= 0 and n_variants >= 1")


@dataclass(frozen=True)
class EvalSettings:
    chunk: int = 20_000                  # rows per predict chunk (a big split must fit the device)
    health_max_rows: int = 100_000       # evenly strided subsample of the validation rows for the health report
    battery: BatterySettings = field(default_factory=BatterySettings)

    def __post_init__(self):
        if self.chunk < 1 or self.health_max_rows < 1:
            raise ValueError("chunk and health_max_rows must be >= 1")


def default_model_settings() -> ModelSettings:
    """The model every run builds unless told otherwise: ``nig_timenet_v2`` with the anti-memorization preset (the TF
    ``main.ipynb`` model, BUILD_PLAN.md V1)."""
    return ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG))


@dataclass(frozen=True)
class RunSettings:
    """Everything one flat single-split run needs, in sections; notebooks change it with ``updated({...})``."""
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
