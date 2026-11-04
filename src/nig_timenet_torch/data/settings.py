"""Settings of the native data pipeline (raw candles -> windows -> contract artifact).

One strict, frozen settings tree replaces the global CONFIG dict of crypto-signal-prediction's data/ package: every
section is a frozen dataclass, unknown keys raise (rule B5), ``DataSettings.updated({...})`` is the way a notebook
changes a value, and ``to_dict`` / ``from_dict`` round-trip the whole tree through JSON. Defaults reproduce the
production preset of the reference pipeline (1h candles, 32-bar windows, stride 8, the 18-column feature set).
"""

from dataclasses import dataclass, field
from typing import Any

from nig_timenet_torch.core import config as cfg
from nig_timenet_torch.data.targets import ENTRY_CLOSE_REGS, TARGET_RAW, parse_mode

__all__ = ["SourceSettings", "CustomFeatureSettings", "MarketSettings", "BreadthSettings", "RankSettings",
           "OrthSettings", "FeatureSettings", "WindowSettings", "TargetSettings", "ScalingSettings",
           "RuntimeSettings", "OutputSettings", "DataSettings", "SOURCE_KINDS", "FETCH_SOURCES", "HIGHER_TF_MODES",
           "PRICE_NORM_MODES", "SCALER_TYPES"]

SOURCE_KINDS = ("files", "synthetic")
FETCH_SOURCES = ("", "futures", "vision")
HIGHER_TF_MODES = ("legacy", "closed")
PRICE_NORM_MODES = ("window_scale", "pct_change")
SCALER_TYPES = ("robust", "minmax")
PRICE_TARGETS = ("high", "low", "close")


def _tuple(value: Any) -> Any:
    """Lists (from JSON / notebooks) become tuples, recursively, so a frozen settings tree stays hashable."""
    if isinstance(value, (list, tuple)):
        return tuple(_tuple(v) for v in value)
    return value


def _fix(obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        object.__setattr__(obj, name, _tuple(getattr(obj, name)))


def _check_choice(what: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{what}: one of {allowed}, not {value!r}")


@dataclass(frozen=True)
class SourceSettings:
    """Where the raw candles come from. ``coins`` (explicit names) wins over ``categories``."""
    kind: str = "files"               # a name of data.sources.SOURCES
    root: str = ""                    # folder that holds raw_dir and the registry (the reference's Drive root)
    raw_dir: str = "history_1h"       # <root>/<raw_dir>/<pattern>: one candle file per coin
    pattern: str = "{name}.csv"       # file name; a missing file is retried with/without ".gz"; {interval} allowed
    registry_path: str = "crypto_data/asset_registry.csv"   # <root>/<registry_path>; read by categories=("all_global",)
    coins: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()  # data.coins.COINS_BY_CATEGORY keys, "all", or "all_global" (= the whole registry)
    exclude: tuple[str, ...] = ()     # coins dropped by hand (the sealed holdout is always dropped on top of this)
    tail: int = 0                     # keep only the last `tail` raw rows of every coin (0 = all)
    start: str = ""                   # UTC lower bound of the raw rows (also where a download starts; "" = none)
    end: str = ""                     # UTC upper bound (also where a download stops; "" = none)
    interval: str = "1h"              # raw bar size of the files (and of a download)
    fetch: str = ""                   # "" = read only; "futures" (fapi) or "vision" (data.binance.vision)
    fetch_workers: int = 4            # coins downloaded in parallel
    synthetic_days: int = 70          # synthetic source only: length of each coin in days
    synthetic_seed: int = 40          # synthetic source only: coin i uses seed `synthetic_seed + i`

    def __post_init__(self):
        _check_choice("SourceSettings.kind", self.kind, SOURCE_KINDS)
        _check_choice("SourceSettings.fetch", self.fetch, FETCH_SOURCES)
        _fix(self, ("coins", "categories", "exclude"))
        if self.tail < 0 or self.synthetic_days < 1 or self.fetch_workers < 1:
            raise ValueError("SourceSettings: tail >= 0, synthetic_days >= 1 and fetch_workers >= 1 are required")


@dataclass(frozen=True)
class CustomFeatureSettings:
    """The families of data.custom_features; an empty window list switches a family off."""
    enabled: bool = True
    returns: tuple[int, ...] = ()
    range_windows: tuple[int, ...] = (50,)
    vol_windows: tuple[tuple[int, int], ...] = ()
    volume_window: int = 20
    candle_structure: bool = False
    time_features: bool = False
    fractal_windows: tuple[int, ...] = ()
    rvi_windows: tuple[int, ...] = ()
    fisher_windows: tuple[int, ...] = ()
    vol_estimator_windows: tuple[int, ...] = ()
    mean_reversion_windows: tuple[int, ...] = ()
    jump_windows: tuple[int, ...] = ()
    supertrend_windows: tuple[int, ...] = (10,)
    rsi_divergence_windows: tuple[int, ...] = ()
    psar_enabled: bool = False
    trend_age_windows: tuple[int, ...] = ()
    direction_consistency_windows: tuple[int, ...] = ()
    vol_asymmetry_windows: tuple[int, ...] = ()
    crash_rebound_regime: dict | None = None
    risk_adj_momentum_windows: tuple[int, ...] = ()
    ath_distance_enabled: bool = False
    vol_term_structure_pairs: tuple[tuple[int, int], ...] = ((3, 30),)

    def __post_init__(self):
        _fix(self, tuple(f for f in self.__dataclass_fields__ if f.endswith(("windows", "pairs", "returns"))))


@dataclass(frozen=True)
class MarketSettings:
    """Market context: how a coin relates to the reference coin (correlation / beta of 1-bar returns)."""
    enabled: bool = True
    reference_symbol: str = "BTCUSDT"
    return_periods: tuple[int, ...] = (1,)
    corr_windows: tuple[int, ...] = (20, 60)
    beta_windows: tuple[int, ...] = (20,)
    lag_windows: tuple[int, ...] = ()
    crash_rebound_regime: dict | None = None

    def __post_init__(self):
        _fix(self, ("return_periods", "corr_windows", "beta_windows", "lag_windows"))


@dataclass(frozen=True)
class BreadthSettings:
    """Market breadth: the share of the universe whose return over `h` bars is positive."""
    enabled: bool = True
    horizons: tuple[int, ...] = (24,)

    def __post_init__(self):
        _fix(self, ("horizons",))


@dataclass(frozen=True)
class RankSettings:
    """Cross-sectional momentum rank (percentile of the coin's `h`-bar log return among all coins)."""
    enabled: bool = False
    horizons: tuple[int, ...] = (6, 24)

    def __post_init__(self):
        _fix(self, ("horizons",))


@dataclass(frozen=True)
class OrthSettings:
    """Momentum made orthogonal to volatility (rank residual of momentum on NATR, per timestamp)."""
    enabled: bool = True
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24)
    natr_length: int = 14
    min_assets: int = 5

    def __post_init__(self):
        _fix(self, ("horizons",))


def _default_indicators() -> dict:
    return {"adx": [14], "rsi": [14], "natr": [14], "bbands": [20], "mfi": [14]}


@dataclass(frozen=True)
class FeatureSettings:
    """Indicators (pandas_ta_classic, by name) + the custom families + the cross-asset columns."""
    indicator_settings: dict = field(default_factory=_default_indicators)
    static_indicators: tuple[str, ...] = ()
    static_indicator_params: dict = field(default_factory=dict)
    exclude_from_features: tuple[str, ...] = ("open", "high", "low", "ADX_14", "BBL_20_2.0", "BBM_20_2.0",
                                              "BBU_20_2.0", "BBP_20_2.0", "SUPERT_DIR_10", "MKT_ret_1")
    custom: CustomFeatureSettings = field(default_factory=CustomFeatureSettings)
    market: MarketSettings = field(default_factory=MarketSettings)
    breadth: BreadthSettings = field(default_factory=BreadthSettings)
    momentum_rank: RankSettings = field(default_factory=RankSettings)
    orth_natr: OrthSettings = field(default_factory=OrthSettings)

    def __post_init__(self):
        _fix(self, ("static_indicators", "exclude_from_features"))


@dataclass(frozen=True)
class WindowSettings:
    """Time frames, window lengths and the grid the windows are cut on."""
    tf_order: tuple[str, ...] = ("1h",)       # the first one is the base frame (targets and grid)
    window_sizes: dict = field(default_factory=lambda: {"1h": 32})
    stride: int = 8
    align_to_grid: bool = True                # windows end on bars whose number is a multiple of `stride`
    grid_anchor: str = "1970-01-01"
    higher_tf_mode: str = "legacy"            # "closed": a higher-frame bar is used only once it is complete
    higher_tf_offset: int = 2                 # legacy mode: how many higher-frame bars back the window ends
    sample_filters: tuple[dict, ...] = ()     # {"feature": col, "op": ">", "value": x} or {"expr": "..."}

    def __post_init__(self):
        _fix(self, ("tf_order", "sample_filters"))
        _check_choice("WindowSettings.higher_tf_mode", self.higher_tf_mode, HIGHER_TF_MODES)
        missing = [tf for tf in self.tf_order if tf not in self.window_sizes]
        if not self.tf_order or missing or self.stride < 1:
            raise ValueError(f"WindowSettings: tf_order needs a window size each (missing {missing}) and stride >= 1")


@dataclass(frozen=True)
class TargetSettings:
    """What the six heads are trained to predict (see data.windows for the exact definitions)."""
    targets: tuple[str, ...] = PRICE_TARGETS
    horizon: int = 1                          # forecast horizon in base bars
    reg_target_mode: str = "return"           # a name of data.targets.TARGET_RAW that works one coin at a time
    reg_target_scale: float = 100.0           # y_*_reg = clip(raw, +-clip) * scale (return-unit modes)
    reg_target_clip: float | None = None      # None = the mode's own default (1.0 for return, 10.0 for window_scale)
    entry_close_reg: str = "abs_return"       # close head of the "entry_range" mode: abs_return or range_pos

    def __post_init__(self):
        _fix(self, ("targets",))
        base, relative = parse_mode(self.reg_target_mode)
        if relative or not TARGET_RAW.entry(base).meta["per_row"]:
            raise ValueError(f"TargetSettings.reg_target_mode {self.reg_target_mode!r} needs the whole dataset (a "
                             "cross-section or a train statistic): build with a per-coin mode, then call "
                             "data.targets.retarget")
        _check_choice("TargetSettings.entry_close_reg", self.entry_close_reg, ENTRY_CLOSE_REGS)
        bad = [t for t in self.targets if t not in PRICE_TARGETS]
        if bad or self.horizon < 1:
            raise ValueError(f"TargetSettings: targets must be among {PRICE_TARGETS} (got {bad}), horizon >= 1")


@dataclass(frozen=True)
class ScalingSettings:
    """Per-window normalisation of the inputs."""
    scaler_type: str = "robust"
    clip_abs: float = 5.0
    eps: float = 1e-8
    price_norm_mode: str = "window_scale"
    price_pct_clip: float = 100.0

    def __post_init__(self):
        _check_choice("ScalingSettings.scaler_type", self.scaler_type, SCALER_TYPES)
        _check_choice("ScalingSettings.price_norm_mode", self.price_norm_mode, PRICE_NORM_MODES)
        if not self.price_pct_clip > 0 or self.clip_abs < 0:
            raise ValueError("ScalingSettings: price_pct_clip must be > 0 and clip_abs >= 0")


@dataclass(frozen=True)
class RuntimeSettings:
    """How the build runs; none of these change the numbers."""
    workers: int = 0                          # 0 = automatic; 1 = sequential
    spill_dir: str = ""                       # non-empty: every coin is spilled here and merged on disk (low RAM)
    x_storage_dtype: str = ""                 # "" keeps float32; "float16" halves X on disk
    verbose: bool = True                      # log a summary of the build


@dataclass(frozen=True)
class OutputSettings:
    """Where the artifact is written and the metadata it carries."""
    path: str = ""                            # the artifact folder (npy_dir); "" = build in memory only
    overwrite: bool = False
    split_dates: dict | None = None           # {"train_end": ..., "val_end": ...} copied into the dataset meta
    holdout_start: str | None = None
    source_commit: str = ""                   # 40 hex; "" = this checkout's git HEAD
    producer: str = "nig_timenet_torch.data"


@dataclass(frozen=True)
class DataSettings:
    """The whole data pipeline."""
    source: SourceSettings = field(default_factory=SourceSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    windows: WindowSettings = field(default_factory=WindowSettings)
    targets: TargetSettings = field(default_factory=TargetSettings)
    scaling: ScalingSettings = field(default_factory=ScalingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = True) -> "DataSettings":
        return cfg.from_dict(cls, d, strict=strict)

    def to_dict(self) -> dict:
        return cfg.to_dict(self)

    def updated(self, patch: dict | None = None, /, **changes) -> "DataSettings":
        """A copy with `patch` applied: ``s.updated({"windows.stride": 4, "features.breadth.enabled": False})``."""
        return cfg.updated(self, patch, **changes)
