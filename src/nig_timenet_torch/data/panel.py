"""Cross-asset data: windows grouped by time into a days x coin-slots grid, targets two-dimensional from the start.

A day (the name follows crypto-signal-prediction) is one time bucket of ``group_bars`` base bars; the windows of
different coins that fall into it are its slots (slot = rank among them, so real coins are packed to the left).
Nothing here flattens: ``PanelGrid.rows`` is a (days, slots) table of dataset rows (-1 = padding) and every target is
a (days, slots, K) array, K = len(targets). ``PanelInputs`` is the lazy x side: indexing it with a set of days reads
from the dataset arrays only those rows and returns ``{"x": (B, N, T, F), "mask": (B, N)}`` (a dict per timeframe
when the dataset has several), which is what ``model.grid_panel.GridPanelModel`` and the trainer consume.

Targets (``relative``): "median"/"mean" make the regression target the dataset's y_<t>_reg minus the day's median/mean
over its real slots and the class label "beats the day median/mean" (the TF cross_asset target); "raw" keeps the
dataset's own y_<t>_reg and y_<t>_class. ``y_rank`` is the day's cross-sectional rank in (-0.5, 0.5), the target of
the rank-IC loss. The regression scale is a robust one fitted on the train grid and reused for the others.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nig_timenet_torch.core.prediction_schema import OUTPUT_PREFIX
from nig_timenet_torch.data.settings import _check_choice, _fix

__all__ = ["PanelSettings", "PanelGrid", "PanelInputs", "build_panel", "panel_inputs", "RELATIVE"]

RELATIVE = ("median", "mean", "raw")
TS_COL = 3
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class PanelSettings:
    group_bars: int = 0                       # bars per day bucket; 0 = the dataset's window stride
    min_group: int = 3                        # days with fewer real coins are dropped
    max_slots: int = 0                        # 0 = no cap; else an evenly spaced subset of that many coins per day
    relative: str = "median"                  # "median" | "mean" | "raw"
    reg_scale: float = 0.0                    # 0 = robust scale (1.4826 * MAD) fitted on the train grid; > 0 = fixed
    targets: tuple[str, ...] = ("high", "low", "close")

    def __post_init__(self):
        _fix(self, ("targets",))
        _check_choice("relative", self.relative, RELATIVE)
        if self.group_bars < 0 or self.max_slots < 0 or self.min_group < 1 or self.reg_scale < 0:
            raise ValueError("group_bars, max_slots, reg_scale >= 0 and min_group >= 1 are required")
        if not self.targets:
            raise ValueError("targets must not be empty")


@dataclass(frozen=True)
class PanelGrid:
    rows: np.ndarray            # (D, N) int64 dataset row of each slot, -1 = padding
    mask: np.ndarray            # (D, N) bool
    group: np.ndarray           # (D,) int64 bucket key (bucket start = group * group_ns)
    group_ns: int
    y_cls: np.ndarray           # (D, N, K) float32 in {0, 1}, 0 at padding
    y_reg: np.ndarray           # (D, N, K) float32, relative and scaled, 0 at padding
    y_rank: np.ndarray          # (D, N, K) float32 in (-0.5, 0.5), 0 at padding
    reg_scale: np.ndarray       # (K,) float32: y_reg = (reg - day centre) / reg_scale
    targets: tuple[str, ...]

    @property
    def n_days(self) -> int:
        return len(self.group)

    @property
    def n_slots(self) -> int:
        return self.rows.shape[1]

    def target_arrays(self) -> dict[str, np.ndarray]:
        """The three (D, N, K) target arrays under the names the panel tasks read."""
        return {"y_cls": self.y_cls, "y_reg": self.y_reg, "y_rank": self.y_rank}

    def day_of_row(self) -> np.ndarray:
        """(D, N) int64: the day index of each slot (padding included), for grouped metrics."""
        return np.broadcast_to(np.arange(self.n_days)[:, None], self.rows.shape)


class PanelInputs:
    """Lazy x side of a grid: ``len`` = days, ``[days] -> {"x", "mask"}`` (numpy), reading only the rows asked for."""

    def __init__(self, arrays: dict[str, np.ndarray], grid: PanelGrid, single: bool):
        self.arrays, self.grid, self.single = arrays, grid, single

    def __len__(self) -> int:
        return self.grid.n_days

    def __getitem__(self, days) -> dict:
        days = np.arange(self.grid.n_days)[np.asarray(days) if not isinstance(days, slice) else days]
        days = np.atleast_1d(days)
        rows, mask = self.grid.rows[days], self.grid.mask[days]
        flat = rows[mask]
        out = {}
        for tf, arr in self.arrays.items():
            buf = np.zeros((*mask.shape, *arr.shape[1:]), dtype=arr.dtype)
            buf[mask] = arr[flat]
            out[tf] = buf
        return {"x": next(iter(out.values())) if self.single else out, "mask": mask}


def panel_inputs(dataset: dict, grid: PanelGrid) -> PanelInputs:
    tfs = list(dataset["timeframes"])
    return PanelInputs({tf: dataset[f"X_{tf}"] for tf in tfs}, grid, single=len(tfs) == 1)


def _group_ns(dataset: dict, cfg: PanelSettings) -> int:
    bars = cfg.group_bars or int(dataset.get("stride", 0))
    if bars < 1:
        raise ValueError("group_bars is 0 and the dataset carries no stride: set PanelSettings.group_bars")
    return bars * int(pd.Timedelta(dataset["base_tf"]).value)


def _slot_table(keys: np.ndarray, cfg: PanelSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows (D, N) with -1 padding, bucket keys (D,), counts) for the dataset rows given in ``keys`` order."""
    order = np.argsort(keys, kind="stable")
    uniq, start, full = np.unique(keys[order], return_index=True, return_counts=True)
    keep = full >= cfg.min_group
    uniq, start, full = uniq[keep], start[keep], full[keep]
    count = np.minimum(full, cfg.max_slots) if cfg.max_slots else full
    width = int(count.max()) if len(count) else 0
    rows = np.full((len(uniq), width), -1, dtype=np.int64)
    for d, (s, n, size) in enumerate(zip(start, count, full, strict=True)):
        pick = np.linspace(0, size - 1, n).round().astype(np.int64) if n < size else np.arange(n)
        rows[d, :n] = order[s + pick]
    return rows, uniq, count


def _centre(values: np.ndarray, how: str) -> np.ndarray:
    return np.nanmedian(values, axis=1) if how == "median" else np.nanmean(values, axis=1)


def build_panel(dataset: dict, idx: np.ndarray, cfg: PanelSettings | None = None, *,
                reg_scale: np.ndarray | None = None) -> PanelGrid:
    """The grid of the dataset rows ``idx``. ``reg_scale`` (K,) from the train grid is reused for val and test; with
    ``None`` it is fitted here (cfg.reg_scale if > 0, else the robust scale of this grid's relative targets)."""
    cfg = cfg or PanelSettings()
    idx = np.asarray(idx, dtype=np.int64)
    group_ns = _group_ns(dataset, cfg)
    keys = (np.asarray(dataset["last_candles"])[idx, TS_COL].astype(np.int64)) // group_ns
    local, uniq, _ = _slot_table(keys, cfg)
    rows = np.where(local >= 0, idx[np.maximum(local, 0)], -1)
    mask = local >= 0
    safe = np.maximum(rows, 0)
    k = len(cfg.targets)
    col = lambda t, kind: np.asarray(dataset[f"{OUTPUT_PREFIX}{t}_{kind}"])[safe]  # noqa: E731
    reg_raw = np.stack([col(t, "reg") for t in cfg.targets], axis=-1).astype(np.float64)
    reg_raw[~mask] = np.nan
    cls = np.stack([col(t, "class") for t in cfg.targets], axis=-1)
    if cfg.relative == "raw":
        rel = np.where(mask[..., None], reg_raw, np.nan)
    else:
        rel = reg_raw - np.stack([_centre(reg_raw[..., j], cfg.relative) for j in range(k)], axis=-1)[:, None, :]
        cls = (rel > 0).astype(np.float32)
    if reg_scale is None:
        reg_scale = np.full(k, cfg.reg_scale, np.float64) if cfg.reg_scale > 0 else np.array(
            [MAD_TO_SIGMA * np.nanmedian(np.abs(rel[..., j] - np.nanmedian(rel[..., j]))) for j in range(k)])
    reg_scale = np.asarray(reg_scale, np.float64)
    if not (reg_scale > 0).all():
        raise ValueError(f"reg_scale must be > 0 for every target, got {reg_scale}: too few rows to estimate it")
    valid = mask[..., None]
    n = mask.sum(axis=1)[:, None, None]
    order = np.argsort(np.where(valid, rel, np.inf), axis=1, kind="stable").argsort(axis=1, kind="stable")
    rank = np.where(valid, (order + 0.5) / np.maximum(n, 1) - 0.5, 0.0)
    zero = lambda a: np.where(valid, a, 0.0).astype(np.float32)  # noqa: E731
    return PanelGrid(rows=rows, mask=mask, group=uniq.astype(np.int64), group_ns=group_ns,
                     y_cls=zero(cls), y_reg=zero(rel / reg_scale), y_rank=zero(rank),
                     reg_scale=reg_scale.astype(np.float32), targets=tuple(cfg.targets))
