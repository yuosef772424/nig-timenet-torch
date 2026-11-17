"""PURPOSE: E4 test environment (owner task 2026-11-16): a synthetic PANEL of coins with one common market factor, a
    different beta per coin and a predictable COIN-SPECIFIC move of known IC, as a contract artifact with oracles.
TAGS: synthetic data, e4 environment, panel, market factor, beta per coin, coin specific signal, known ic, oracle,
    multi coin, cross section, shared volatility regime, contract artifact
PITFALLS: the market move itself is NOT predictable (its conditional mean is 0): absolute direction has a small known
    ceiling and only the move RELATIVE to the market carries signal; the signal is the coin's own recent
    coin-specific innovation, which a reader sees only after removing beta x market from the window (a per-coin
    reader sees it under the market noise). ``oracle_*`` columns are label-side data, never a model input. Rows are
    at stride = horizon, every coin has every timestamp (whole cross-sections). Volatility is one shared regime
    (AR(1) per block), fixed inside a row's horizon, so the oracle moments are exact (``data_io.walk_max``).

The market, per bar: ``substeps`` iid Gaussian steps; coin i's step = ``beta_i * market step + specific step_i +
drift_i``. In block d (``horizon`` bars), with the shared regime ``v_d = exp(AR(1))``:

* market step sd ``market_sigma * v_d / sqrt(substeps)``; specific step sd ``sigma_i * v_d / sqrt(substeps)``;
* drift over the horizon ``a * sqrt(horizon) * sigma_i * v_d * z_i``, ``z_i`` = the standardised sum of coin i's last
  ``signal_bars`` SPECIFIC bar innovations (exactly N(0, 1), independent of the block's noise), ``a = r /
  sqrt(1 - r^2)``, ``r = specific_ic``: the Pearson IC of the standardised specific label move with ``z`` is ``r``.

Why it exists: on real data the close head shows almost no signal (N-08). E4 separates "the model cannot read a
relative signal" from "the data hold none": ``summary['readers']`` gives the IC three readers of the INPUT window
reach (own move, own minus the timestamp mean, own minus beta x market), the ladder a multi-coin model should climb.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS
from nig_timenet_torch.data_io.synthetic import (
    LABEL_CODECS,
    ORACLE_PREFIX,
    bar_kappa,
    dataset_from_rows,
    oracle_keys,
    sim_rows,
)
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics

__all__ = ["PanelSyntheticSettings", "build_panel_synthetic", "write_panel_synthetic", "panel_summary",
           "BETA_FLOOR"]

BETA_FLOOR = 0.2                   # betas are clipped at this (every coin moves WITH the market)


@dataclass(frozen=True)
class PanelSyntheticSettings:
    n_coins: int = 50
    n_bars: int = 2400                 # bars per coin after the burn-in; rows per coin = n_bars // horizon
    burn_in_bars: int = 128            # simulated before the first row (>= window, a multiple of horizon)
    window: int = 32
    horizon: int = 2
    substeps: int = 48
    base_tf: str = "4h"
    start: str = "2023-01-01T00:00:00+00:00"
    market_sigma: float = 0.012        # per-bar sd of the market factor at regime 1
    beta_mean: float = 1.0
    beta_dispersion: float = 0.3       # sd of beta over coins (clipped at BETA_FLOOR)
    specific_sigma_median: float = 0.010   # median over coins of the per-bar coin-specific sd sigma_i
    specific_sigma_dispersion: float = 0.25
    vol_persistence: float = 0.97      # AR(1) coefficient per block of the shared log-volatility regime
    vol_sd: float = 0.35               # stationary sd of the log regime
    specific_ic: float = 0.15          # planted Pearson IC of the standardised coin-specific label move
    signal_bars: int = 3
    label_codec: str = "logret_close"
    reg_target_scale: float = 1.0
    seed: int = 0

    def __post_init__(self):
        for name in ("n_coins", "n_bars", "burn_in_bars", "window", "horizon", "substeps", "signal_bars", "seed"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < (0 if name == "seed" else 1):
                raise ValueError(f"PanelSyntheticSettings.{name} must be an integer >= 1, got {v!r}")
        if self.n_coins < 2 or self.signal_bars > self.window:
            raise ValueError("need n_coins >= 2 and signal_bars <= window")
        if self.burn_in_bars < self.window or self.burn_in_bars % self.horizon or self.n_bars % self.horizon:
            raise ValueError("burn_in_bars >= window, and burn_in_bars and n_bars are multiples of horizon")
        if not (abs(self.specific_ic) < 1.0 and 0.0 <= self.vol_persistence < 1.0 and self.vol_sd >= 0
                and min(self.market_sigma, self.specific_sigma_median, self.reg_target_scale) > 0
                and min(self.beta_dispersion, self.specific_sigma_dispersion) >= 0):
            raise ValueError("need |specific_ic| < 1, 0 <= vol_persistence < 1, vol_sd >= 0, positive sigmas and "
                             "scale, dispersions >= 0")
        if self.label_codec not in LABEL_CODECS:
            raise ValueError(f"label_codec must be one of {sorted(LABEL_CODECS)}, not {self.label_codec!r}")
        pd.Timedelta(self.base_tf), pd.Timestamp(self.start)


def _simulate(s: PanelSyntheticSettings) -> dict:
    """Bar arrays (coins, bars) and per-block parameters; ``eps`` = the standardised SPECIFIC bar innovation."""
    rng = np.random.default_rng(s.seed)
    c, h, k = s.n_coins, s.horizon, s.substeps
    n_bars = s.burn_in_bars + s.n_bars
    n_blocks = n_bars // h
    beta = np.maximum(s.beta_mean + s.beta_dispersion * rng.standard_normal(c), BETA_FLOOR)
    sig_i = np.exp(math.log(s.specific_sigma_median) + s.specific_sigma_dispersion * rng.standard_normal(c))
    log_c0 = 2.0 * rng.standard_normal(c)
    a = s.specific_ic / math.sqrt(1.0 - s.specific_ic ** 2)
    bars = {name: np.zeros((c, n_bars)) for name in ("lh", "ll", "lc", "eps")}
    blocks = {name: np.zeros((c, n_blocks)) for name in ("sigma", "mu", "z_sig", "market", "spec_sd")}
    regime = np.zeros(n_blocks)
    log_v = s.vol_sd * rng.standard_normal()
    for d in range(n_blocks):
        t = d * h - 1
        log_v = s.vol_persistence * log_v + math.sqrt(1.0 - s.vol_persistence ** 2) * s.vol_sd * rng.standard_normal()
        v = math.exp(log_v)
        z = (bars["eps"][:, t - s.signal_bars + 1:t + 1].sum(axis=1) / math.sqrt(s.signal_bars)
             if t >= s.signal_bars - 1 else np.zeros(c))
        mu = a * math.sqrt(h) * sig_i * v * z / (h * k)                     # per step
        m_steps = s.market_sigma * v / math.sqrt(k) * rng.standard_normal((h, k))
        spec = (sig_i * v / math.sqrt(k))[:, None, None] * rng.standard_normal((c, h, k))
        steps = beta[:, None, None] * m_steps[None] + spec + mu[:, None, None]
        path = np.cumsum(steps.reshape(c, h * k), axis=1).reshape(c, h, k)
        start = np.concatenate([np.zeros((c, 1)), path[:, :-1, -1]], axis=1)
        sl = slice(d * h, (d + 1) * h)
        bars["lh"][:, sl] = np.maximum(path.max(axis=2), start) - start
        bars["ll"][:, sl] = np.minimum(path.min(axis=2), start) - start
        bars["lc"][:, sl] = path[:, :, -1] - start
        bars["eps"][:, sl] = spec.sum(axis=2) / (sig_i * v)[:, None]
        blocks["sigma"][:, d] = v * np.sqrt((beta * s.market_sigma) ** 2 + sig_i ** 2)   # per-bar sd of the coin
        blocks["mu"][:, d], blocks["z_sig"][:, d] = mu, z
        blocks["market"][:, d] = m_steps.sum()
        blocks["spec_sd"][:, d] = sig_i * v * math.sqrt(h)
        regime[d] = v
    log_close = log_c0[:, None] + np.cumsum(bars["lc"], axis=1)
    return {**bars, **blocks, "log_close": log_close, "beta": beta, "specific_sigma": sig_i, "regime": regime,
            "kappa": bar_kappa(k)}


def panel_summary(ds: dict) -> dict:
    """The known answers of an E4 dataset, measured on its rows: the planted specific IC, the market's share of the
    close label variance, and three readers of the INPUT window: the coin's own last ``signal_bars`` close moves, the
    same minus the timestamp mean, and the planted signal itself (own minus beta x market, standardised). Per
    reader: the rank IC inside the timestamp (the same for the first two: ranks inside a timestamp do not see its
    mean), the POOLED Pearson correlation with the close label and with the label relative to the timestamp mean
    (what a loss summed over all rows learns from: here the three differ)."""
    s = ds["synthetic"]["settings"]
    y = np.asarray(ds[head_spec("nig_regression", "close").label_key], np.float64)
    ts = np.asarray(ds["last_candles"])[:, LAST_COLUMNS.index("timestamp")].astype(np.int64)
    beta, market = np.asarray(ds[f"{ORACLE_PREFIX}beta"]), np.asarray(ds[f"{ORACLE_PREFIX}market"])
    z, spec_sd = np.asarray(ds[f"{ORACLE_PREFIX}signal"]), np.asarray(ds[f"{ORACLE_PREFIX}specific_sd"])
    grp = pd.Series(y).groupby(ts)
    y_rel = y - grp.transform("mean").to_numpy()
    own = np.asarray(ds[f"X_{s['base_tf']}"], np.float64)[:, -s["signal_bars"]:, 2].sum(axis=1)
    own_rel = own - pd.Series(own).groupby(ts).transform("mean").to_numpy()
    readers = {"own_move": own, "own_minus_timestamp_mean": own_rel, "oracle_signal": z}
    oracle_mean = np.asarray(ds[oracle_keys("close")[0]], np.float64)
    return {"specific_ic_planted": float(s["specific_ic"]),
            "specific_ic_measured": metrics.pearson((y - beta * market) / spec_sd, z),
            "market_share_of_close_variance": float(np.var(beta * market) / np.var(y)),
            "readers": {k: {"rank_ic_by_timestamp": metrics.rank_ic(y, v, ts), "pearson_pooled": metrics.pearson(y, v),
                            "pearson_pooled_relative": metrics.pearson(y_rel, v),
                            "pearson_with_signal": metrics.pearson(v, z)} for k, v in readers.items()},
            "oracle_mean_rank_ic_close": metrics.rank_ic(y, oracle_mean, ts),
            "share_up_close": float((y > 0).mean())}


def build_panel_synthetic(settings: PanelSyntheticSettings | None = None) -> dict:
    """The E4 dataset dict: the keys of the real artifact, the E1 oracle columns (exact conditional mean / sd of every
    label), plus ``oracle_beta``, ``oracle_market`` (the market factor's realised log move over the row's horizon),
    ``oracle_specific_sd`` and ``oracle_regime``; ``synthetic['panel']`` holds ``panel_summary``."""
    s = settings or PanelSyntheticSettings()
    sim = _simulate(s)
    ds = dataset_from_rows(s, sim_rows(s, sim), sim["kappa"], "nig_timenet_torch.data_io.synthetic_panel")
    h = s.horizon
    blocks = np.arange(s.burn_in_bars // h, (s.burn_in_bars + s.n_bars) // h)
    flat = lambda a: np.ascontiguousarray(a).reshape(-1)  # noqa: E731
    ds[f"{ORACLE_PREFIX}beta"] = flat(np.repeat(sim["beta"][:, None], len(blocks), axis=1))
    ds[f"{ORACLE_PREFIX}market"] = flat(sim["market"][:, blocks])
    ds[f"{ORACLE_PREFIX}specific_sd"] = flat(sim["spec_sd"][:, blocks])
    ds[f"{ORACLE_PREFIX}regime"] = flat(np.repeat(sim["regime"][None, blocks], s.n_coins, axis=0))
    ds["environment"] = {"name": "E4", "coins": [b["name"] for b in ds["asset_bounds"]],
                         "rows": int(len(ds["last_candles"]))}
    ds["synthetic"]["panel"] = panel_summary(ds)
    return ds


def write_panel_synthetic(out_dir, settings: PanelSyntheticSettings | None = None, *, source_commit: str = "",
                          overwrite: bool = False) -> dict:
    """Build E4 and write it as a contract artifact (rule D1) that ``data_io.load_dataset`` reads; returns the
    manifest. ``source_commit`` "" = the git HEAD of this package."""
    from nig_timenet_torch.data.artifact_writer import write_artifact

    return write_artifact(build_panel_synthetic(settings), out_dir, source_commit=source_commit, overwrite=overwrite,
                          producer="nig_timenet_torch.data_io.synthetic_panel")
