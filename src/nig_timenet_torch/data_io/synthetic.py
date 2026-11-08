"""PURPOSE: E1 test environment (MODEL_DEV_PROTOCOL section 3): synthetic 4h bars from intrabar random walks, HAR-like
    volatility, a planted close drift of known IC; a contract artifact with each label's exact conditional mean / sd.
TAGS: synthetic data, e1 environment, oracle, planted signal, known ic, heteroscedastic noise, har volatility,
    intrabar path, contract artifact, gates
PITFALLS: rows are at stride = horizon (one row per horizon block) because drift and volatility are constant over a
    row's horizon, which makes the high / low moments exact (data_io.walk_max); the oracle columns (``oracle_*``) are
    label-side data and must never be a model input; label scale 1.0 (the default, the real artifact's) reproduces the
    NIG beta-floor pathology B-052 on purpose.

The market, per coin c (``SyntheticSettings`` names every constant; D-018 records the defaults):

* every bar is ``substeps`` iid Gaussian log steps; a bar's high / low is the max / min of its path INCLUDING its open
  (= the previous close), so high >= max(open, close) >= min(open, close) >= low on every bar;
* bars come in blocks of ``horizon`` bars; a block's per-bar sd ``sigma`` and close drift are fixed at the close of the
  bar before it, from that bar's 32-bar window only (plus the coin scale):
  ``log sigma = (1 - har_weight) log sigma_c + har_weight log(HAR / kappa) - leverage * z_lev`` (clipped to
  ``sigma_c`` x / 8), with ``HAR`` = mean of the last 1 / 6 / ``window`` bar ranges log(high / low),
  ``kappa`` = E[range] / sigma of a ``substeps``-step bar, ``z_lev`` = standardised close innovation of the last
  ``leverage_bars`` bars;
* close drift over the horizon: ``a * sqrt(horizon) * sigma * z_sig`` with ``z_sig`` the standardised innovation of
  the last ``signal_bars`` closes (exactly N(0, 1), independent of the block's noise) and ``a = r / sqrt(1 - r^2)``,
  ``r = close_ic``: the Pearson IC of the standardised close label with ``z_sig`` is exactly ``close_ic``;
* one row per block, at the bar before it: inputs ``X_<base_tf>`` = per bar [log(high / prev close), log(low / prev
  close), log(close / prev close)] over ``window`` bars (the real artifact's channels), labels encoded by the codec
  ``label_codec`` (default ``logret_close``: declared like the real artifact, reg_target_mode "return" +
  price_norm_mode "logret_std"), class labels = own future > own last price.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io import walk_max
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS, anchor_and_future, codec_name, own_anchor_and_future
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics

__all__ = ["SyntheticSettings", "ORACLE_PREFIX", "oracle_keys", "build_synthetic", "write_synthetic",
           "oracle_summary", "bar_kappa", "LABEL_CODECS"]

ORACLE_PREFIX = "oracle_"
_HAR_LAGS = (1, 6)                 # plus the whole window
_SIGMA_CLIP = math.log(8.0)        # log sigma stays within log sigma_c +- this
_PRICE_LOG_SD = 2.0                # first price of a coin: exp(N(0, 2))
_Z80 = 1.2815515655446004          # Phi^-1(0.9): the symmetric 80 % interval mean +- this x sd


def _log_moments(p: dict, sc: float) -> dict:
    """Oracle (mean, sd) per head in label units for the ``logret_close`` codec: y = log(future / last_close)."""
    hi_m, hi_v = walk_max.max_moments(p["mu"], p["s"], p["n"])
    lo_m, lo_v = walk_max.max_moments(-p["mu"], p["s"], p["n"])
    return {"high": (hi_m * sc, np.sqrt(hi_v) * sc), "low": (-lo_m * sc, np.sqrt(lo_v) * sc),
            "close": (p["n"] * p["mu"] * sc, math.sqrt(p["n"]) * p["s"] * sc)}


def _return_moments(p: dict, sc: float) -> dict:
    """Oracle (mean, sd) per head for the ``return`` codec: y = future / own last price - 1."""
    out = {}
    for t, sign, last in (("high", 1.0, p["last_high"]), ("low", -1.0, p["last_low"])):
        e1 = walk_max.max_mgf(sign * p["mu"], p["s"], p["n"], sign)
        e2 = walk_max.max_mgf(sign * p["mu"], p["s"], p["n"], 2.0 * sign)
        ratio = p["last_close"] / last
        out[t] = ((ratio * e1 - 1.0) * sc, ratio * np.sqrt(np.maximum(e2 - e1 * e1, 0.0)) * sc)
    m, v = p["n"] * p["mu"], p["n"] * p["s"] ** 2
    out["close"] = ((np.exp(m + 0.5 * v) - 1.0) * sc, np.sqrt(np.expm1(v)) * np.exp(m + 0.5 * v) * sc)
    return out


#: label codec -> (oracle moments, the price_norm_mode the artifact declares so that codec_name picks this codec)
LABEL_CODECS = {"logret_close": (_log_moments, "logret_std"), "return": (_return_moments, None)}


@dataclass(frozen=True)
class SyntheticSettings:
    n_coins: int = 32
    n_bars: int = 4000                 # bars per coin after the burn-in; rows per coin = n_bars // horizon
    burn_in_bars: int = 128            # simulated before the first row (>= window, a multiple of horizon)
    window: int = 32                   # input bars per row (the real artifact's)
    horizon: int = 2                   # label bars (the real artifact's forecast_horizon) = row stride
    substeps: int = 48                 # intrabar steps per bar
    base_tf: str = "4h"
    start: str = "2023-01-01T00:00:00+00:00"   # open time of the first simulated bar
    coin_sigma_median: float = 0.02    # median over coins of the per-bar log sd scale sigma_c
    coin_sigma_dispersion: float = 0.25   # sd of log sigma_c over coins
    har_weight: float = 0.7            # weight of the HAR term in log sigma (1 - this on the coin scale)
    leverage: float = 0.2              # vol rises by exp(leverage) per unit of past standardised down move
    leverage_bars: int = 6
    close_ic: float = 0.05             # planted Pearson IC of the standardised close label (protocol: 0.05 / 0.10)
    signal_bars: int = 3
    label_codec: str = "logret_close"  # a key of LABEL_CODECS (a TARGET_CODECS name)
    reg_target_scale: float = 1.0      # y_*_reg = codec label x this (1.0 = the real artifact, B-052)
    seed: int = 0

    def __post_init__(self):
        for name in ("n_coins", "n_bars", "burn_in_bars", "window", "horizon", "substeps", "leverage_bars",
                     "signal_bars", "seed"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < (0 if name == "seed" else 1):
                raise ValueError(f"SyntheticSettings.{name} must be an integer >= 1, got {v!r}")
        if self.window < max(_HAR_LAGS) or max(self.leverage_bars, self.signal_bars) > self.window:
            raise ValueError("window must cover the HAR lags, leverage_bars and signal_bars")
        if self.burn_in_bars < self.window or self.burn_in_bars % self.horizon or self.n_bars % self.horizon:
            raise ValueError("burn_in_bars >= window, and burn_in_bars and n_bars are multiples of horizon")
        if not (abs(self.close_ic) < 1.0 and 0.0 <= self.har_weight < 1.0 and self.coin_sigma_median > 0
                and self.coin_sigma_dispersion >= 0 and self.reg_target_scale > 0):
            raise ValueError("need |close_ic| < 1, 0 <= har_weight < 1, coin_sigma_median > 0, "
                             "coin_sigma_dispersion >= 0, reg_target_scale > 0")
        if self.label_codec not in LABEL_CODECS:
            raise ValueError(f"label_codec must be one of {sorted(LABEL_CODECS)}, not {self.label_codec!r}")
        pd.Timedelta(self.base_tf), pd.Timestamp(self.start)


def oracle_keys(target: str) -> tuple[str, str]:
    """(mean column, sd column) of a head's oracle, in label units."""
    key = head_spec("nig_regression", target).label_key
    return f"{ORACLE_PREFIX}{key}_mean", f"{ORACLE_PREFIX}{key}_sd"


def bar_kappa(substeps: int) -> float:
    """E[log(high / low)] / sigma of one bar of ``substeps`` iid zero-drift steps (high / low include the open):
    2 E[M_K] / sigma = 2 / sqrt(2 pi K) * sum_k k^-1/2."""
    k = np.arange(1, substeps + 1, dtype=np.float64)
    return float(2.0 / math.sqrt(2.0 * math.pi * substeps) * np.sum(k ** -0.5))


def _simulate(s: SyntheticSettings) -> dict:
    """Bar arrays (coins, bars) of the whole simulation plus the per-block parameters of every block."""
    rng = np.random.default_rng(s.seed)
    c, h, k, w = s.n_coins, s.horizon, s.substeps, s.window
    n_bars = s.burn_in_bars + s.n_bars
    log_sc = math.log(s.coin_sigma_median) + s.coin_sigma_dispersion * rng.standard_normal(c)
    log_c0 = _PRICE_LOG_SD * rng.standard_normal(c)
    a = s.close_ic / math.sqrt(1.0 - s.close_ic ** 2)
    kappa = bar_kappa(k)
    bars = {name: np.zeros((c, n_bars)) for name in ("lh", "ll", "lc", "eps")}
    blocks = {name: np.zeros((c, n_bars // h)) for name in ("sigma", "mu", "z_sig")}
    for d in range(n_bars // h):
        t = d * h - 1                                   # the bar the block's parameters are fixed at
        log_sig, z_sig = log_sc.copy(), np.zeros(c)
        if t >= w - 1:
            rng_ = bars["lh"][:, t - w + 1:t + 1] - bars["ll"][:, t - w + 1:t + 1]
            har = (rng_[:, -1] + rng_[:, -_HAR_LAGS[1]:].mean(axis=1) + rng_.mean(axis=1)) / 3.0
            z_lev = bars["eps"][:, t - s.leverage_bars + 1:t + 1].sum(axis=1) / math.sqrt(s.leverage_bars)
            log_sig = ((1.0 - s.har_weight) * log_sc + s.har_weight * np.log(np.maximum(har, 1e-12) / kappa)
                       - s.leverage * z_lev)
            log_sig = np.clip(log_sig, log_sc - _SIGMA_CLIP, log_sc + _SIGMA_CLIP)
            z_sig = bars["eps"][:, t - s.signal_bars + 1:t + 1].sum(axis=1) / math.sqrt(s.signal_bars)
        sigma = np.exp(log_sig)
        mu = a * math.sqrt(h) * sigma * z_sig / (h * k)            # per step
        steps = mu[:, None, None] + (sigma / math.sqrt(k))[:, None, None] * rng.standard_normal((c, h, k))
        path = np.cumsum(steps.reshape(c, h * k), axis=1).reshape(c, h, k)
        start = np.concatenate([np.zeros((c, 1)), path[:, :-1, -1]], axis=1)      # each bar's open vs the last close
        close = path[:, :, -1]
        sl = slice(d * h, (d + 1) * h)
        prev = start                                                    # log open = log previous close
        bars["lh"][:, sl] = np.maximum(path.max(axis=2), start) - prev
        bars["ll"][:, sl] = np.minimum(path.min(axis=2), start) - prev
        bars["lc"][:, sl] = close - prev
        bars["eps"][:, sl] = (bars["lc"][:, sl] - mu[:, None] * k) / sigma[:, None]
        blocks["sigma"][:, d], blocks["mu"][:, d], blocks["z_sig"][:, d] = sigma, mu, z_sig
    log_close = log_c0[:, None] + np.cumsum(bars["lc"], axis=1)
    return {**bars, **blocks, "log_close": log_close, "log_sigma_c": log_sc, "kappa": kappa}


def _rows(s: SyntheticSettings, sim: dict) -> dict:
    """Row-aligned arrays (coin-major): inputs, last_candles and block parameters of every written row."""
    h, w = s.horizon, s.window
    blocks = np.arange(s.burn_in_bars // h, (s.burn_in_bars + s.n_bars) // h)
    t = blocks * h - 1
    lc_prev = np.concatenate([sim["log_close"][:, :1] - sim["lc"][:, :1], sim["log_close"][:, :-1]], axis=1)
    log_h, log_l, log_c = lc_prev + sim["lh"], lc_prev + sim["ll"], sim["log_close"]
    feats = np.stack([sim["lh"], sim["ll"], sim["lc"]], axis=2)                 # (coins, bars, 3)
    win = np.lib.stride_tricks.sliding_window_view(feats, w, axis=1)            # (coins, bars-w+1, 3, w)
    x = win[:, t - w + 1].transpose(0, 1, 3, 2)                                 # (coins, rows, w, 3)
    fut = t[:, None] + np.arange(1, h + 1)
    bar_ns = int(pd.Timedelta(s.base_tf).value)
    ts = pd.Timestamp(s.start).tz_convert("UTC").value + t.astype(np.int64) * bar_ns
    cols = {"last_high": log_h[:, t], "last_low": log_l[:, t], "last_close": log_c[:, t],
            "future_close": log_c[:, t + h], "future_low_min": log_l[:, fut].min(axis=2),
            "future_high_max": log_h[:, fut].max(axis=2)}
    last = np.zeros((s.n_coins, len(t), len(LAST_COLUMNS)))
    for name, v in cols.items():
        last[:, :, LAST_COLUMNS.index(name)] = np.exp(v)
    last[:, :, TS_COL] = ts[None, :].astype(np.float64)
    closes = np.exp(np.lib.stride_tricks.sliding_window_view(log_c, w, axis=1)[:, t - w + 1])   # (coins, rows, w)
    q1, med, q3 = np.quantile(closes, [0.25, 0.5, 0.75], axis=2)
    flat = lambda a: a.reshape(s.n_coins * len(t), *a.shape[2:])  # noqa: E731
    return {"X": flat(x).astype(np.float32), "last": flat(last), "base": flat(np.stack([med, q3 - q1], axis=2)),
            "sigma": flat(sim["sigma"][:, blocks]), "mu": flat(sim["mu"][:, blocks]),
            "z_sig": flat(sim["z_sig"][:, blocks]), "rows_per_coin": len(t)}


def build_synthetic(settings: SyntheticSettings | None = None) -> dict:
    """The E1 dataset dict (the keys of the real TF artifact, plus the ``oracle_*`` columns and a ``synthetic`` meta
    block); ``write_synthetic`` writes it as a contract artifact."""
    s = settings or SyntheticSettings()
    sim = _simulate(s)
    r = _rows(s, sim)
    targets = list(TARGET_COLUMNS)
    moments_fn, norm_mode = LABEL_CODECS[s.label_codec]
    ds = {"base_params": r["base"].astype(np.float32), "last_candles": r["last"], "mode": "direction_robust",
          "scaler_type": "robust", "feature_order": list(TARGET_COLUMNS), "timeframes": [s.base_tf],
          "targets": [head_spec(kind, t).name for t in targets for kind in ("binary_classification",
                                                                               "nig_regression")],
          "price_targets": targets,
          "asset_bounds": [{"name": f"SYN{i:03d}USDT", "start": i * r["rows_per_coin"],
                            "end": (i + 1) * r["rows_per_coin"]} for i in range(s.n_coins)],
          "base_tf": s.base_tf, "window_sizes": {s.base_tf: s.window}, "stride": s.horizon,
          "forecast_horizon": s.horizon, "reg_target_mode": "return", "reg_target_scale": float(s.reg_target_scale),
          "split_dates": None, "holdout_start": None, "skipped_assets": {}}
    if norm_mode is not None:
        ds.update(price_norm_mode=norm_mode, price_norm_state="unfitted")
    if codec_name(ds) != s.label_codec:
        raise AssertionError(f"the declared target definition decodes as {codec_name(ds)!r}, not {s.label_codec!r}")
    state = TARGET_CODECS[s.label_codec]().fit(ds)
    codec, lc = TARGET_CODECS[s.label_codec](), r["last"]
    ds[f"X_{s.base_tf}"] = r["X"]
    for t in targets:
        own = own_anchor_and_future(lc, t)                       # class: own future above own last price
        ds[head_spec("binary_classification", t).label_key] = (own[:, 1] > own[:, 0]).astype(np.float32)
        ds[head_spec("nig_regression", t).label_key] = codec.encode(anchor_and_future(lc, t, state), state).astype(
            np.float32)
    params = {"mu": r["mu"], "s": r["sigma"] / math.sqrt(s.substeps), "n": s.horizon * s.substeps,
              **{name: lc[:, LAST_COLUMNS.index(name)] for name in ("last_high", "last_low", "last_close")}}
    for t, (mean, sd) in moments_fn(params, s.reg_target_scale).items():
        ds[oracle_keys(t)[0]], ds[oracle_keys(t)[1]] = mean, sd
    ds[f"{ORACLE_PREFIX}sigma"], ds[f"{ORACLE_PREFIX}signal"] = r["sigma"], r["z_sig"]
    ds["synthetic"] = {"generator": "nig_timenet_torch.data_io.synthetic", "settings": asdict(s),
                       "kappa": sim["kappa"], "codec_state": state}
    ds["synthetic"]["summary"] = oracle_summary(ds)
    return ds


def oracle_summary(dataset: dict) -> dict:
    """What the oracle scores on its own data, per head: rank IC by timestamp (the gates' IC), the T-sharp share
    (oracle sd below the oracle's own predicted move, in price units, as eval.target_report measures it), the coverage
    of the SYMMETRIC 80 % interval mean +- 1.28 sd (about 0.88 for high / low: their labels are skewed, so even the
    true mean and sd over-cover with a symmetric head), the label sd; plus the planted close IC (Pearson of the
    standardised close label with the planted signal)."""
    lc = np.asarray(dataset["last_candles"])
    state = TARGET_CODECS[codec_name(dataset)]().fit(dataset)
    codec = TARGET_CODECS[state["mode"]]()
    ts, last_close = lc[:, TS_COL], lc[:, LAST_COLUMNS.index("last_close")]
    out = {}
    for t in dataset["price_targets"]:
        y = np.asarray(dataset[head_spec("nig_regression", t).label_key], np.float64)
        mean, sd = (np.asarray(dataset[k]) for k in oracle_keys(t))
        anchor = anchor_and_future(lc, t, state)[:, 0]
        pred = codec.decode(mean, state, anchor)
        out[t] = {"rank_ic_by_timestamp": metrics.rank_ic(y, mean, groups=ts),
                  "share_sd_below_move": metrics.width_below_share(codec.decode(mean + sd, state, anchor) - pred,
                                                                   np.abs(pred - last_close)),
                  "coverage_symmetric_80": metrics.interval_coverage(y, mean - _Z80 * sd, mean + _Z80 * sd),
                  "label_sd": float(y.std()), "label_mean_abs": float(np.abs(y).mean())}
    sd_c = np.asarray(dataset[oracle_keys("close")[1]])
    y_c = np.asarray(dataset[head_spec("nig_regression", "close").label_key], np.float64)
    out["close"]["planted_ic_pearson"] = metrics.pearson(y_c / sd_c, np.asarray(dataset[f"{ORACLE_PREFIX}signal"]))
    return out


def write_synthetic(out_dir, settings: SyntheticSettings | None = None, *, source_commit: str = "",
                    overwrite: bool = False) -> dict:
    """Build E1 and write it as a contract artifact (rule D1) that ``data_io.load_dataset`` reads; returns the
    manifest. ``source_commit`` "" = the git HEAD of this package (the generator's code)."""
    from nig_timenet_torch.data.artifact_writer import write_artifact

    return write_artifact(build_synthetic(settings), out_dir, source_commit=source_commit, overwrite=overwrite,
                          producer="nig_timenet_torch.data_io.synthetic")
