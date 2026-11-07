"""Cross-asset feature columns

PURPOSE: Market context, breadth, momentum rank and orthogonal momentum columns: how a coin relates to the reference
    coin and to the universe.
TAGS: cross asset, market context, breadth, momentum rank, orthogonal momentum
PITFALLS: shared frames are computed once from every coin's high/low/close; missing information gives neutral
    values, not NaN.

Cross-asset feature columns: how a coin relates to the reference coin (market context), how much of the universe
is advancing (breadth), where its momentum ranks (momentum rank) and its momentum with volatility taken out
(orthogonal momentum). The `*_columns` functions name the columns, `build_*` compute the shared frames once from
every coin's high/low/close, and `add_*` attach them to one coin's frames (missing information -> neutral values).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data.settings import FeatureSettings, ScalingSettings, WindowSettings

__all__ = ["market_columns", "rank_columns", "breadth_columns", "orth_columns", "cross_columns",
           "market_return_frame", "add_market_context", "build_breadth", "add_breadth", "build_rank", "add_rank",
           "build_orth", "add_orth"]

LOG_FLOOR = 1e-12


# ------------------------------------------------------------------------------------------------ column names
def market_columns(f: FeatureSettings) -> list[str]:
    m = f.market
    if not m.enabled:
        return []
    cols = [f"MKT_ret_{p}" for p in m.return_periods]
    if m.corr_windows and 1 in m.return_periods:
        cols += [f"MKT_CORR_{w}" for w in m.corr_windows]
    if m.beta_windows and 1 in m.return_periods:
        cols += [f"MKT_BETA_{w}" for w in m.beta_windows]
    cols += [f"MKT_LAG_RET_{k}" for k in m.lag_windows]
    if m.crash_rebound_regime is not None:
        cols.append("MKT_CRASH_REBOUND_REGIME")
    return cols


def rank_columns(f: FeatureSettings) -> list[str]:
    return [f"MOM_RANK_{h}" for h in f.momentum_rank.horizons] if f.momentum_rank.enabled else []


def breadth_columns(f: FeatureSettings) -> list[str]:
    return [f"MKT_BREADTH_{h}" for h in f.breadth.horizons] if f.breadth.enabled else []


def orth_columns(f: FeatureSettings) -> list[str]:
    return ["MOM_ORTH_NATR"] if f.orth_natr.enabled else []


def cross_columns(f: FeatureSettings) -> list[str]:
    """All cross-asset columns in the order they are attached (and appear in the feature matrix)."""
    return market_columns(f) + rank_columns(f) + breadth_columns(f) + orth_columns(f)


# ------------------------------------------------------------------------------------------------ market context
def market_return_frame(price_df: pd.DataFrame, f: FeatureSettings) -> pd.DataFrame:
    """The reference coin's returns (and optional lagged return / crash regime) on its own time index."""
    m = f.market
    close = price_df["close"]
    out = pd.DataFrame(index=price_df.index)
    for p in m.return_periods:
        out[f"MKT_ret_{p}"] = close.pct_change(int(p))
    if m.lag_windows:
        ref_ret_1 = close.pct_change(1)
        for k in m.lag_windows:
            out[f"MKT_LAG_RET_{k}"] = ref_ret_1.shift(int(k))
    crr = m.crash_rebound_regime
    if crr is not None:
        high_lb, recent_w = int(crr.get("high_lookback", 60)), int(crr.get("recent_window", 30))
        drawdown = close / close.rolling(high_lb, min_periods=max(2, high_lb // 2)).max() - 1.0
        out["MKT_CRASH_REBOUND_REGIME"] = (drawdown.rolling(recent_w, min_periods=1).min()
                                           <= float(crr.get("drawdown_threshold", -0.3))).astype("float64")
    return out


def add_market_context(dfs: dict, market_dfs: dict | None, is_reference: bool, f: FeatureSettings) -> dict:
    """Attach the MKT_* columns to every frame; the reference coin itself and a missing frame get zeros."""
    cols = market_columns(f)
    if not cols:
        return dfs
    m = f.market
    own = {f"MKT_CORR_{w}" for w in m.corr_windows} | {f"MKT_BETA_{w}" for w in m.beta_windows}
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        if is_reference or market_dfs is None or tf not in market_dfs:
            for c in cols:
                d[c] = 0.0
        else:
            aligned = market_dfs[tf].reindex(d.index, method="ffill")
            for c in cols:
                if c not in own:
                    d[c] = aligned[c].fillna(0.0) if c in aligned.columns else 0.0
            _attach_relation(d, aligned, own, m)
        out[tf] = d
    return out


def _attach_relation(d: pd.DataFrame, aligned: pd.DataFrame, own: set, m) -> None:
    """MKT_CORR_w / MKT_BETA_w: rolling correlation and beta of the coin's 1-bar return on the reference's."""
    if not own:
        return
    if "MKT_ret_1" not in aligned.columns:
        for c in own:
            d[c] = 0.0
        return
    ret_self = d["close"].astype("float64").pct_change(1)
    ret_ref = aligned["MKT_ret_1"]
    for w in m.corr_windows:
        d[f"MKT_CORR_{w}"] = ret_self.rolling(int(w)).corr(ret_ref).fillna(0.0)
    for w in m.beta_windows:
        cov = ret_self.rolling(int(w)).cov(ret_ref)
        var = ret_ref.rolling(int(w)).var()
        d[f"MKT_BETA_{w}"] = (cov / var.replace(0, np.nan)).fillna(0.0)


# ------------------------------------------------------------------------------------------------ breadth / rank
def build_breadth(data: dict, f: FeatureSettings, tf_order: tuple[str, ...]) -> dict | None:
    """MKT_BREADTH_h per time frame: the share of coins whose h-bar return is positive at each timestamp."""
    if not f.breadth.enabled or not data:
        return None
    out = {}
    for tf in tf_order:
        rets = {a: d[tf]["close"].astype("float64").pct_change(1) for a, d in data.items()
                if tf in d and "close" in d[tf]}
        if not rets:
            continue
        ret_frame = pd.concat(rets, axis=1)
        frame = pd.DataFrame(index=ret_frame.index)
        for h in f.breadth.horizons:
            h_ret = ret_frame.rolling(int(h)).sum() if h > 1 else ret_frame
            advancing = (h_ret > 0).astype("float64").where(h_ret.notna())
            frame[f"MKT_BREADTH_{h}"] = advancing.mean(axis=1, skipna=True).astype("float64")
        out[tf] = frame
    return out


def add_breadth(dfs: dict, breadth_dfs: dict | None, f: FeatureSettings) -> dict:
    cols = breadth_columns(f)
    if not cols:
        return dfs
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        breadth = (breadth_dfs or {}).get(tf)
        aligned = breadth.reindex(d.index, method="ffill") if breadth is not None else None
        for c in cols:
            d[c] = aligned[c].astype("float64").fillna(0.5) if aligned is not None and c in aligned.columns else 0.5
        out[tf] = d
    return out


def build_rank(data: dict, f: FeatureSettings, tf_order: tuple[str, ...]) -> dict | None:
    """MOM_RANK_h: percentile of the coin's h-bar log return among all coins at the same timestamp."""
    if not f.momentum_rank.enabled or not data:
        return None
    out: dict = {asset: {} for asset in data}
    for tf in tf_order:
        rets = {}
        for asset, tf_dict in data.items():
            if tf not in tf_dict or "close" not in tf_dict[tf]:
                continue
            log_close = np.log(tf_dict[tf]["close"].astype("float64").clip(lower=LOG_FLOOR))
            rets[asset] = pd.DataFrame({f"MOM_RANK_{h}": (log_close - log_close.shift(int(h))) * 100.0
                                        for h in f.momentum_rank.horizons})
        for h in f.momentum_rank.horizons:
            col = f"MOM_RANK_{h}"
            if not rets:
                continue
            rank_pct = pd.concat({a: r[col] for a, r in rets.items()}, axis=1).rank(axis=1, pct=True, na_option="keep")
            for asset in rets:
                out[asset].setdefault(tf, pd.DataFrame(index=rank_pct.index))[col] = rank_pct[asset]
    return out


def add_rank(dfs: dict, rank_dfs: dict | None, f: FeatureSettings) -> dict:
    cols = rank_columns(f)
    if not cols:
        return dfs
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        src = (rank_dfs or {}).get(tf)
        for c in cols:
            d[c] = src[c].reindex(d.index).fillna(0.5) if src is not None and c in src.columns else 0.5
        out[tf] = d
    return out


# ------------------------------------------------------------------------------------------------ orthogonal momentum
def _natr(df: pd.DataFrame, length: int) -> pd.Series:
    from nig_timenet_torch.data.ta_backend import ta_module
    ta_module()
    out = df[list(TARGET_COLUMNS)].astype("float64").ta.natr(length=int(length))
    return pd.Series(np.asarray(out, dtype="float64"), index=df.index)


def _rank_residual(y: np.ndarray, x: np.ndarray, min_n: int) -> np.ndarray:
    """Per row: rank(y) with its linear (rank-)dependence on rank(x) removed, scaled to about [-1, 1]."""
    out = np.full(y.shape, np.nan)
    for i in range(y.shape[0]):
        m = np.isfinite(y[i]) & np.isfinite(x[i])
        n = int(m.sum())
        if n < min_n:
            continue
        ry = pd.Series(y[i, m]).rank().to_numpy()
        rx = pd.Series(x[i, m]).rank().to_numpy()
        rx_c = rx - rx.mean()
        denom = float(rx_c @ rx_c)
        beta = float(rx_c @ (ry - ry.mean())) / denom if denom > 0 else 0.0
        out[i, m] = (ry - ry.mean() - beta * rx_c) / (n / 2.0)
    return out


def build_orth(data: dict, f: FeatureSettings, w: WindowSettings, s: ScalingSettings) -> dict | None:
    """MOM_ORTH_NATR: z-scored multi-horizon momentum, rank-orthogonalised to NATR at every timestamp."""
    o = f.orth_natr
    if not o.enabled or not data:
        return None
    horizons = [int(h) for h in o.horizons]
    out: dict = {asset: {} for asset in data}
    for tf in w.tf_order:
        norm_window = int(w.window_sizes.get(tf, 32))
        z_by_h: dict = {h: {} for h in horizons}
        natr = {}
        for asset, tf_dict in data.items():
            df = tf_dict.get(tf)
            if df is None or not set(TARGET_COLUMNS) <= set(df.columns):
                continue
            log_close = np.log(df["close"].astype("float64").clip(lower=LOG_FLOOR))
            for h in horizons:
                r = (log_close - log_close.shift(h)) * 100.0
                mag = 1.4826 * r.abs().rolling(norm_window, min_periods=norm_window).median()
                fallback = r.rolling(norm_window, min_periods=norm_window).std(ddof=0)
                mag = mag.where(mag >= s.eps, np.maximum(fallback, s.eps))
                z_by_h[h][asset] = (r / mag).clip(-s.clip_abs, s.clip_abs)
            natr[asset] = _natr(df, o.natr_length)
        if natr:
            _orth_for_frame(out, tf, natr, z_by_h, horizons, o.min_assets, s.eps)
    return out


def _orth_for_frame(out: dict, tf: str, natr: dict, z_by_h: dict, horizons: list, min_assets: int, eps: float) -> None:
    natr_df = pd.concat(natr, axis=1).sort_index()
    assets, idx = list(natr_df.columns), natr_df.index
    parts = []
    for h in horizons:
        zh = pd.concat(z_by_h[h], axis=1).reindex(index=idx, columns=assets)
        sd = zh.std(axis=1, ddof=0)
        parts.append(zh.div(sd.where(sd > eps), axis=0))
    mom = sum(p.to_numpy() for p in parts) / len(parts)
    res_df = pd.DataFrame(_rank_residual(mom, natr_df.to_numpy(), min_assets), index=idx, columns=assets)
    for asset in assets:
        out.setdefault(asset, {})[tf] = pd.DataFrame({"MOM_ORTH_NATR": res_df[asset]})


def add_orth(dfs: dict, orth_dfs: dict | None, f: FeatureSettings) -> dict:
    if not orth_columns(f):
        return dfs
    out = {}
    for tf, df in dfs.items():
        d = df.copy()
        src = (orth_dfs or {}).get(tf)
        has = src is not None and "MOM_ORTH_NATR" in src.columns
        d["MOM_ORTH_NATR"] = src["MOM_ORTH_NATR"].reindex(d.index).fillna(0.0) if has else 0.0
        out[tf] = d
    return out
