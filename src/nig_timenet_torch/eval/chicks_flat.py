"""PURPOSE: The flat prediction frame of the TF chicks report (one row per coin x head x sample) that every report
    reads, and the latest-samples table printed under each coin.
TAGS: chicks, flat dataframe, flat predictions, latest samples table, signal, confidence, uncertainty, expected move,
    per asset results, tf parity
PITFALLS: the frame is built from the per-asset results of ``eval.chicks_assets`` (a DataFrame with one dict per
    head), row order = coin, then head, then sample: the batch report cuts it in consecutive blocks by that order;
    ``correct`` and ``direction_correct`` stay integers, every other number is float64; the ``*_ret`` columns are the
    uncertainties as a share of the entry price (unit-free, what pattern discovery reads); a row without a true price
    is left out; the latest table shows ``#`` (the row number) instead of a date, because the chicks report passes no
    timestamps, and a head without a price is shown as ``—``.
SOURCE: crypto-signal-prediction ``evaluation/flat.py`` (build_flat_dataframe) and ``evaluation/latest_table.py``
    (build_latest_table, format_latest_table, print_latest_table).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_outputs import fmt_num, fmt_price
from nig_timenet_torch.eval.chicks_text import Say

__all__ = ["build_flat_frame", "latest_table_frame", "format_latest_frame", "print_latest_frame", "latest_legend"]

_UNC = ("uncertainty", "aleatoric", "epistemic")


def _head_rows(asset: str, target: str, d: dict) -> pd.DataFrame | None:
    """The flat rows of one coin x head (only the samples with a true price), or None when it has no truth."""
    if "true_real" not in d:
        return None
    keep = np.asarray(d["has_truth"], dtype=bool)
    n = len(d["pred_real"])
    if not keep.any():
        return None
    nan = np.full(n, np.nan)
    pred, true, entry = (np.asarray(d[k], dtype=np.float64) for k in ("pred_real", "true_real", "entry_price"))
    direction = np.asarray(d["direction_correct"], dtype=np.int32)
    change = pred - entry
    cols = {"asset": [asset] * n, "target": [target] * n, "kind": ["continuous"] * n, "pred": pred, "true": true,
            "entry": entry, "abs_error": np.asarray(d["error_abs"], dtype=np.float64),
            "pct_error": np.asarray(d["error_pct"], dtype=np.float64),
            "confidence": np.asarray(d.get("confidence", nan), dtype=np.float64),
            "uncertainty": np.asarray(d.get("uncertainty_real", nan), dtype=np.float64),
            "aleatoric": np.asarray(d.get("aleatoric", nan), dtype=np.float64),
            "epistemic": np.asarray(d.get("epistemic", nan), dtype=np.float64),
            "unc_clipped": np.asarray(d.get("unc_clipped", np.zeros(n, dtype=bool)), dtype=bool),
            "direction_correct": direction.astype(np.int64), "correct": direction.astype(np.int64),
            "predicted_change": change, "price_change": true - entry,
            "timestamp": np.asarray(d.get("timestamp", nan), dtype=np.float64),
            "p_up": np.asarray(d.get("p_up", nan), dtype=np.float64),
            "predicted_change_pct": change / (np.abs(entry) + 1e-7) * 100.0}
    return pd.DataFrame(cols)[keep]


def build_flat_frame(per_asset: pd.DataFrame, targets) -> pd.DataFrame:
    """The TF ``build_flat_dataframe``: ``per_asset`` is the per-coin results (columns ``asset``, ``n_samples`` and
    one dict per head), ``targets`` the head names in report order."""
    parts = []
    for _, row in per_asset.iterrows():
        for name in targets:
            if name in row and isinstance(row[name], dict):
                part = _head_rows(row["asset"], name, row[name])
                if part is not None:
                    parts.append(part)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not df.empty and "entry" in df.columns:
        den = df["entry"].abs() + 1e-7
        for c in _UNC:
            df[f"{c}_ret"] = df[c] / den
    for c in [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("correct", "direction_correct")]:
        df[c] = df[c].astype(np.float64)
    return df


def latest_table_frame(decoded: dict, targets, n_display: int = 5, asset: str | None = None) -> pd.DataFrame:
    """The last ``n_display`` samples of each head of one coin (the newest at the bottom): row number, head, signal
    (up / down arrow), confidence %, uncertainty % of the predicted price, expected move %, predicted and true price,
    the entry and whether the direction was right (None when there is no true price)."""
    names = [t for t in targets if isinstance(decoded.get(t), dict)]
    if not names or not n_display or n_display <= 0:
        return pd.DataFrame()
    n = min(len(decoded[t]["pred_real"]) for t in names)
    k = int(min(n_display, n))
    rows = []
    for i in range(n - k, n):
        for name in names:
            r = decoded[name]
            pred = float(r["pred_real"][i])
            entry = float(r["entry_price"][i])
            entry_ok = np.isfinite(entry) and entry != 0.0
            move = (pred - entry) / (abs(entry) + 1e-7) * 100 if entry_ok else float("nan")
            unc = float(r["uncertainty_real"][i])
            true = r.get("true_real")
            if true is None:
                true = r.get("target_real")
            true_i = float(true[i]) if true is not None and np.isfinite(true[i]) else float("nan")
            has_true = bool(np.isfinite(true_i))
            row = {"idx": i, "date": pd.NaT, "target": name, "kind": "continuous"}
            if asset is not None:
                row["asset"] = asset
            row.update({
                "signal": "—" if not np.isfinite(move) else ("▲" if move > 0 else "▼" if move < 0 else "■"),
                "confidence_%": float(r["confidence"][i]) * 100,
                "uncertainty_%": unc / (abs(pred) + 1e-7) * 100 if np.isfinite(unc) else float("nan"),
                "move_%": move, "pred": pred, "pred_lo": pred - unc, "pred_hi": pred + unc,
                "entry": entry if entry_ok else float("nan"), "true": true_i, "true_label": None,
                "has_true": has_true,
                "ok": bool(np.sign(pred - entry) == np.sign(true_i - entry)) if (has_true and entry_ok) else None})
            rows.append(row)
    return pd.DataFrame(rows)


def format_latest_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The latest table as compact text columns, ready to print."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    if df["date"].notna().any():
        out["date"] = df["date"].map(lambda d: d.strftime("%Y-%m-%d %H:%M") if pd.notna(d) else "—")
    else:
        out["#"] = df["idx"]
    if "asset" in df.columns:
        out["asset"] = df["asset"]
    out["target"] = df["target"]
    out["signal"] = df["signal"]
    out["conf"] = df["confidence_%"].map(lambda v: fmt_num(v, ".1f") + ("%" if np.isfinite(v) else ""))
    out["unc"] = df["uncertainty_%"].map(lambda v: fmt_num(v, ".2f") + ("%" if np.isfinite(v) else ""))
    out["move"] = df["move_%"].map(lambda v: fmt_num(v, "+.3f") + ("%" if np.isfinite(v) else ""))
    out["pred"] = [fmt_price(p) if k == "continuous" else "—" for p, k in zip(df["pred"], df["kind"], strict=True)]
    if (df["kind"] == "continuous").any():
        out["entry"] = df["entry"].map(fmt_price)
    if df["has_true"].any():
        out["true"] = [tl if (k == "categorical" and isinstance(tl, str)) else fmt_price(t)
                       for t, tl, k in zip(df["true"], df["true_label"], df["kind"], strict=True)]
    if df["ok"].notna().any() or df["has_true"].any():
        out["ok"] = df["ok"].map(lambda v: "⏳" if (v is None or (isinstance(v, float) and np.isnan(v)))
                                 else ("✅" if bool(v) else "❌"))
    return out


def latest_legend(df: pd.DataFrame, say: Say) -> str:
    txt = say.text("latest_legend")
    if df is not None and not df.empty and (df["ok"].notna().any() or df["has_true"].any()):
        txt += say.text("latest_legend_hourglass")
    return txt


def print_latest_frame(df: pd.DataFrame, title: str | None = None, legend: bool = True, indent: str = "  ",
                       say: Say | None = None) -> None:
    say = say or Say()
    if df is None or df.empty:
        print(indent + say.text("latest_none"))
        return
    if title:
        print(title)
    txt = format_latest_frame(df).to_string(index=False)
    print("\n".join(indent + ln for ln in txt.split("\n")))
    if legend:
        print(indent + latest_legend(df, say))
