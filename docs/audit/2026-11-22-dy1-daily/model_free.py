"""DY1 model-free bound (ROADMAP N3 at the daily frame): ridge and trivial rules on the folds of the rolling entry.

Pre-registered in PREREGISTERED.md (section "Model-free first"). Same folds as entry ``rolling`` (expanding, 1095 D,
180 D, step 180 D, embargo 2 D, 7 folds, val = last 15 % of the train window), same universe (the 100 coins with the
most TRAIN rows per fold), and each score goes through the SAME account (``eval/account.py::account_section``).

    python docs/audit/2026-11-22-dy1-daily/model_free.py ARTIFACT OUT_JSON [--books 2] [--tables DIR]
Prints one line per rule; writes the stats (and the per-fold rank IC) to OUT_JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.asset_index import asset_ids
from nig_timenet_torch.eval.account import account_section
from nig_timenet_torch.workflow.rolling import fold_with_val, rolling_folds
from nig_timenet_torch.workflow.run_common import top_coin_rows
from nig_timenet_torch.workflow.settings import RollingSettings

ROLLING = RollingSettings(mode="expanding", train_span="1095D", test_span="180D", step="180D", embargo="2D",
                          max_folds=7)
RIDGE_ALPHA = 1e-3   # x the trace / n of the train Gram matrix (scale-free)
RULES = ("ridge", "momentum_7", "momentum_30", "reversal_1", "reversal_3", "low_vol_30")


def features(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """x (N, 32, 3) daily [log h, log l, log c] moves -> (ridge design, named simple scores)."""
    c = x[:, :, 2]
    rng = (x[:, -30:, 0] - x[:, -30:, 1]).mean(axis=1)
    past = {k: c[:, -k:].sum(axis=1) for k in (1, 3, 7, 14, 30)}
    design = np.concatenate([x.reshape(len(x), -1), np.stack([*past.values(), rng], axis=1)], axis=1)
    simple = {"momentum_7": past[7], "momentum_30": past[30], "reversal_1": -past[1], "reversal_3": -past[3],
              "low_vol_30": -rng}
    return design, simple


def demean_by(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """v minus its mean over the rows of the same timestamp (2-d: per column)."""
    _, inv = np.unique(t, return_inverse=True)
    v2 = v.reshape(len(v), -1)
    sums = np.zeros((inv.max() + 1, v2.shape[1]))
    np.add.at(sums, inv, v2)
    mean = sums / np.bincount(inv)[:, None]
    return (v2 - mean[inv]).reshape(v.shape)


def rank_ic(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"s": score, "y": y, "t": t})
    return df.groupby("t").apply(lambda g: g["s"].rank().corr(g["y"].rank()) if len(g) > 4 else np.nan,
                                 include_groups=False).dropna().to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("out")
    ap.add_argument("--books", type=int, default=2)
    ap.add_argument("--tables", default="", help="also write <dir>/<rule>/fold_<k>/battery_table_{val,test}.parquet")
    a = ap.parse_args()
    ds = npy_dir.load(a.artifact, mmap=False)
    tf = ds["base_tf"]
    x = np.asarray(ds[f"X_{tf}"], np.float64)
    lc = np.asarray(ds["last_candles"])
    t = lc[:, TS_COL].astype(np.int64)
    last_close = lc[:, LAST_COLUMNS.index("last_close")]
    y_ret = lc[:, LAST_COLUMNS.index("future_close")] / last_close - 1.0
    coins = np.array([b["name"] for b in ds["asset_bounds"]])
    design, simple = features(x)
    y_log = np.log1p(y_ret)
    end_time = pd.to_datetime(t + pd.Timedelta(tf.upper()).value)   # the bar has closed: the prediction is made then
    folds = rolling_folds(ds, ROLLING)
    tables = {r: [] for r in RULES}
    ics = {r: [] for r in RULES}
    for fold in folds:
        plan, _ = fold_with_val(fold, ROLLING.embargo, ROLLING.val_share)
        tr, va, te = top_coin_rows(ds, (plan.train_idx, plan.val_idx, plan.test_idx), 100)
        both = np.concatenate([tr, va, te])
        z = np.zeros((len(design), design.shape[1]))
        z[both] = demean_by(design[both], t[both])
        mu_x, sd_x = z[tr].mean(axis=0), z[tr].std(axis=0) + 1e-12
        zz = (z - mu_x) / sd_x
        yd = np.zeros(len(y_log))
        yd[tr] = demean_by(y_log[tr], t[tr])
        g = zz[tr].T @ zz[tr]
        w = np.linalg.solve(g + RIDGE_ALPHA * np.trace(g) / len(g) * np.eye(len(g)), zz[tr].T @ yd[tr])
        scores = {"ridge": zz @ w, **simple}
        for r in RULES:
            s = scores[r]
            mk = lambda idx: pd.DataFrame({"mu": s[idx], "y_ret": y_ret[idx], "coin": coins[asset_ids(ds, idx)],  # noqa: E731
                                           "end_time": end_time[idx], "last_close": last_close[idx]})
            tables[r].append((mk(va), mk(te)))
            if a.tables:
                d = Path(a.tables) / r / f"fold_{fold.index}"
                d.mkdir(parents=True, exist_ok=True)
                tables[r][-1][0].to_parquet(d / "battery_table_val.parquet")
                tables[r][-1][1].to_parquet(d / "battery_table_test.parquet")
            ics[r].append(float(np.mean(rank_ic(s[te], y_ret[te], t[te]))))
        print(f"fold {fold.index}: train {len(tr)} val {len(va)} test {len(te)} rows, test {fold.test_start.date()}"
              f" .. {fold.test_end.date()}, ridge IC {ics['ridge'][-1]:+.4f}", flush=True)
    res = {}
    for r in RULES:
        all_ic = rank_ic(np.concatenate([tb[1]["mu"] for tb in tables[r]]),
                         np.concatenate([tb[1]["y_ret"] for tb in tables[r]]),
                         np.concatenate([tb[1]["end_time"].to_numpy().astype(np.int64) for tb in tables[r]]))
        sec = account_section(tables[r], n_books=a.books, fee_grid=(0.0, 0.0002, 0.0005, 0.001))
        st = sec["stats"]
        test = pd.concat([tb[1].assign(fold=k) for k, tb in enumerate(tables[r])])
        thr = [float(np.quantile(tb[0]["mu"], 0.9)) for tb in tables[r]]
        test["long"] = test["mu"].to_numpy() >= np.repeat(thr, [len(tb[1]) for tb in tables[r]])
        excess = (test["y_ret"] - test.groupby("end_time")["y_ret"].transform("mean"))[test["long"]]
        res[r] = {"ic_folds": ics[r], "ic_mean": float(all_ic.mean()),
                  "ic_t": float(all_ic.mean() / all_ic.std() * np.sqrt(len(all_ic) / 2)),   # /2: overlapping labels
                  "gross_excess_per_trade": float(excess.mean()), "stats": st, "rows": sec["rows"]}
        m, p = st["model"], st["placebo"]
        print(f"{r:<12} IC {res[r]['ic_mean']:+.4f} (t~{res[r]['ic_t']:.1f}) excess/trade "
              f"{res[r]['gross_excess_per_trade'] * 100:+.3f}% | CAGR fee0 {st['model fee 0']['cagr'] * 100:+6.1f}% "
              f"fee.08 {m['cagr'] * 100:+6.1f}% Sharpe {m['sharpe']:.2f} DD {m['max_dd'] * 100:.0f}% | placebo p95 "
              f"{p['cagr_p95'] * 100:+.1f}% | always long {st['always long']['cagr'] * 100:+.1f}% | BTC "
              f"{st.get('buy and hold', {}).get('cagr', float('nan')) * 100:+.1f}%", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
