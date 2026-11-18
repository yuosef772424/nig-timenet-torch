"""The failure reading (PHILOSOPHY section 3) of the batch_by walk-forward, from the saved TEST tables of every fold x
seed: labels joined back from the artifact, the volatility baseline (rule C1), the error map by fold / volatility
tercile / coin, the worst rows, the direction bias of the close head. Prints markdown.

    python docs/audit/2026-11-17-batch-by/failure_reading.py ARTIFACT STUDY [CONTROL]

CONTROL: the size-control arm of A2 (C20 in study 1, C34 in study 2); an arm's per-seed folders ``<arm>_s<seed>``
are merged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.workflow.flat_run import row_coins
from nig_timenet_torch.workflow.run_common import input_key

ARMS = ("A0", "A1", "A2", "C60", "C20", "C34", "H")
HEADS = ("high", "low", "close")


def labels(artifact: str) -> pd.DataFrame:
    """One row per artifact window: coin, timestamp, the three regression labels, the last bar's range and the mean
    range of the last 6 bars (volatility baselines of rule C1, from the INPUT window only)."""
    ds = load_dataset(LoaderSettings(name=artifact, mmap=True, local_dir="")).dataset
    x = np.asarray(ds[input_key(ds)])
    rng = x[:, :, 0] - x[:, :, 1]                      # log(high / prev close) - log(low / prev close)
    idx = np.arange(x.shape[0])
    out = pd.DataFrame({"asset": row_coins(ds, idx).astype(str),
                        "timestamp": pd.to_datetime(np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64),
                                                    utc=True),
                        "range1": rng[:, -1], "range6": rng[:, -6:].mean(axis=1)})
    for h in HEADS:
        out[f"lab_{h}"] = np.asarray(ds[f"y_{h}_reg"])
    return out


def tables(study: Path, lab: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for arm in ARMS:
        dirs = [study / arm, *sorted(study.glob(f"{arm}_s*"))]
        for p in sorted(q for d in dirs for q in d.glob("fold_*_seed_*/test_table.parquet")):
            t = pd.read_parquet(p)
            t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
            t["arm"], t["seed"] = arm, int(p.parent.name.split("_seed_")[1])
            parts.append(t.merge(lab, on=["asset", "timestamp"], how="left", validate="many_to_one"))
    return pd.concat(parts, ignore_index=True)


def ic(df: pd.DataFrame, pred: str, lab: str) -> float:
    return metrics.rank_ic(df[lab].to_numpy(), df[pred].to_numpy(), df["timestamp"].astype("int64").to_numpy())


def main(artifact: str, study: str, control: str = "C20") -> int:
    lab = labels(artifact)
    df = tables(Path(study), lab)
    assert df["lab_high"].notna().all(), "a test row did not join back to the artifact"
    a0 = df[(df.arm == "A0") & (df.seed == 0)]
    print("## Volatility baselines (rule C1), test rows of all folds (seed 0 rows of A0; the same rows in every arm)\n")
    for h in ("high", "low"):
        sign = 1 if h == "high" else -1
        print(f"- {h}: rank IC by timestamp of last-bar range {ic(a0.assign(b=sign * a0.range1), 'b', f'lab_{h}'):.4f},"
              f" of 6-bar mean range {ic(a0.assign(b=sign * a0.range6), 'b', f'lab_{h}'):.4f}")
    print("\n## Rank IC by timestamp per arm (all folds pooled, mean over the seeds) and the gap to the 6-bar range\n")
    print("| arm | high | low | close | low - range6 baseline |\n|---|---|---|---|---|")
    base_low = ic(a0.assign(b=-a0.range6), "b", "lab_low")
    for arm in [a for a in ARMS if a in set(df.arm)]:
        g = df[df.arm == arm]
        v = {h: np.mean([ic(s, f"y_{h}", f"lab_{h}") for _, s in g.groupby("seed")]) for h in HEADS}
        print(f"| {arm} | {v['high']:.4f} | {v['low']:.4f} | {v['close']:.4f} | {v['low'] - base_low:+.4f} |")
    print("\n## Error map: low-head rank IC by volatility tercile (last-bar range) and by fold (seeds pooled)\n")
    df["vol_t"] = pd.qcut(df["range1"].rank(method="first"), 3, labels=["calm", "mid", "volatile"])
    shown = [a for a in ("A0", "A2", control) if a in set(df.arm)]
    print("| group | rows A0 | " + " | ".join(f"{a} low IC" for a in shown) + " |\n|---|---|" + "---|" * len(shown))
    for key, name in (("vol_t", "volatility"), ("fold", "fold")):
        for gv in sorted(df[key].dropna().unique(), key=str):
            cells = []
            for arm in shown:
                s = df[(df.arm == arm) & (df[key] == gv)]
                cells.append(np.mean([ic(q, "y_low", "lab_low") for _, q in s.groupby("seed")]))
            n = int(((df.arm == "A0") & (df[key] == gv)).sum())
            print(f"| {name} {gv} | {n} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
    cand = "A2" if "A2" in set(df.arm) else control
    print(f"\n## Coins where {cand} gains / loses most on the low head (|mu - y| relative to A0, seeds pooled)\n")
    err = {arm: df[df.arm == arm].assign(e=lambda q: (q.y_low - q.lab_low).abs()).groupby("asset").e.mean()
           for arm in ("A0", cand)}
    gap = (err[cand] - err["A0"]).sort_values()
    print(f"| coin | mean abs err A0 | {cand} - A0 |\n|---|---|---|")
    for c in list(gap.index[:3]) + list(gap.index[-3:]):
        print(f"| {c} | {err['A0'][c]:.4f} | {gap[c]:+.4f} |")
    print("\n## The worst rows of A0 (high head, largest |mu - y|, seed 0)\n")
    w = a0.assign(e=(a0.y_high - a0.lab_high).abs()).nlargest(8, "e")
    print("| coin | time | predicted high | realised high | last-bar range |\n|---|---|---|---|---|")
    for r in w.itertuples():
        print(f"| {r.asset} | {r.timestamp:%Y-%m-%d %H:%M} | {r.y_high:.4f} | {r.lab_high:.4f} | {r.range1:.4f} |")
    print("\n## Bias of the close head (share of rows with mu > 0 against the realised share)\n")
    print("| arm | predicted up | realised up | mean mu | mean label |\n|---|---|---|---|---|")
    for arm in [a for a in ARMS if a in set(df.arm)]:
        g = df[df.arm == arm]
        print(f"| {arm} | {(g.y_close > 0).mean():.3f} | {(g.lab_close > 0).mean():.3f} | {g.y_close.mean():+.5f} | "
              f"{g.lab_close.mean():+.5f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
