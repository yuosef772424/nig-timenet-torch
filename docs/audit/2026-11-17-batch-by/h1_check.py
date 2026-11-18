"""H1 of PREREGISTERED_2.md: A2 (timestamp batches) vs A0 (rows 256) and vs C60 (rows 60) on the low head's rank IC by
timestamp, per fold x seed; rule: A2 higher in >= 9 of 12 pairs AND the 95 % t-interval (3 df) of the per-fold mean
paired difference above 0, for BOTH comparisons. The high head and the test raw loss are reported, not judged.

    python docs/audit/2026-11-17-batch-by/h1_check.py STUDY
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T975_3DF = 3.182446
METRICS = {"low IC": "targets.low.skill.rank_ic_by_timestamp", "high IC": "targets.high.skill.rank_ic_by_timestamp",
           "close IC": "targets.close.skill.rank_ic_by_timestamp", "test raw loss": "test_raw_loss",
           "low cov": "targets.low.sharp.coverage", "high cov": "targets.high.sharp.coverage"}


def load(study: Path, arm: str) -> pd.DataFrame:
    loss = pd.DataFrame([json.loads(x) for x in (study / arm / "test_loss.jsonl").read_text().splitlines() if x])
    return loss.merge(pd.read_csv(study / arm / "rolling_folds.csv"), on=["fold", "seed"]).set_index(["fold", "seed"])


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, higher_is_better: bool = True) -> dict:
    common = a.index.intersection(b.index)
    d = (a.loc[common, col] - b.loc[common, col]) * (1 if higher_is_better else -1)
    per_fold = d.groupby(level="fold").mean()
    half = T975_3DF * per_fold.std(ddof=1) / np.sqrt(len(per_fold))
    return {"pairs": len(d), "better": int((d > 0).sum()), "mean": float(per_fold.mean()),
            "lo": float(per_fold.mean() - half), "hi": float(per_fold.mean() + half),
            "per_fold": [round(float(v), 4) for v in per_fold]}


def main(study) -> int:
    study = Path(study)
    arms = {a: load(study, a) for a in ("A0", "A2", "C60")}
    print("## Per arm (mean over folds and seeds; sd over the fold x seed runs)\n")
    print("| arm | runs | " + " | ".join(METRICS) + " | minutes |\n|---|---|" + "---|" * len(METRICS) + "---|")
    for a, f in arms.items():
        cells = [f"{f[c].mean():.4f} ({f[c].std():.4f})" for c in METRICS.values()]
        print(f"| {a} | {len(f)} | " + " | ".join(cells) + f" | {f['seconds'].max() / 60:.1f} |")
    print("\n## H1 comparisons (A2 better in how many pairs; mean per-fold difference in the 'better' direction;"
          " 95 % t-interval of the per-fold mean, 3 df)\n")
    print("| comparison | metric | better | mean diff | 95 % CI | per fold | passes |\n|---|---|---|---|---|---|---|")
    verdict = True
    for ref in ("A0", "C60"):
        for name, col in METRICS.items():
            if "cov" in name:
                continue
            r = paired(arms["A2"], arms[ref], col, higher_is_better=name != "test raw loss")
            ok = r["better"] >= 9 and r["lo"] > 0
            judged = name == "low IC"
            verdict &= ok if judged else True
            print(f"| A2 vs {ref} | {name}{'' if judged else ' (read)'} | {r['better']} / {r['pairs']} | "
                  f"{r['mean']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['per_fold']} | {ok} |")
    print(f"\n**H1 {'supported' if verdict else 'refuted'}** (pre-registered: low IC, both comparisons).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
