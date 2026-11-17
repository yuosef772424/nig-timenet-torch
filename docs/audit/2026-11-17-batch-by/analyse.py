"""The pre-registered reading of the batch_by walk-forward (PREREGISTERED.md, accept rule (a) + (b)), from the runner's
outputs: ``<study>/<arm>/test_loss.jsonl`` and ``<study>/<arm>/rolling_folds.csv``. Prints markdown tables.

    python docs/audit/2026-11-17-batch-by/analyse.py STUDY
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("A0", "A1", "A2", "C60", "C20")
COMPARISONS = (("A1", "A0"), ("A2", "A0"), ("A1", "C60"), ("A2", "C20"), ("C60", "A0"), ("C20", "A0"))
T975_3DF = 3.182446        # two-sided 95 % t quantile, 3 degrees of freedom (4 folds)
IC = "targets.{h}.skill.rank_ic_by_timestamp"


def load(study: Path) -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        p = study / arm / "test_loss.jsonl"
        if not p.exists():
            continue
        loss = pd.DataFrame([json.loads(line) for line in p.read_text().splitlines() if line.strip()])
        folds = pd.read_csv(study / arm / "rolling_folds.csv")
        frames.append(loss.merge(folds, on=["fold", "seed"], how="left"))
    return pd.concat(frames, ignore_index=True)


def compare(df: pd.DataFrame, x: str, y: str) -> dict:
    a = df[df.arm == x].set_index(["fold", "seed"]).sort_index()
    b = df[df.arm == y].set_index(["fold", "seed"]).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    diff = a["test_raw_loss"] - b["test_raw_loss"]
    per_fold = diff.groupby(level="fold").mean()
    mean, n = float(per_fold.mean()), len(per_fold)
    half = T975_3DF * float(per_fold.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    rule_a = bool((diff < 0).sum() >= 7 and mean + half < 0) if len(diff) == 8 else False
    rule_b, ic = True, {}
    for h in ("high", "low"):
        col = IC.format(h=h)
        ic[h] = (float(a[col].mean()), float(b[col].mean()))
        spread = float(b[col].groupby(level="fold").agg(lambda s: abs(s.iloc[0] - s.iloc[-1])).mean())
        rule_b &= ic[h][0] >= ic[h][1] - 2 * spread
    return {"x": x, "y": y, "pairs": len(diff), "x_wins": int((diff < 0).sum()), "mean_diff": mean,
            "ci": (mean - half, mean + half), "rule_a": rule_a, "rule_b": rule_b, "better": rule_a and rule_b,
            "ic": ic, "per_fold": per_fold.round(5).to_dict()}


def main(study) -> int:
    study = Path(study)
    df = load(study)
    heads = [c for c in df.columns if c.startswith("test_") and c.endswith("_loss") and c not in
             ("test_raw_loss", "test_total_loss")]
    print("## Per arm (mean over folds and seeds; sd over the 8 fold x seed runs)\n")
    cols = ["test_raw_loss"] + [IC.format(h=h) for h in ("close", "high", "low")] + \
        [f"targets.{h}.sharp.coverage" for h in ("close", "high", "low")]
    print("| arm | runs | " + " | ".join(c.replace("targets.", "").replace(".skill.rank_ic_by_timestamp", " IC")
                                         .replace(".sharp.coverage", " cov") for c in cols) + " | minutes |")
    print("|---|---|" + "---|" * len(cols) + "---|")
    for arm, g in df.groupby("arm", sort=False):
        cells = [f"{g[c].mean():.4f} ({g[c].std():.4f})" for c in cols]
        print(f"| {arm} | {len(g)} | " + " | ".join(cells) + f" | {g['seconds'].max() / 60:.1f} |")
    print("\n## Per head test loss (mean over folds and seeds)\n")
    print("| arm | " + " | ".join(h[5:-5] for h in heads) + " |\n|---|" + "---|" * len(heads))
    for arm, g in df.groupby("arm", sort=False):
        print(f"| {arm} | " + " | ".join(f"{g[h].mean():.4f}" for h in heads) + " |")
    print("\n## Per fold test raw loss (mean over seeds)\n")
    pf = df.pivot_table(index="fold", columns="arm", values="test_raw_loss", aggfunc="mean")
    print("| fold | " + " | ".join(pf.columns) + " |\n|---|" + "---|" * len(pf.columns))
    for f, r in pf.iterrows():
        print(f"| {f} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
    print("\n## Pre-registered comparisons (X better than Y: rule (a) AND rule (b))\n")
    print("| X vs Y | pairs | X lower | mean diff (X - Y) | 95 % CI over folds | rule a | high IC X / Y | "
          "low IC X / Y | rule b | better |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    res = [compare(df, x, y) for x, y in COMPARISONS if {x, y} <= set(df.arm)]
    for r in res:
        print(f"| {r['x']} vs {r['y']} | {r['pairs']} | {r['x_wins']} | {r['mean_diff']:+.5f} | "
              f"[{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}] | {r['rule_a']} | {r['ic']['high'][0]:.4f} / "
              f"{r['ic']['high'][1]:.4f} | {r['ic']['low'][0]:.4f} / {r['ic']['low'][1]:.4f} | {r['rule_b']} | "
              f"**{r['better']}** |")
    (study / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
