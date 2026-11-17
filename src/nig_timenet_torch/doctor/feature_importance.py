"""PURPOSE: Which input features (and which bars of the window) a trained model uses: grouped permutation importance on
    held-out rows, the change of the trainer's loss per head and of the rank IC, time-block CIs, Bonferroni and a
    split-half check.
TAGS: feature importance, permutation importance, grouped permutation, correlated features, clusters, within time,
    cross section, time blocks, block bootstrap, bonferroni, split half, model doctor, attribution
PITFALLS: run it on rows the model was NOT trained on (validation), or it measures memorisation; a feature that is
    highly correlated with another looks unimportant alone (the twin still carries it): read the ``cluster`` rows,
    which permute the correlated set together; ``within_time`` permutes only across coins of the same timestamp, so a
    feature that is the same for every coin at a time (market-wide) scores 0 there by construction; importance is
    "what this model loses", not "what the data hold" (a feature the model ignores scores 0 even if it is
    predictive); a feature the model relies on that has NO link to the label on these rows also scores ~0 (the
    permuted values are as useless as the real ones), only a REVERSED link scores below 0 (``hurts``).

Method (chosen 2026-11-17, owner task "the effect of each feature"; candidates compared in
docs/INFLUENCE.md): for each group of input channels (one feature, a cluster of correlated
features, or every feature over one block of bars) the same random row permutation replaces those channels over the
whole window (the within-window time pattern of the channel is kept, only its link to the label and to the other
channels is broken), the model predicts again, and the trainer's raw loss (and each head's loss, and each regression
head's rank IC) is compared with the unpermuted loss on the SAME rows. The rows are cut into time blocks (whole
timestamps, ``n_blocks`` equal parts of the time range); the per-block loss difference, averaged over ``repeats``
permutations, gives the mean and a block-bootstrap CI at level 1 - alpha / n_groups (rows of one block move together,
so blocks, not rows, are the units). A group is ``uses`` when the CI is above 0, ``hurts`` when below (the model does
better without it: a candidate to drop), else ``none``. The split-half check: the Spearman of the importances computed
on even blocks against odd blocks (a ranking that does not repeat across halves is noise).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.stats import metrics
from nig_timenet_torch.trainer.groups import group_chunks, row_groups

__all__ = ["ImportanceSettings", "PERMUTERS", "feature_importance", "correlation_clusters", "time_blocks",
           "render_importance", "write_importance"]


def _global_permutation(n: int, times: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def _within_time_permutation(n: int, times: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Row k receives the channels of a random row with the SAME timestamp (rows of one time shuffled among
    themselves)."""
    if times is None:
        raise ValueError("the within_time permutation needs the row times")
    base = np.lexsort((np.arange(n), times))
    mixed = np.lexsort((rng.random(n), times))
    perm = np.empty(n, dtype=np.int64)
    perm[base] = mixed
    return perm


#: how the replacing rows are drawn: across all rows, or only across rows of the same timestamp (coin-specific use)
PERMUTERS = {"global": _global_permutation, "within_time": _within_time_permutation}


@dataclass(frozen=True)
class ImportanceSettings:
    """``repeats``: permutations per group (their mean is reported); ``permutation``: a PERMUTERS name; ``n_blocks``:
    time blocks for the CI (>= 4); ``bar_blocks``: also the importance of each of that many blocks of bars (all
    features at once; 0 = off); ``cluster_threshold``: |Spearman| of the last-bar values above which features are
    also permuted together as a cluster (1 = no clusters); ``alpha``: family error rate (Bonferroni over groups);
    ``chunk``: rows per forward pass; ``seed``: the permutations' numpy seed."""
    repeats: int = 5
    permutation: str = "global"
    n_blocks: int = 40
    bar_blocks: int = 0
    cluster_threshold: float = 0.9
    alpha: float = 0.05
    chunk: int = 4096
    seed: int = 0

    def __post_init__(self):
        if self.permutation not in PERMUTERS:
            raise ValueError(f"permutation must be one of {sorted(PERMUTERS)}, not {self.permutation!r}")
        if self.repeats < 1 or self.n_blocks < 4 or self.bar_blocks < 0 or not 0 < self.alpha < 1:
            raise ValueError("repeats >= 1, n_blocks >= 4, bar_blocks >= 0 and 0 < alpha < 1 are required")


def time_blocks(times: np.ndarray, n_blocks: int) -> np.ndarray:
    """Block id per row: the rank of the row's timestamp among the distinct times cut into ``n_blocks`` equal parts
    (rows of one timestamp share a block)."""
    uniq, inv = np.unique(np.asarray(times), return_inverse=True)
    return (inv * min(n_blocks, len(uniq)) // len(uniq)).astype(np.int64)


def correlation_clusters(x: np.ndarray, threshold: float, max_rows: int = 20000) -> list[list[int]]:
    """Clusters (>= 2 members) of input channels whose last-bar values have |Spearman| >= ``threshold`` (linked
    transitively), on at most ``max_rows`` evenly strided rows."""
    x = np.asarray(x)
    step = max(1, len(x) // max_rows)
    rho = np.abs(metrics.spearman_matrix(x[::step, -1, :]))
    parent = list(range(rho.shape[0]))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(np.nan_to_num(rho) >= threshold, k=1)), strict=True):
        parent[root(i)] = root(j)
    members: dict[int, list[int]] = {}
    for i in range(rho.shape[0]):
        members.setdefault(root(i), []).append(i)
    return [m for m in members.values() if len(m) > 1]


def _groups(names: list[str], x, s: ImportanceSettings, extra: dict | None) -> list[dict]:
    t = x.shape[1]
    out = [{"group": n, "kind": "feature", "channels": [j], "bars": slice(0, t)} for j, n in enumerate(names)]
    if s.cluster_threshold < 1:
        for c in correlation_clusters(x[..., :len(names)], s.cluster_threshold):
            out.append({"group": "+".join(names[j] for j in c), "kind": "cluster", "channels": c,
                        "bars": slice(0, t)})
    for b in range(s.bar_blocks):
        lo, hi = b * t // s.bar_blocks, (b + 1) * t // s.bar_blocks
        out.append({"group": f"bars[{lo - t}:{hi - t or ''}]", "kind": "bars", "channels": list(range(len(names))),
                    "bars": slice(lo, hi)})
    for name, ch in (extra or {}).items():
        out.append({"group": name, "kind": "custom", "channels": list(ch), "bars": slice(0, t)})
    return out


class _Scorer:
    """The model's per-block losses (raw and per head) and the rank IC of the regression heads, for x with an optional
    channel replacement."""

    def __init__(self, trainer, x: torch.Tensor, y: dict, blocks: np.ndarray, chunk: int):
        self.trainer, self.x, self.blocks = trainer, x, blocks
        self.dev = trainer._device()
        self.y = {k: v.to(self.dev) for k, v in y.items()}
        self.grouped = trainer.group_channel
        self.cuts = (group_chunks(row_groups(x), chunk) if self.grouped
                     else [slice(i, i + chunk) for i in range(0, len(x), chunk)])
        self.block_rows = [torch.as_tensor(np.flatnonzero(blocks == b), device=self.dev) for b in np.unique(blocks)]
        cfg = trainer.targets_cfg
        self.ic_heads = {t: (c["output_keys"]["mu"] if "mu" in c["output_keys"] else c["output_keys"]["pred"],
                             c["true_key"]) for t, c in cfg.items() if {"mu", "pred"} & set(c["output_keys"])}

    @torch.no_grad()
    def __call__(self, perm: np.ndarray | None = None, channels=None, bars: slice | None = None) -> dict:
        model = self.trainer.model
        model.eval()
        parts = []
        for c in self.cuts:
            xb = self.x[c].clone()
            if perm is not None:
                ch = torch.as_tensor(channels)
                xb[:, bars, ch] = self.x[torch.as_tensor(perm[c])][:, bars, ch]
            parts.append(model(xb.to(self.dev)))
        out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        raw, per = [], []
        for rows in self.block_rows:
            yb = {k: v[rows] for k, v in self.y.items()}
            ob = {k: v[rows] for k, v in out.items()}
            gid = row_groups(self.x[rows.cpu()]).to(self.dev) if self.grouped else None
            _, r, pt, _, _ = self.trainer.compute_losses(None, yb, outputs=ob, groups=gid)
            raw.append(float(r))
            per.append({t: float(v) for t, v in pt.items()})
        ic = {t: metrics.spearman(out[k].float().cpu().numpy().reshape(-1), self.y[lk].cpu().numpy().reshape(-1))
              for t, (k, lk) in self.ic_heads.items()}
        return {"raw": np.array(raw), "per": pd.DataFrame(per), "ic": ic}


def feature_importance(trainer, x: torch.Tensor, y: dict, times: np.ndarray, names: list[str] | None = None,
                       settings: ImportanceSettings | None = None, groups: dict | None = None) -> tuple[pd.DataFrame,
                                                                                                       dict]:
    """(table, summary) of the permutation importance of every group of input channels of the trained ``trainer``
    on the rows (x, y) (validation rows), ``times`` = each row's timestamp. ``names``: one per input channel (the
    group channel of a cross-section run excluded); ``groups``: extra named channel sets (e.g. feature families).
    Table columns: group, kind, n_channels, delta_loss (mean over blocks and repeats; > 0 = the model uses it), lo,
    hi (block-bootstrap CI at 1 - alpha / n_groups), verdict (uses / hurts / none), delta_<head> (per-head loss),
    ic_<head> (change of the rank IC; < 0 = the head's ranking used it), repeat_sd, rank."""
    s = settings or ImportanceSettings()
    n_ch = x.shape[2] - int(trainer.group_channel)
    names = list(names) if names is not None else [f"f{j}" for j in range(n_ch)]
    if len(names) != n_ch:
        raise ValueError(f"{len(names)} names for {n_ch} input channels")
    times = np.asarray(times)
    blocks = time_blocks(times, s.n_blocks)
    score = _Scorer(trainer, x, y, blocks, s.chunk)
    base = score()
    todo = _groups(names, x, s, groups)
    level = 1.0 - s.alpha / len(todo)
    rng = np.random.default_rng(s.seed)
    permute = PERMUTERS[s.permutation]
    rows, halves = [], []
    for g in todo:
        diffs, heads, ics = [], [], []
        for _ in range(s.repeats):
            res = score(permute(len(x), times, rng), g["channels"], g["bars"])
            diffs.append(res["raw"] - base["raw"])
            heads.append((res["per"] - base["per"]).mean())
            ics.append({t: res["ic"][t] - base["ic"][t] for t in base["ic"]})
        d = np.mean(diffs, axis=0)
        mean, lo, hi = metrics.block_bootstrap_ci(d, np.arange(len(d)), level=level, n_boot=2000, seed=s.seed)
        row = {"group": g["group"], "kind": g["kind"], "n_channels": len(g["channels"]), "delta_loss": mean,
               "lo": lo, "hi": hi, "verdict": "uses" if lo > 0 else "hurts" if hi < 0 else "none",
               "repeat_sd": float(np.std([x_.mean() for x_ in diffs]))}
        row.update({f"delta_{t}": float(v) for t, v in pd.DataFrame(heads).mean().items()})
        row.update({f"ic_{t}": float(np.mean([i[t] for i in ics])) for t in base["ic"]})
        rows.append(row)
        halves.append((d[0::2].mean(), d[1::2].mean()))
    table = pd.DataFrame(rows).sort_values("delta_loss", ascending=False, ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1)
    h = np.array(halves)
    summary = {"rows": int(len(x)), "blocks": int(len(np.unique(blocks))), "groups": len(todo),
               "permutation": s.permutation, "repeats": s.repeats, "ci_level": level,
               "base_loss": float(base["raw"].mean()), "base_ic": base["ic"],
               "uses": int((table["verdict"] == "uses").sum()), "hurts": int((table["verdict"] == "hurts").sum()),
               "split_half_rho": metrics.spearman(h[:, 0], h[:, 1]) if len(h) > 2 else float("nan")}
    return table, summary


def render_importance(table: pd.DataFrame, summary: dict, title: str = "Feature importance", top: int = 40) -> str:
    """Markdown: the summary lines and the table (the ``top`` rows by delta_loss and every ``hurts`` row)."""
    lines = [f"# {title}", "",
             f"- rows {summary['rows']}, time blocks {summary['blocks']}, groups {summary['groups']}, permutation "
             f"`{summary['permutation']}` x {summary['repeats']}; base loss {summary['base_loss']:.5g}",
             f"- uses: {summary['uses']}, hurts: {summary['hurts']} (block-bootstrap CI at "
             f"{summary['ci_level']:.4f}, Bonferroni over groups)",
             f"- split-half Spearman of the importances (even vs odd blocks): {summary['split_half_rho']:.3f} "
             f"(< 0.5 = the ranking is mostly noise: more rows or repeats)",
             "- base rank IC per head: " + ", ".join(f"{k} {v:.4f}" for k, v in summary["base_ic"].items()), ""]
    show = pd.concat([table.head(top), table[table["verdict"] == "hurts"]]).drop_duplicates("group")
    cols = ["rank", "group", "kind", "delta_loss", "lo", "hi", "verdict"] + \
        [c for c in table.columns if c.startswith("ic_")]
    lines += [f"| {' | '.join(cols)} |", "|" + "---|" * len(cols)]
    for r in show[cols].itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def write_importance(out, table: pd.DataFrame, summary: dict, name: str = "feature_importance",
                     title: str = "Feature importance") -> None:
    """``<name>.md`` / ``<name>.csv`` / ``<name>.json`` (summary) under ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(render_importance(table, summary, title))
    table.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float))
