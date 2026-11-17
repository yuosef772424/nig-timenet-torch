"""PURPOSE: Read a batch-influence record (trainer/batch_influence.py): reliability of the step deltas (split-half,
    detrended), steps that helped or hurt beyond the typical one, modules, and row groups that hurt vs a permutation
    null.
TAGS: batch influence, influence report, split half reliability, spearman brown, excess delta, harmful batches, group
    attribution, permutation null, bonferroni, memorisation, first order check
PITFALLS: early in training every step lowers the loss, so the RAW split-half correlation is inflated by that common
    drift; the detrended one (each half minus its rolling median) is the honest reliability of a batch's own effect; a
    group score is the share-weighted mean EXCESS delta of the batches holding the group, judged against shuffled deltas
    within the epoch (with random batches the shares vary little, so the test is conservative: a non-flagged group is
    "not shown", not "harmless").

Pure numpy / pandas over the monitor's frame (one row per optimizer step) and its (step, row) composition arrays; no
model. ``influence_summary`` -> dict, ``group_attribution`` / ``top_steps`` -> frames, ``render_markdown`` and
``write_influence_report`` write ``batch_influence_report.md`` / ``.json``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.stats import metrics

__all__ = ["excess_delta", "influence_summary", "group_attribution", "top_steps", "render_markdown",
           "write_influence_report", "spearman_brown"]


def spearman_brown(r: float) -> float:
    """Reliability of the whole probe from the correlation r of its two halves: 2 r / (1 + r) (NaN stays NaN)."""
    return float(2.0 * r / (1.0 + r)) if np.isfinite(r) and r > -1.0 else float("nan")


def _trend(v: pd.Series, window: int) -> pd.Series:
    return v.rolling(window, center=True, min_periods=1).median()


def excess_delta(frame: pd.DataFrame, column: str = "delta", window: int = 9) -> pd.Series:
    """The step's delta minus the rolling median of the deltas around it (``window`` steps, centred): the part of a
    batch's effect that is its own and not the drift every step shares at that point of training."""
    return frame[column] - _trend(frame[column], window)


def _split_half(frame: pd.DataFrame, pre: str, window: int) -> dict:
    a, b = frame[f"{pre}delta_a"], frame[f"{pre}delta_b"]
    raw = metrics.pearson(a.to_numpy(), b.to_numpy())
    det = metrics.pearson((a - _trend(a, window)).to_numpy(), (b - _trend(b, window)).to_numpy())
    return {"raw_r": raw, "detrended_r": det, "reliability": spearman_brown(det)}


def influence_summary(frame: pd.DataFrame, *, window: int = 9) -> dict:
    """Headline numbers of the primary probe: total change (= sum of the step deltas), harmful share (raw delta > 0
    and excess > 0), split-half reliability, the first-order check, per epoch, per module, and the other probes."""
    if frame.empty:
        raise ValueError("the influence frame is empty (no optimizer step was recorded)")
    exc = excess_delta(frame, window=window)
    out = {"n_steps": int(len(frame)), "probe_start": float(frame["probe_before"].iloc[0]),
           "total_delta": float(frame["delta"].sum()), "harmful_share": float((frame["delta"] > 0).mean()),
           "excess_sd": float(exc.std()), "split_half": _split_half(frame, "", window)}
    fo = frame.dropna(subset=["fo_delta"]) if "fo_delta" in frame else frame.iloc[:0]
    if len(fo):
        out["first_order"] = {"n_steps": int(len(fo)), "r_vs_exact": metrics.pearson(fo["fo_delta"].to_numpy(),
                                                                                     fo["delta"].to_numpy()),
                              "sum_ratio": float(fo["fo_delta"].sum() / fo["delta"].sum())
                              if fo["delta"].sum() != 0 else float("nan"),
                              "mean_grad_alignment": float(fo["grad_alignment"].mean())}
        out["modules"] = {c[3:]: float(fo[c].sum()) for c in fo.columns
                          if c.startswith("fo_") and c != "fo_delta"}
    per_epoch = frame.assign(excess=exc).groupby("epoch")
    out["per_epoch"] = [{"epoch": int(e), "steps": int(len(g)), "delta": float(g["delta"].sum()),
                         "harmful_share": float((g["delta"] > 0).mean()),
                         "train_loss": float(g["train_loss"].mean())} for e, g in per_epoch]
    out["heads"] = {c[6:]: float(frame[c].sum()) for c in frame.columns
                    if c.startswith("delta_") and c not in ("delta_a", "delta_b")}
    others = sorted({c[:-len("_delta")] for c in frame.columns if c.endswith("_delta") and c != "fo_delta"
                     and not c.startswith("delta")})
    out["other_probes"] = {}
    for p in others:
        d = frame[f"{p}_delta"]
        out["other_probes"][p] = {
            "total_delta": float(d.sum()), "split_half": _split_half(frame, f"{p}_", window),
            "r_with_primary": metrics.pearson(excess_delta(frame, f"{p}_delta", window).to_numpy(), exc.to_numpy()),
            # memorisation: the step helped this probe (e.g. training rows) and hurt the primary (validation)
            "helps_it_hurts_primary": float(((d < 0) & (frame["delta"] > 0)).mean())}
    return out


def _shares(steps: np.ndarray, groups: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    names, inv = np.unique(groups, return_inverse=True)
    counts = np.zeros((n_steps, len(names)))
    np.add.at(counts, (steps, inv), 1.0)
    tot = counts.sum(axis=1, keepdims=True)
    return names, np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)


def group_attribution(frame: pd.DataFrame, steps: np.ndarray, rows: np.ndarray, row_group: np.ndarray, *,
                      window: int = 9, n_perm: int = 1000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Per group (``row_group[row]``: a coin, a month, a regime of each TRAINING row): the share-weighted mean excess
    delta of the batches that held it, its z against ``n_perm`` shuffles of the excess deltas within each epoch, the
    two-sided p (normal tail of z; ``p_perm`` = the plain permutation p, floored at 1 / (n_perm + 1)) and ``flag`` =
    harmful / helpful when p < alpha / n_groups (Bonferroni), else ""."""
    exc = excess_delta(frame, window=window).to_numpy()
    names, share = _shares(np.asarray(steps), np.asarray(row_group)[np.asarray(rows)], len(frame))
    weight = share.sum(axis=0)
    score = share.T @ exc / np.where(weight > 0, weight, np.nan)
    rng = np.random.default_rng(seed)
    epochs = frame["epoch"].to_numpy()
    blocks = [np.flatnonzero(epochs == e) for e in np.unique(epochs)]
    null = np.empty((n_perm, len(names)))
    for i in range(n_perm):
        perm = exc.copy()
        for b in blocks:
            perm[b] = perm[rng.permutation(b)]
        null[i] = share.T @ perm / np.where(weight > 0, weight, np.nan)
    sd, centre = null.std(axis=0), null.mean(axis=0)
    p_perm = (1.0 + (np.abs(null - centre) >= np.abs(score - centre)).sum(axis=0)) / (n_perm + 1)
    z = np.divide(score - centre, sd, out=np.full_like(sd, np.nan), where=sd > 0)
    # the null of a weighted mean over many steps is close to normal: its tail gives p below 1 / (n_perm + 1), which
    # a Bonferroni cut over hundreds of coins needs; the permutation p is kept as the check of that approximation
    p = np.array([math.erfc(abs(v) / math.sqrt(2.0)) if np.isfinite(v) else 1.0 for v in z])
    cut = alpha / max(len(names), 1)
    flag = np.where(p < cut, np.where(score > centre, "harmful", "helpful"), "")
    return pd.DataFrame({"group": names, "rows_seen": (share > 0).sum(axis=0), "weight": weight, "score": score,
                         "z": z, "p": p, "p_perm": p_perm, "flag": flag}
                        ).sort_values("z", ascending=False, ignore_index=True)


def top_steps(frame: pd.DataFrame, steps: np.ndarray, rows: np.ndarray, row_group: np.ndarray | None = None, *,
              k: int = 10, window: int = 9) -> pd.DataFrame:
    """The ``k`` most harmful and ``k`` most helpful steps by excess delta, with their batch composition (number of
    groups and the three most frequent ones) when ``row_group`` is given."""
    f = frame.assign(excess=excess_delta(frame, window=window))
    pick = pd.concat([f.nlargest(k, "excess").assign(kind="harmful"), f.nsmallest(k, "excess").assign(kind="helpful")])
    cols = ["kind", "epoch", "step", "n_rows", "train_loss", "delta", "excess"] + \
        [c for c in ("grad_alignment",) if c in f]
    out = pick[cols].reset_index(drop=True)
    if row_group is not None and len(steps):
        groups = np.asarray(row_group)[np.asarray(rows)]
        comp = []
        for s in out["step"]:
            vc = pd.Series(groups[np.asarray(steps) == s]).value_counts()
            comp.append((int(len(vc)), ", ".join(f"{g} ({n})" for g, n in vc.head(3).items())))
        out["n_groups"], out["top_groups"] = zip(*comp, strict=True) if comp else ((), ())
    return out


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _table(df: pd.DataFrame) -> list[str]:
    """A markdown table without the optional ``tabulate`` dependency."""
    head = [f"| {' | '.join(map(str, df.columns))} |", "|" + "---|" * len(df.columns)]
    return head + [f"| {' | '.join(_fmt(v) for v in row)} |" for row in df.itertuples(index=False)]


def render_markdown(summary: dict, groups: pd.DataFrame | None = None, top: pd.DataFrame | None = None,
                    title: str = "Batch influence") -> str:
    """The report as markdown: headline, reliability, first-order check, epochs, modules, groups, top steps."""
    sh = summary["split_half"]
    lines = [f"# {title}", "",
             f"- steps: {summary['n_steps']}; primary probe loss at start {_fmt(summary['probe_start'])}, "
             f"total change {_fmt(summary['total_delta'])} (= sum of the step deltas)",
             f"- harmful steps (raw delta > 0): {summary['harmful_share']:.1%}; sd of the excess delta "
             f"{_fmt(summary['excess_sd'])}",
             f"- split-half r: raw {_fmt(sh['raw_r'])}, detrended {_fmt(sh['detrended_r'])} -> reliability of a "
             f"step's own effect {_fmt(sh['reliability'])} (Spearman-Brown; < 0.5 = mostly probe noise: raise "
             f"probe_rows)"]
    if "first_order" in summary:
        fo = summary["first_order"]
        lines.append(f"- first-order g.dtheta vs exact on {fo['n_steps']} steps: r {_fmt(fo['r_vs_exact'])}, sum "
                     f"ratio {_fmt(fo['sum_ratio'])}, mean cos(probe grad, batch grad) "
                     f"{_fmt(fo['mean_grad_alignment'])}")
    lines += ["", "## Per epoch", "", "| epoch | steps | delta | harmful share | train loss |", "|---|---|---|---|---|"]
    lines += [f"| {r['epoch']} | {r['steps']} | {_fmt(r['delta'])} | {r['harmful_share']:.1%} | "
              f"{_fmt(r['train_loss'])} |" for r in summary["per_epoch"]]
    lines += ["", "## Per head (sum of deltas)", ""] + [f"- {h}: {_fmt(v)}" for h, v in summary["heads"].items()]
    if summary.get("modules"):
        lines += ["", "## Per module (first-order, sum of g.dtheta; < 0 = moved toward the probe)", ""]
        lines += [f"- {m}: {_fmt(v)}" for m, v in sorted(summary["modules"].items(), key=lambda kv: kv[1])]
    for p, o in summary.get("other_probes", {}).items():
        lines += ["", f"## Probe `{p}`", "",
                  f"- total change {_fmt(o['total_delta'])}; detrended split-half r "
                  f"{_fmt(o['split_half']['detrended_r'])}; r of excess deltas with the primary "
                  f"{_fmt(o['r_with_primary'])}; steps that helped it and hurt the primary "
                  f"{o['helps_it_hurts_primary']:.1%}"]
    if groups is not None and len(groups):
        flagged = groups[groups["flag"] != ""]
        lines += ["", f"## Groups ({len(groups)}; flagged at Bonferroni: {len(flagged)})", "",
                  "| group | weight | score | z | p | flag |", "|---|---|---|---|---|---|"]
        show = pd.concat([groups.head(10), groups.tail(5)]).drop_duplicates("group")
        lines += [f"| {r.group} | {r.weight:.3g} | {r.score:.3g} | {r.z:.2f} | {r.p:.3g} | {r.flag} |"
                  for r in show.itertuples()]
    if top is not None and len(top):
        lines += ["", "## Most harmful / helpful steps (excess delta)", "", *_table(top)]
    return "\n".join(lines) + "\n"


def write_influence_report(out, frame: pd.DataFrame, steps: np.ndarray, rows: np.ndarray,
                           row_group: np.ndarray | None = None, *, window: int = 9, n_perm: int = 1000,
                           title: str = "Batch influence") -> dict:
    """Write ``batch_influence_report.md`` / ``.json`` (summary, groups, top steps) under ``out``; returns the
    summary with the groups and top steps as records."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = influence_summary(frame, window=window)
    groups = (group_attribution(frame, steps, rows, row_group, window=window, n_perm=n_perm)
              if row_group is not None and len(steps) else None)
    top = top_steps(frame, steps, rows, row_group, window=window)
    (out / "batch_influence_report.md").write_text(render_markdown(summary, groups, top, title))
    res = {**summary, "groups": None if groups is None else groups.to_dict("records"),
           "top_steps": top.to_dict("records")}
    (out / "batch_influence_report.json").write_text(json.dumps(res, indent=1, default=str))
    return res
