"""PURPOSE: The 'deep side' hypothesis: is the side where price goes deeper over the horizon (log high + log low vs
    the last close) ranked by mu_high + mu_low beyond the close head, last-bar and volatility baselines, on VAL and
    TEST with day-block CIs?
TAGS: deep side, derived target, direction hypothesis, rank ic, auc, baselines, partial rank ic, volatility buckets,
    near ties, label clip, day block bootstrap
PITFALLS: deep is a SUM of two labels clipped at |log| = 1 (B-054), so clipped rows are counted; baseline signs
    and every threshold (volatility edges, near-tie margin) come from VAL and are applied unchanged to TEST; the
    direction of the high / low heads ALONE stays suspended (rule D), only their sum is read here, as a hypothesis.

Input frame (one row per window, a ``workflow.reading`` battery table plus ``last_bar_deep``): ``fut_high``,
``fut_low``, ``pred_high``, ``pred_low`` (realised / predicted extreme as a return vs the last close, decoded by the
artifact's codec), ``mu`` (close head return), ``p_dir``, ``y_ret``, ``vol`` (ATR-14 of the window), ``last_ret``,
``last_bar_deep`` (log(H / C) + log(L / C) of the last input bar), ``end_time``. ``deep = log1p(fut_high) +
log1p(fut_low)``; ``deep_pred = log1p(pred_high) + log1p(pred_low)`` (= mu_high + mu_low for the logret_close codec).
Rank IC is per timestamp (cross-section), its CI resamples days; ``evidence`` needs, on BOTH splits, the model's IC
CI above 0, its AUC CI above 0.5 and the paired (model - baseline) IC CI above 0 for EVERY baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nig_timenet_torch.stats import metrics

__all__ = ["deep_frame", "score_columns", "orient", "skill", "close_link", "vol_buckets", "near_ties", "clip_counts",
           "deep_side", "render", "BASELINES", "N_VOL_BUCKETS", "TIE_SHARE", "CLIP_LEVEL"]

#: baseline name -> frame column (oriented by the sign of its VAL IC); "random" / "shuffled model" are added per split
BASELINES = {"last-bar asymmetry": "last_bar_deep", "close head mu": "close_log", "close head p_dir": "p_dir",
             "last-bar return": "last_ret", "volatility (ATR-14)": "vol"}
N_VOL_BUCKETS = 5
TIE_SHARE = 0.2
CLIP_LEVEL = 1.0
_CLIP_TOL = 1e-6


def deep_frame(battery: pd.DataFrame, last_bar_deep: np.ndarray) -> pd.DataFrame:
    """The battery table plus ``deep``, ``deep_pred``, ``close_log`` (log1p of the close mu), ``ts`` (end time),
    ``day`` and ``last_bar_deep``."""
    d = battery.copy()
    d["deep"] = np.log1p(d["fut_high"].to_numpy(np.float64)) + np.log1p(d["fut_low"].to_numpy(np.float64))
    d["deep_pred"] = np.log1p(d["pred_high"].to_numpy(np.float64)) + np.log1p(d["pred_low"].to_numpy(np.float64))
    d["close_log"] = np.log1p(d["mu"].to_numpy(np.float64))
    d["last_bar_deep"] = np.asarray(last_bar_deep, dtype=np.float64)
    d["ts"] = d["end_time"].values
    d["day"] = d["end_time"].values.astype("datetime64[D]")
    return d


def score_columns(d: pd.DataFrame, signs: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Score name -> array: the model, every oriented baseline, a random score and the model's scores shuffled."""
    rng = np.random.default_rng(seed)
    out = {"model": d["deep_pred"].to_numpy(np.float64)}
    out.update({name: signs[name] * d[col].to_numpy(np.float64) for name, col in BASELINES.items()})
    out["random"] = rng.normal(size=len(d))
    out["shuffled model"] = rng.permutation(out["model"])
    return out


def orient(val: pd.DataFrame) -> dict:
    """Baseline name -> +1 / -1, the sign of its rank IC on VAL (a baseline is allowed its better direction)."""
    return {name: (1.0 if metrics.rank_ic(val["deep"].values, val[col].values, val["ts"].values) >= 0 else -1.0)
            for name, col in BASELINES.items()}


def _ic_ci(d, score, baseline=None, minus=None, seed: int = 0) -> dict:
    """Per-timestamp rank IC (partial when ``baseline``; paired difference vs ``minus``), mean and day-block CI."""
    y, g = d["deep"].values, d["ts"].values
    keys, vals = metrics.group_rank_ics(y, score, g, baseline)
    if minus is not None:
        k2, v2 = metrics.group_rank_ics(y, minus, g)
        both, i1, i2 = np.intersect1d(keys, k2, return_indices=True)
        keys, vals = both, vals[i1] - v2[i2]
    if len(vals) == 0:
        return {"ic": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_ts": 0}
    day = pd.Series(d["day"].values, index=g).groupby(level=0).first().reindex(keys).values
    mean, lo, hi = metrics.block_bootstrap_ci(vals, day, seed=seed)
    return {"ic": mean, "lo": lo, "hi": hi, "n_ts": int(len(vals))}


def skill(d: pd.DataFrame, signs: dict, seed: int = 0, with_auc: bool = True) -> dict:
    """Per score: rank IC with its day-block CI and the AUC of sign(deep) > 0 with its day-block CI; for every
    non-model score also the paired (model - score) IC difference with its CI."""
    sc = score_columns(d, signs, seed)
    up = (d["deep"].values > 0).astype(np.float64)
    out = {}
    for name, s in sc.items():
        r = {"ic": _ic_ci(d, s, seed=seed)}
        if with_auc:
            r["auc"] = dict(zip(("auc", "lo", "hi"), metrics.auc_block_ci(up, s, d["day"].values, seed=seed),
                                strict=True))
        if name != "model":
            r["model_minus_it"] = _ic_ci(d, sc["model"], minus=s, seed=seed)
        out[name] = r
    return out


def close_link(d: pd.DataFrame, signs: dict, seed: int = 0) -> dict:
    """Is deep the close direction renamed? Spearman of deep with sign(true close move) and of deep_pred with the
    close mu (pooled and per timestamp), and the model's partial IC net of each baseline and of all of them."""
    g = d["ts"].values
    out = {"spearman_deep_vs_close_sign": metrics.spearman(d["deep"].values, np.sign(d["y_ret"].values)),
           "ic_deep_vs_close_sign": metrics.rank_ic(d["deep"].values, np.sign(d["y_ret"].values), g),
           "spearman_pred_vs_close_mu": metrics.spearman(d["deep_pred"].values, d["close_log"].values),
           "ic_pred_vs_close_mu": metrics.rank_ic(d["deep_pred"].values, d["close_log"].values, g),
           "share_deep_up": float((d["deep"].values > 0).mean()),
           "share_pred_up": float((d["deep_pred"].values > 0).mean()),
           "sign_agreement_deep_close": float((np.sign(d["deep"].values) == np.sign(d["y_ret"].values)).mean())}
    model = d["deep_pred"].values
    net = {name: _ic_ci(d, model, baseline=d[col].values, seed=seed) for name, col in BASELINES.items()}
    net["all baselines"] = _ic_ci(d, model, baseline=d[list(BASELINES.values())].to_numpy(np.float64), seed=seed)
    out["model_net_of"] = net
    return out


def _bucket_rows(d, signs, seed) -> dict:
    sk = skill(d, signs, seed, with_auc=False)
    best = max((n for n in BASELINES), key=lambda n: sk[n]["ic"]["ic"])
    return {"n": int(len(d)), "model": sk["model"]["ic"], "best_baseline": best, "best_baseline_ic": sk[best]["ic"],
            "model_minus_best": sk[best]["model_minus_it"]}


def vol_buckets(val: pd.DataFrame, test: pd.DataFrame, signs: dict, n: int = N_VOL_BUCKETS, seed: int = 0) -> dict:
    """Model and best-baseline IC inside volatility (ATR-14) buckets whose edges are VAL quantiles."""
    edges = np.quantile(val["vol"].values, np.linspace(0, 1, n + 1)[1:-1])
    out = {"edges": [float(e) for e in edges]}
    for split, d in (("VAL", val), ("TEST", test)):
        b = np.searchsorted(edges, d["vol"].values, side="right")
        out[split] = {int(k): _bucket_rows(d[b == k], signs, seed) for k in range(n)}
    return out


def near_ties(val: pd.DataFrame, test: pd.DataFrame, signs: dict, share: float = TIE_SHARE, seed: int = 0) -> dict:
    """Skill after dropping rows with |deep| below the VAL ``share`` quantile of |deep| (the same margin on TEST)."""
    margin = float(np.quantile(np.abs(val["deep"].values), share))
    out = {"share": share, "margin": margin}
    for split, d in (("VAL", val), ("TEST", test)):
        kept = d[np.abs(d["deep"].values) >= margin]
        out[split] = {"n": int(len(kept)), "dropped": int(len(d) - len(kept)), "skill": skill(kept, signs, seed)}
    return out


def clip_counts(d: pd.DataFrame, level: float = CLIP_LEVEL) -> dict:
    """Rows whose high, low or close label sits on the clip |log| = ``level`` (B-054), and rows with either extreme."""
    at = {t: np.abs(np.log1p(d[c].to_numpy(np.float64))) >= level - _CLIP_TOL
          for t, c in (("high", "fut_high"), ("low", "fut_low"), ("close", "y_ret"))}
    return {"rows": int(len(d)), **{t: int(v.sum()) for t, v in at.items()},
            "high_or_low": int((at["high"] | at["low"]).sum()), "high_and_low": int((at["high"] & at["low"]).sum())}


def _passes(sk: dict) -> dict:
    m = sk["model"]
    worst = min((n for n in sk if n != "model"), key=lambda n: sk[n]["model_minus_it"]["lo"])
    beats = all(sk[n]["model_minus_it"]["lo"] > 0 for n in sk if n != "model")
    return {"ic_ci_above_0": bool(m["ic"]["lo"] > 0), "auc_ci_above_half": bool(m["auc"]["lo"] > 0.5),
            "beats_every_baseline": bool(beats), "hardest_baseline": worst}


def deep_side(val: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict:
    """Items 1-5 of the deep-side check on two ``deep_frame`` tables, the clip counts and the verdict."""
    signs = orient(val)
    res = {"signs": signs, "clip": {"VAL": clip_counts(val), "TEST": clip_counts(test)}}
    res["skill"] = {"VAL": skill(val, signs, seed), "TEST": skill(test, signs, seed)}
    res["close_link"] = {"VAL": close_link(val, signs, seed), "TEST": close_link(test, signs, seed)}
    res["vol_buckets"] = vol_buckets(val, test, signs, seed=seed)
    res["near_ties"] = near_ties(val, test, signs, seed=seed)
    checks = {s: _passes(res["skill"][s]) for s in ("VAL", "TEST")}
    ok = all(c["ic_ci_above_0"] and c["auc_ci_above_half"] and c["beats_every_baseline"] for c in checks.values())
    res["verdict"] = {"checks": checks, "skill_above_every_baseline_both_splits": bool(ok),
                      "profit_section": "run" if ok else "not reached (the IC does not beat the baselines)"}
    return res


def _ci(r: dict, f: str = "ic", pct: bool = False) -> str:
    k = 100.0 if pct else 1.0
    return f"{r[f] * k:+.4f} [{r['lo'] * k:+.4f}, {r['hi'] * k:+.4f}]"


def _skill_lines(sk: dict) -> list[str]:
    lines = [f"    {'score':<22}{'rank IC [CI95]':<32}{'AUC [CI95]':<30}model - it [CI95]"]
    for name, r in sk.items():
        auc = _ci(r["auc"], "auc") if "auc" in r else "-"
        diff = _ci(r["model_minus_it"]) if "model_minus_it" in r else "-"
        lines.append(f"    {name:<22}{_ci(r['ic']):<32}{auc:<30}{diff}")
    return lines


def render(res: dict) -> str:
    """The text report of ``deep_side``."""
    lines = ["DEEP SIDE - deep = log(fh/lc) + log(fl/lc); deep_pred = mu_high + mu_low (decoded); rank IC per "
             "timestamp, CI = day-block bootstrap 95 %", f"baseline signs (from VAL): {res['signs']}", "",
             "LABEL CLIP (|log| = 1, B-054): rows per split"]
    lines += [f"    {s}: {c}" for s, c in res["clip"].items()]
    for s, sk in res["skill"].items():
        lines += ["", f"1-2. SKILL {s}", *_skill_lines(sk)]
    for s, cl in res["close_link"].items():
        lines += ["", f"3. CLOSE LINK {s}: " + ", ".join(f"{k} {v:+.4f}" for k, v in cl.items() if k != "model_net_of")]
        lines += [f"    model net of {n:<22}{_ci(r)}" for n, r in cl["model_net_of"].items()]
    vb = res["vol_buckets"]
    lines += ["", f"4. VOLATILITY BUCKETS (ATR-14, VAL edges {[round(e, 4) for e in vb['edges']]})"]
    for s in ("VAL", "TEST"):
        for k, r in vb[s].items():
            lines.append(f"    {s:<5}bucket {k} n {r['n']:>7,}  model {_ci(r['model'])}  best baseline "
                         f"{r['best_baseline']} {r['best_baseline_ic']['ic']:+.4f}  model - best "
                         f"{_ci(r['model_minus_best'])}")
    nt = res["near_ties"]
    lines += ["", f"5. NEAR TIES dropped: |deep| < {nt['margin']:.5f} (VAL {nt['share']:.0%} quantile)"]
    for s in ("VAL", "TEST"):
        lines += [f"  {s}: kept {nt[s]['n']:,}, dropped {nt[s]['dropped']:,}", *_skill_lines(nt[s]["skill"])]
    v = res["verdict"]
    lines += ["", "VERDICT", *[f"    {s}: {c}" for s, c in v["checks"].items()],
              f"    skill above every baseline on both splits: {v['skill_above_every_baseline_both_splits']}",
              f"    6. profit after cost: {v['profit_section']}"]
    return "\n".join(lines)
