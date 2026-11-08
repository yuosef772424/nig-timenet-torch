"""PURPOSE: Gates B (confidence), C (uncertainty), rule D (information) and the leakage canary of MODEL_DEV_PROTOCOL
    section 2 as check rows per env x seed from a stage's table and target_report; per gate the AND over seeds.
TAGS: gates, gate report, gate b, gate c, rule d, leakage canary, oracle relative, day block ci, verdicts, e1 e2 e3
PITFALLS: on E1 a fixed band the ORACLE itself fails (high / low symmetric coverage ~0.88, B-056) is judged relative
    to the oracle on the same rows (D-019); high / low coverage on E2 / E3 is information until the owner decides
    B-056; close-head T-sharp is information (O-1); rule D, the HAR rows and the E1 signal check never block.

Thresholds are the protocol's tables (constants below); ``ORACLE_TOL`` / ``ORACLE_TOL_ONE_SIDED`` are the D-019
tolerances. ``stage_rows(stage)`` -> check rows ``{env, seed, gate, check, head, value, threshold, verdict, blocking,
reference, mode, note}``; ``gate_a_rows`` turns a ``grad_report`` into rows; ``summarize(rows)`` -> verdicts per
(env, gate) and ``passed``; ``report_markdown`` renders a gate report.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.target_report import NON_BLOCKING_HEADS
from nig_timenet_torch.eval.vol_baseline import head_vs_baselines
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.stats.nig import nig_predictive_std

__all__ = ["GateStage", "stage_rows", "gate_a_rows", "summarize", "report_markdown", "ENVS", "MIN_SEEDS",
           "B_E1_DECILE", "B_E1_NOISE", "B_E3_DECILE", "C_E1_NOISE", "C_SCALE_BAND", "COVERAGE_BAND", "C_SHARP_TOL",
           "T_SHARP_MIN", "SKILL_SHARE", "CANARY_SE", "CANARY_ABS_IC", "CANARY_LOSS", "ORACLE_TOL",
           "ORACLE_TOL_ONE_SIDED", "Z80"]

ENVS = ("E1", "E2", "E3")
MIN_SEEDS = 3                       # protocol section 1: the seed set
B_E1_DECILE, B_E1_NOISE, B_E3_DECILE = -0.8, 0.7, -0.5
C_E1_NOISE = 0.8
C_SCALE_BAND = (0.8, 1.25)
COVERAGE_BAND = (0.75, 0.85)
C_SHARP_TOL = 0.10                  # E1: T-sharp share within this of the oracle's
T_SHARP_MIN = 0.80                  # E3: high / low T-sharp share
SKILL_SHARE = 0.7                   # section 3: E1 rank IC >= this x the oracle's
CANARY_SE, CANARY_ABS_IC, CANARY_LOSS = 3.0, 0.01, 0.01
ORACLE_TOL = None                   # two-sided band judged relative to the oracle: its half width (0.05 for coverage)
ORACLE_TOL_ONE_SIDED = 0.10         # one-sided bound judged relative to the oracle (the protocol's T-sharp tolerance)
Z80 = 1.2815515655446004            # the oracle's symmetric 80 % interval: mean +- this x sd
INFO_SHARP_HEADS = NON_BLOCKING_HEADS     # O-1: close-head T-sharp is information
#: the running-max / running-min heads: their symmetric coverage on E2 / E3 is information until B-056 is decided
INFO_COVERAGE_HEADS = tuple(t for t in TARGET_COLUMNS if t not in NON_BLOCKING_HEADS)
DIRECTION_HEAD = "close"                 # rule D: direction is judged on this head only
_DAY_NS = 86_400 * 10 ** 9


@dataclass
class GateStage:
    """One evaluated stage of one gate run (the env's held-out split, the trained model of one seed)."""
    env: str
    seed: int
    table: pd.DataFrame                 # PredictionTable
    labels: dict                        # label key -> array (aligned)
    report: dict                        # eval.target_report output of the same rows
    anchors: dict                       # target -> codec anchor price
    last_close: np.ndarray
    codec_state: dict
    ranges: dict | None = None          # eval.vol_baseline.window_ranges of the rows' input windows
    last_move: np.ndarray | None = None  # last bar's close log move (momentum baseline of rule D)
    oracle: dict | None = None          # E1: target -> (mean, sd) in label units
    losses: dict | None = None          # E2: {"model": val raw loss, "constant": constant-prediction loss}


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _band(value, lo, hi, oracle=None, tol=ORACLE_TOL):
    """(verdict, mode): inside [lo, hi]; when the oracle itself is outside, |value - oracle| <= tol (half width)."""
    v = _f(value)
    if oracle is None or lo <= oracle <= hi:
        return ("pass" if lo <= v <= hi else "fail"), "band"
    tol = (hi - lo) / 2.0 if tol is None else tol
    return ("pass" if abs(v - oracle) <= tol else "fail"), "relative to the oracle"


def _at_most(value, bound, oracle=None):
    v = _f(value)
    if oracle is None or oracle <= bound:
        return ("pass" if v <= bound else "fail"), "bound"
    return ("pass" if v <= oracle + ORACLE_TOL_ONE_SIDED else "fail"), "relative to the oracle"


def _at_least(value, bound, oracle=None):
    v = _f(value)
    if oracle is None or oracle >= bound:
        return ("pass" if v >= bound else "fail"), "bound"
    return ("pass" if v >= oracle - ORACLE_TOL_ONE_SIDED else "fail"), "relative to the oracle"


class _Rows:
    def __init__(self, stage: GateStage):
        self.stage, self.rows = stage, []

    def add(self, gate, check, head, value, verdict, threshold, *, blocking=True, reference=None, mode="", note=""):
        self.rows.append({"env": self.stage.env, "seed": int(self.stage.seed), "gate": gate, "check": check,
                          "head": head, "value": value, "threshold": threshold, "verdict": verdict,
                          "blocking": bool(blocking), "reference": reference, "mode": mode, "note": note})


def _head(stage: GateStage, t: str) -> dict:
    spec, n = head_spec("nig_regression", t), len(stage.table)
    col = lambda k: stage.table[spec.output_keys[k]].to_numpy(np.float64)  # noqa: E731
    y = np.asarray(stage.labels[spec.label_key], np.float64).reshape(-1)
    if y.shape[0] != n:
        raise ValueError(f"label {spec.label_key} has {y.shape[0]} rows, the table {n}")
    mu = col("mu")
    return {"y": y, "mu": mu, "std": nig_predictive_std(col("nu"), col("alpha"), col("beta")),
            "conf": col("confidence"), "err": np.abs(mu - y)}


def _ctx(stage: GateStage) -> dict:
    ts = stage.table["timestamp"]
    ns = ts.astype("int64").to_numpy() if not np.issubdtype(np.asarray(ts).dtype, np.integer) else np.asarray(ts)
    return {"assets": stage.table["asset"].to_numpy(dtype=object), "ts": ns, "days": ns // _DAY_NS}


def _oracle_scores(stage: GateStage, t: str, h: dict, c: dict) -> dict:
    """What the oracle scores on the same rows with the same functions (symmetric 80 % interval)."""
    mean, sd = (np.asarray(a, np.float64) for a in stage.oracle[t])
    codec = TARGET_CODECS[stage.codec_state["mode"]]()
    anchor = np.asarray(stage.anchors[t], np.float64)
    pred = codec.decode(mean, stage.codec_state, anchor)
    width = codec.decode(mean + sd, stage.codec_state, anchor) - pred
    err = np.abs(mean - h["y"])
    return {"decile": metrics.confidence_error_spearman(err, -sd, c["assets"]),
            "gap_lo": metrics.confidence_error_gap(err, -sd, c["assets"], blocks=c["days"], seed=stage.seed)[1],
            "coverage": metrics.interval_coverage(h["y"], mean - Z80 * sd, mean + Z80 * sd),
            "share": metrics.width_below_share(width, np.abs(pred - np.asarray(stage.last_close, np.float64))),
            "rank_ic": metrics.rank_ic(h["y"], mean, c["ts"]), "sd": sd}


def _gate_b(r: _Rows, t: str, h: dict, c: dict, rep: dict, o: dict | None) -> None:
    env, seed = r.stage.env, r.stage.seed
    gap, lo, hi = metrics.confidence_error_gap(h["err"], h["conf"], c["assets"], blocks=c["days"], seed=seed)
    dec = rep["conf"]["decile_error_spearman"]
    if env == ENVS[1]:
        r.add("B", "gap_ci_contains_0", t, [lo, hi], "pass" if lo <= 0.0 <= hi else "fail", "CI95 contains 0",
              mode="day-block CI", note=f"gap {gap:.4f}")
        return
    bound = B_E1_DECILE if env == ENVS[0] else B_E3_DECILE
    v, mode = _at_most(dec, bound, None if o is None else o["decile"])
    r.add("B", "decile_error_spearman", t, dec, v, f"<= {bound}", mode=mode,
          reference=None if o is None else o["decile"])
    v, mode = _at_least(lo, 0.0, None if o is None else o["gap_lo"])
    r.add("B", "gap_ci95_lower", t, lo, "pass" if lo > 0 else v if mode != "bound" else "fail", "> 0",
          mode=f"day-block CI; {mode}", reference=None if o is None else o["gap_lo"],
          note=f"gap {gap:.4f}, iid CI {rep['conf']['gap_ci95']}")
    if o is not None:
        rho = metrics.within_group_spearman(h["conf"], -o["sd"], c["assets"])
        r.add("B", "conf_vs_planted_noise", t, rho, _at_least(rho, B_E1_NOISE)[0], f">= {B_E1_NOISE}",
              note="within-coin Spearman(confidence, -oracle sd)")


def _gate_c(r: _Rows, t: str, h: dict, c: dict, rep: dict, o: dict | None) -> None:
    env, sharp = r.stage.env, rep["sharp"]
    cov = sharp["coverage"]
    info_cov = env != ENVS[0] and t in INFO_COVERAGE_HEADS
    v, mode = _band(cov, *COVERAGE_BAND, None if o is None else o["coverage"])
    r.add("C", "coverage_80", t, cov, v, list(COVERAGE_BAND), blocking=not info_cov, mode=mode,
          reference=None if o is None else o["coverage"],
          note="information until the owner decides B-056" if info_cov else "")
    if o is not None:
        rho = metrics.within_group_spearman(h["std"], o["sd"], c["assets"])
        r.add("C", "std_vs_planted_noise", t, rho, _at_least(rho, C_E1_NOISE)[0], f">= {C_E1_NOISE}",
              note="within-coin Spearman(pred std, oracle sd)")
        ratio = float(np.nanmean(h["std"]) / np.nanmean(o["sd"]))
        r.add("C", "std_scale_vs_planted", t, ratio, _band(ratio, *C_SCALE_BAND)[0], list(C_SCALE_BAND))
        d = abs(_f(sharp["share_std_below_move"]) - o["share"])
        r.add("C", "t_sharp_vs_oracle", t, sharp["share_std_below_move"], "pass" if d <= C_SHARP_TOL else "fail",
              f"|share - oracle| <= {C_SHARP_TOL}", blocking=t not in INFO_SHARP_HEADS, reference=o["share"],
              mode="relative to the oracle", note="O-1: information" if t in INFO_SHARP_HEADS else "")
    elif env == ENVS[1]:
        ratio = float(np.nanmean(h["std"]) / np.nanstd(h["y"]))
        r.add("C", "std_scale_vs_label_sd", t, ratio, _band(ratio, *C_SCALE_BAND)[0], list(C_SCALE_BAND))
    else:
        share = sharp["share_std_below_move"]
        r.add("C", "t_sharp_share", t, share, _at_least(share, T_SHARP_MIN)[0], f">= {T_SHARP_MIN}",
              blocking=t not in INFO_SHARP_HEADS, note="O-1: information" if t in INFO_SHARP_HEADS else "")


def _canary(r: _Rows, t: str, h: dict, c: dict, rep: dict) -> None:
    ic, se, n = metrics.rank_ic_se(h["y"], h["mu"], c["ts"])
    base = rep["skill"]["rank_ic_by_timestamp_shuffled_baseline"]
    ok = abs(ic) < CANARY_ABS_IC or abs(ic - base) <= CANARY_SE * se
    r.add("canary", "rank_ic_vs_shuffled", t, ic, "pass" if ok else "fail",
          f"|IC| < {CANARY_ABS_IC} or within {CANARY_SE:g} se of the shuffled baseline", reference=base,
          note=f"se {se:.4f}, {n} timestamps")


def _rule_d(r: _Rows, t: str, h: dict, c: dict) -> None:
    st = r.stage
    if t == DIRECTION_HEAD:
        if st.last_move is not None:
            _direction(r, h, c, np.asarray(st.last_move, np.float64))
    elif st.ranges is not None:
        sign = -1.0 if np.nanmean(h["y"]) < 0 else 1.0       # the low head: a bigger move ranks higher
        cmp = head_vs_baselines(h["mu"], h["y"], {k: st.ranges[k] for k in ("bar1", "mean6")}, c["ts"], c["assets"],
                                sign)
        r.add("D", "rank_ic_net_of_last_range", t, cmp["bar1"]["model_net_of_it"], "info", "> 0 is evidence",
              blocking=False, reference=cmp["bar1"]["rank_ic"],
              note=f"model rank IC {cmp['model']['rank_ic']:.4f}; reference = the last-bar range's IC")
        r.add("D", "rank_ic_net_of_har_mean6", t, cmp["mean6"]["model_net_of_it"], "info", "information (B-057)",
              blocking=False, reference=cmp["mean6"]["rank_ic"], note="HAR mean6 of the same window; not rule D")


def _direction(r: _Rows, h: dict, c: dict, last_move: np.ndarray) -> None:
    up = h["y"] > 0
    hit = lambda s: ((s > 0) == up).astype(np.float64)  # noqa: E731
    rng = np.random.default_rng(r.stage.seed)
    base = {"always_up": hit(np.ones_like(h["y"])), "always_down": hit(-np.ones_like(h["y"])),
            "shuffled": hit(rng.permutation(h["mu"])), "momentum": hit(last_move), "reversal": hit(-last_move)}
    best = max(base, key=lambda k: base[k].mean())
    mean, lo, hi = metrics.block_bootstrap_ci(hit(h["mu"]) - base[best], c["days"], seed=r.stage.seed)
    r.add("D", "close_direction_vs_best_trivial", DIRECTION_HEAD, mean, "info",
          "CI lower > 0 on every seed is evidence", blocking=False,
          reference={k: float(v.mean()) for k, v in base.items()},
          note=f"hit rate {hit(h['mu']).mean():.4f} vs {best}; day-block CI [{lo:.4f}, {hi:.4f}]")


def stage_rows(stage: GateStage) -> list[dict]:
    """Every gate B / C / canary / rule D / E1-signal row of one stage."""
    if stage.env not in ENVS:
        raise ValueError(f"unknown env {stage.env!r} (one of {ENVS})")
    r, c = _Rows(stage), _ctx(stage)
    for t, rep in stage.report["targets"].items():
        h = _head(stage, t)
        o = _oracle_scores(stage, t, h, c) if stage.oracle is not None else None
        _gate_b(r, t, h, c, rep, o)
        _gate_c(r, t, h, c, rep, o)
        if stage.env == ENVS[1]:
            _canary(r, t, h, c, rep)
        if o is not None:
            ic = rep["skill"]["rank_ic_by_timestamp"]
            r.add("S", "signal_recovery", t, ic, _at_least(ic, SKILL_SHARE * o["rank_ic"])[0],
                  f">= {SKILL_SHARE} x oracle", blocking=False, reference=o["rank_ic"],
                  note="protocol section 3 (E1 'why' column); not one of the gates A-C")
        _rule_d(r, t, h, c)
    if stage.losses is not None:
        m, k = stage.losses["model"], stage.losses["constant"]
        ok = m >= k - CANARY_LOSS * abs(k)
        r.add("canary", "val_loss_vs_constant", "all", m, "pass" if ok else "fail",
              f"not below the constant-prediction loss by more than {CANARY_LOSS:.0%}", reference=k)
    return r.rows


def gate_a_rows(env: str, seed: int, phase: str, report: dict) -> list[dict]:
    """Rows of one ``trainer.grad_report`` measurement (``phase``: "start" / "end"); the check's other fields (dead
    tensors, per-layer values) go to ``reference``."""
    return [{"env": env, "seed": int(seed), "gate": "A", "check": k, "head": phase, "value": _jsonable(c["value"]),
             "threshold": c["rule"], "verdict": c["verdict"], "blocking": True,
             "reference": _jsonable({f: v for f, v in c.items() if f not in ("verdict", "value", "rule")}) or None,
             "mode": phase, "note": ""} for k, c in report["checks"].items()]


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    return _f(v)


def summarize(rows: list[dict], *, envs=ENVS, min_seeds: int = MIN_SEEDS) -> dict:
    """Verdict per (env, gate, check, head) = AND over seeds of the blocking rows (A6: a fail blocks only when it
    fails on EVERY seed, else warn); per (env, gate) pass / fail; ``passed`` needs every env with >= ``min_seeds``
    seeds and every blocking key passing. Warn counts as pass."""
    keys: dict[tuple, list[str]] = {}
    seeds: dict[str, set] = {e: set() for e in envs}
    for row in rows:
        seeds.setdefault(row["env"], set()).add(row["seed"])
        if row["blocking"]:
            keys.setdefault((row["env"], row["gate"], row["check"], row["head"]), []).append(row["verdict"])
    per_key = {}
    for (env, gate, check, head), vs in keys.items():
        if check == "A6":
            v = "fail" if all(x == "fail" for x in vs) else "warn" if "fail" in vs or "warn" in vs else "pass"
        else:
            v = "fail" if "fail" in vs else "warn" if "warn" in vs else "pass"
        per_key[f"{env}|{gate}|{check}|{head}"] = v
    gates: dict[str, dict] = {}
    for k, v in per_key.items():
        env, gate = k.split("|")[:2]
        cur = gates.setdefault(env, {}).get(gate, "pass")
        gates[env][gate] = "fail" if "fail" in (cur, v) else "pass"
    missing = [e for e in envs if len(seeds.get(e, ())) < min_seeds]
    failed = sorted(k for k, v in per_key.items() if v == "fail")
    return {"per_check": per_key, "gates": gates, "seeds": {e: sorted(s) for e, s in seeds.items()},
            "envs_missing_seeds": missing, "failed": failed, "passed": not failed and not missing}


def _cell(v) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_cell(x)}" for k, x in list(v.items())[:6])
    if isinstance(v, list):
        return "[" + ", ".join(_cell(x) for x in v[:6]) + (", ..." if len(v) > 6 else "") + "]"
    return str(v)


def report_markdown(report: dict) -> str:
    """The gate report as markdown: verdicts per env and gate, then every row."""
    s = report["summary"]
    lines = [f"# Gate report `{report['key'][:16]}`", "",
             f"- passed: **{s['passed']}** (baseline arm: {report.get('baseline_arm')}); seeds {s['seeds']}; envs "
             f"missing seeds: {s['envs_missing_seeds'] or 'none'}",
             f"- configuration: `{report.get('configuration')}`", f"- code: `{report.get('code_sha')}`", "",
             "| env | " + " | ".join(("A", "B", "C", "canary")) + " |", "|---|---|---|---|---|"]
    for env in ENVS:
        g = s["gates"].get(env, {})
        lines.append(f"| {env} | " + " | ".join(g.get(k, "-") for k in ("A", "B", "C", "canary")) + " |")
    lines += ["", "Failed checks: " + (", ".join(s["failed"]) or "none"), "",
              "| env | seed | gate | check | head | value | threshold | verdict | blocking | reference | mode | note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append("| " + " | ".join(_cell(r[k]) for k in ("env", "seed", "gate", "check", "head", "value",
                                                               "threshold", "verdict", "blocking", "reference",
                                                               "mode", "note")) + " |")
    return "\n".join(lines) + "\n"
