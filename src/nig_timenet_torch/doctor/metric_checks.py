"""PURPOSE: Card D: do the numbers mean what we say? Every metric next to its trivial baseline, the decode path to
    price, profit after cost with its uncertainty, and the selection metric against the trading goal.
TAGS: doctor, card d, metric checks, trivial baseline, climatology, nll, coverage, sharpness, decode round trip,
    profit after cost, block bootstrap, selection metric, trading goal
PITFALLS: scored on the VALIDATION rows (the test split only when ``include_test``); "profit" is the battery's own
    definition (cost per SIDE, day-block CI): a strategy counts only when its CI is above 0 AND it beats every
    baseline row of its section; the high / low heads are price levels, never a direction (owner rule).

``metric_checks(ctx) -> list[Check]``.
"""

import math

import numpy as np

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.doctor.common import Check, DoctorContext, judge, reg_key
from nig_timenet_torch.eval.battery_profit import profit_sections
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.metrics import auc_block_ci, rank_ic, spearman
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.battery_run import stage_inputs

__all__ = ["metric_checks", "student_nll", "head_metrics", "profit_rows"]

_Z80 = 1.2815515655446004
_T = 0.02     # IC differences below this are not read


def student_nll(y, mu, nu, alpha, beta) -> np.ndarray:
    """Per-row negative log density of the NIG's Student-t predictive (2 alpha degrees of freedom, scale^2 =
    beta (1 + nu) / (nu alpha)) at ``y``: the proper score the evidential loss is built on, without its regulariser."""
    y, mu, nu, alpha, beta = (np.asarray(a, dtype=np.float64) for a in (y, mu, nu, alpha, beta))
    df = 2.0 * alpha
    s2 = beta * (1.0 + nu) / (nu * alpha)
    lg = np.vectorize(math.lgamma)
    return (-lg((df + 1) / 2) + lg(df / 2) + 0.5 * np.log(df * np.pi * s2)
            + (df + 1) / 2 * np.log1p((y - mu) ** 2 / (df * s2)))


def head_metrics(table, y: np.ndarray, y_train: np.ndarray, t: str) -> dict:
    """The model's numbers on one head next to the constant predictor / climatology fitted on the TRAIN labels."""
    k = head_spec("nig_regression", t).output_keys
    mu, nu, alpha, beta = (table[k[x]].to_numpy(dtype=np.float64) for x in ("mu", "nu", "alpha", "beta"))
    m, sd = float(np.mean(y_train)), float(np.std(y_train))
    df = 2.0 * alpha
    scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
    pred_sd = scale * np.sqrt(df / np.maximum(df - 2.0, 1e-6))
    ts = table["timestamp"].to_numpy()
    gauss = 0.5 * np.log(2 * np.pi * sd * sd) + (y - m) ** 2 / (2 * sd * sd)
    return {"ic_by_ts": float(rank_ic(y, mu, groups=ts)), "ic": float(spearman(y, mu)),
            "mae": float(np.abs(y - mu).mean()), "mae_constant": float(np.abs(y - np.median(y_train)).mean()),
            "nll": float(student_nll(y, mu, nu, alpha, beta).mean()), "nll_climatology": float(gauss.mean()),
            "coverage80_sd": float((np.abs(y - mu) <= _Z80 * pred_sd).mean()),
            "sd_ratio_vs_climatology": float(np.median(pred_sd) / sd),
            "mu_sd_over_label_sd": float(mu.std() / y.std()),
            "slope_y_on_mu": float(np.cov(y, mu)[0, 1] / max(mu.var(), 1e-300))}


def profit_rows(sections: dict) -> list[dict]:
    """One row per strategy family of the battery's T-profit sections: the model line, the best baseline line of the
    same family and whether the model is profit (CI above 0 and above every baseline)."""
    out = []
    for name, sec in sections.items():
        stats = sec.get("stats") or {}
        groups: dict[str, dict] = {}
        for label, s in stats.items():
            if not s:
                continue
            fam = label.split(" - ")[0] if " - " in label else name
            is_model = label.startswith("model") or label.endswith("- model") or label == "sized by 1/sigma"
            groups.setdefault(fam, {"model": None, "base": []})
            if is_model:
                groups[fam]["model"] = (label, s)
            else:
                groups[fam]["base"].append((label, s))
        for fam, g in groups.items():
            if g["model"] is None:
                continue
            label, s = g["model"]
            best = max(g["base"], key=lambda b: b[1]["avg"]) if g["base"] else (None, {"avg": float("nan")})
            out.append({"family": fam if fam == name else f"{name}: {fam}", "avg": s["avg"], "ci_lo": s["ci_lo"],
                        "ci_hi": s["ci_hi"],
                        "n": s["n"], "best_baseline": best[0], "best_baseline_avg": best[1]["avg"],
                        "profit": bool(s["ci_lo"] > 0 and (not g["base"] or s["avg"] > best[1]["avg"]))})
    return out


def metric_checks(ctx: DoctorContext, *, include_test: bool = False) -> list[Check]:
    ds, r = ctx.dataset, ctx.run
    checks: list[Check] = []
    ceiling = ctx.notes.get("ceiling", {})
    codec_state = run_common.codec_state(ds)
    codec = TARGET_CODECS[codec_state["mode"]]()
    lc = np.asarray(ds["last_candles"])
    for stage in ("val", "test") if include_test else ("val",):
        table = r.table(stage)
        idx = r.idx[stage]
        for t in ctx.targets:
            y = r.y[stage][reg_key(t)].numpy().astype(np.float64)
            m = head_metrics(table, y, r.y["train"][reg_key(t)].numpy(), t)
            where = f"metrics/{stage}/{t}"
            c = ceiling.get(t, {})
            simple = max((v["ic_by_ts"] for kk, v in c.items() if kk.endswith(f"_{stage}")), default=float("nan"))
            both_zero = abs(m["ic_by_ts"]) < _T and not simple > _T
            v = "info" if both_zero or not math.isfinite(simple) else judge(
                m["ic_by_ts"] - simple, fail_below=-max(2 * _T, 0.2 * abs(simple)), warn_below=-_T)
            checks.append(Check("D", where, "rank IC by timestamp: model vs the simple model on the same inputs",
                                m["ic_by_ts"], simple, v,
                                "the model ranks worse than ridge / boosting on HAR-style features: it adds nothing "
                                "over a volatility baseline (cards A and B say where it loses)" if v in ("warn", "fail")
                                else "neither reads this label (card C, input information)" if both_zero else "", m))
            skill = 1.0 - m["mae"] / m["mae_constant"]
            checks.append(Check("D", where, "MAE skill vs the constant predictor (1 - MAE / MAE constant)", skill,
                                "0 = no better than the train median", judge(skill, fail_below=0.0, warn_below=0.005),
                                "" if skill >= 0.005 else "the point forecast is not better than a constant"))
            dn = m["nll_climatology"] - m["nll"]
            checks.append(Check("D", where, "NLL gain over climatology (train mean / sd Gaussian), nats per row", dn,
                                "0 = the predictive distribution is no better than a fixed one",
                                judge(dn, fail_below=0.0, warn_below=0.01),
                                "" if dn >= 0.01 else "the predictive distribution scores no better than one fixed "
                                "Gaussian: the uncertainty outputs carry no information (or are miscalibrated)"))
            cov = m["coverage80_sd"]
            checks.append(Check("D", where, "coverage of mu +- 1.28 predictive sd / median sd vs climatology sd",
                                f"{cov:.3f} / {m['sd_ratio_vs_climatology']:.2f}", "0.80 (0.88 for a skewed high / "
                                "low label, B-056) / < 1 = sharper than climatology",
                                judge(cov, fail_below=0.6, warn_below=0.72, warn_above=0.92, fail_above=0.97),
                                "" if 0.72 <= cov <= 0.92 else "the intervals are too narrow (coverage < 0.72) or too "
                                "wide (> 0.92): the predictive sd is off (head scale / beta floor, card C units)"))
            checks.append(Check("D", where, "slope of the label on mu / sd(mu) over sd(label)",
                                f"{m['slope_y_on_mu']:.2f} / {m['mu_sd_over_label_sd']:.2f}", "1.0 = calibrated mean",
                                "info" if both_zero else judge(m["slope_y_on_mu"], fail_below=0.5, warn_below=0.75,
                                                               warn_above=1.33, fail_above=2.0),
                                "" if both_zero or 0.75 <= m["slope_y_on_mu"] <= 1.33 else "slope > 1: predictions are "
                                "shrunk towards their average; < 1: over-dispersed (noise fitted)"))
        # D2 decode path: the round trip of the predictions and the price-space error
        for t in ctx.targets:
            mu = table[head_spec("nig_regression", t).output_keys["mu"]].to_numpy(dtype=np.float64)
            anchor = flat_run.anchors_by_name(lc, idx, [t], codec_state)[t]
            price = codec.decode(mu, codec_state, anchor)
            back = codec.encode(np.stack([anchor, price], axis=1), {**codec_state, "clip": None})
            err = float(np.abs(back - mu).max())
            fut = lc[idx, LAST_COLUMN_INDEX[{"high": "future_high_max", "low": "future_low_min",
                                             "close": "future_close"}[t]]]
            y = r.y[stage][reg_key(t)].numpy().astype(np.float64)
            rel = np.abs(price / fut - 1.0)
            lab = np.abs(y - mu) / codec_state["scale"]
            ratio = float(np.median(rel) / max(np.median(lab), 1e-300))
            ok = err < 1e-6 and 0.8 <= ratio <= 1.25
            checks.append(Check("D", f"metrics/{stage}/{t}", "decode: encode(decode(mu)) error / price error vs label "
                                "error (median ratio)", f"{err:.1e} / {ratio:.3f}", "0 / ~1", "pass" if ok else "fail",
                                "" if ok else "a prediction decoded to a price does not match the label-space error: "
                                "wrong anchor or codec for this head"))
        # direction of the close head with a block CI
        if "close" in ctx.targets:
            ck = head_spec("binary_classification", "close")
            if ck.output_keys["logits"] in table.columns:
                yc = r.y[stage][ck.label_key].numpy()
                days = table["prediction_made_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype(
                    "datetime64[D]")
                a, lo, hi = auc_block_ci(yc, table[ck.output_keys["logits"]].to_numpy(), days)
                checks.append(Check("D", f"metrics/{stage}/close", "direction AUC of the close class head (day-block "
                                    "CI)", f"{a:.4f} [{lo:.4f}, {hi:.4f}]", "0.5", "pass" if lo > 0.5 else "fail",
                                    "" if lo > 0.5 else "no direction skill: the CI contains 0.5"))
        # D3 profit after cost
        cost = r.settings.evaluation.battery.cost_per_side
        bt = stage_inputs(r, stage).battery_table()
        rows = profit_rows(profit_sections(bt, cost, 1))
        any_profit = False
        for row in rows:
            any_profit |= row["profit"]
            checks.append(Check("D", f"profit/{stage}", f"{row['family']}: net return per trade % (CI)",
                                f"{row['avg'] * 100:+.4f} [{row['ci_lo'] * 100:+.4f}, {row['ci_hi'] * 100:+.4f}]",
                                f"best baseline {row['best_baseline']}: {row['best_baseline_avg'] * 100:+.4f}",
                                "pass" if row["profit"] else "fail",
                                "" if row["profit"] else "not profit: the CI contains 0 or a trivial rule does as well",
                                row))
        checks.append(Check("D", f"profit/{stage}", "GOAL: any use of the outputs profitable after cost "
                            f"({cost:.2%} per side)", "yes" if any_profit else "no", "",
                            "pass" if any_profit else "fail",
                            "" if any_profit else "the model is NOT ready for trading on this split: read the failing "
                            "rows above and cards A-C for the stage to work on"))
    # D4 selection metric vs the goal
    sm = r.trainer.selection_metric
    checks.append(Check("D", "selection", "selection metric vs the goal", sm, "goal: profit after cost", "info",
                        "the epoch is picked by a likelihood-type validation loss; nothing in the fit measures profit "
                        "or rank skill per epoch (card B 'picked epoch' shows the regret against MAE)"
                        if sm == "val_raw_loss" else ""))
    return checks
