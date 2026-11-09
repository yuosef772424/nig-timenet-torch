"""PURPOSE: The T5r reading of a finished run: volatility baselines of the high / low heads (rule C1), every use of
    the outputs with controls on VAL and TEST, deep diagnosis of profitable-looking rules, the evidence check.
TAGS: t5r reading, rule c1, volatility baseline, strategy book, strategy diagnosis, evidence check, run reading
PITFALLS: needs the high, low and close heads and the X window channels high / low / close as log(price / prev
    close); thresholds come from VAL only; 'evidence' needs the model to beat every control with a CI above 0 on BOTH
    splits and the same sign on both, and it stays exploratory (rule C5).

``reading_step(run, out_dir)`` writes ``reading_report.txt`` and ``reading.json``. ``extended_table`` adds to a
battery table what the controls need: ``ale_high`` / ``ale_low`` (aleatoric std of the head, return units vs the last
close), ``vol`` (ATR-type range of the last 14 input bars, return units), ``last_ret`` (last bar's close return).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.eval import error_map as em
from nig_timenet_torch.eval import strategy_diagnosis as sd
from nig_timenet_torch.eval import vol_baseline as vb
from nig_timenet_torch.eval.strategy_book import book
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.stats.nig import nig_interval_halfwidth, nig_nll, nig_predictive_std
from nig_timenet_torch.workflow import run_common
from nig_timenet_torch.workflow.battery_run import stage_inputs

__all__ = ["extended_table", "stage_data", "c1_section", "evidence", "reading_step", "row_coins", "error_frame",
           "coin_info", "REPORT", "JSON", "ERROR_REPORT"]

REPORT, JSON, ERROR_REPORT = "reading_report.txt", "reading.json", "error_map_report.txt"
HAR = ("bar1", "mean6", "mean32")
BASELINES = ("bar1", "mean6", "mean32", "atr14")


def extended_table(battery: pd.DataFrame, table: pd.DataFrame, ranges: dict, x_last_close: np.ndarray) -> pd.DataFrame:
    """The battery table plus ``ale_high``, ``ale_low``, ``vol`` and ``last_ret`` (see the module doc)."""
    out = battery.copy()
    for t in (c for c in TARGET_COLUMNS if c != "close"):
        spec = head_spec("nig_regression", t)
        ratio = (1.0 + out[f"pred_{t}"].to_numpy()) / (1.0 + table[spec.output_keys["mu"]].to_numpy())
        out[f"ale_{t}"] = table[spec.output_keys["mu"] + "_aleatoric"].to_numpy() * ratio
    out["vol"] = np.expm1(ranges["atr14"])
    out["last_ret"] = np.expm1(np.asarray(x_last_close, dtype=np.float64))
    return out


def stage_data(run, stage: str) -> dict:
    """One stage of ``run``: the extended battery table, the window ranges, the artifact rows, the prediction table
    and the labels (keys ``battery``, ``ranges``, ``idx``, ``table``, ``labels``, ``x_last``: the last input bar)."""
    si = stage_inputs(run, stage)
    ds, idx = run.dataset, run.idx[stage]
    x = np.asarray(ds[run_common.input_key(ds)])[idx]
    fo = list(ds["feature_order"])
    ranges = vb.window_ranges(x, fo)
    bt = extended_table(si.battery_table(), si.table, ranges, x[:, -1, fo.index("close")])
    return {"battery": bt, "ranges": ranges, "idx": idx, "table": si.table, "labels": si.labels,
            "x_last": dict(zip(fo, x[:, -1, :].T, strict=True))}


def _realised_log_range(lc: np.ndarray, idx: np.ndarray) -> np.ndarray:
    hi, lo = lc[idx, LAST_COLUMN_INDEX["future_high_max"]], lc[idx, LAST_COLUMN_INDEX["future_low_min"]]
    return np.log(np.maximum(hi, vb.EPS) / np.maximum(lo, vb.EPS))


def c1_section(run, val: dict, test: dict) -> dict:
    """Rule C1 numbers: per head and split the rank IC of model vs baselines and net of them; the HAR table; 80 %
    intervals at matched calibration."""
    out = {"heads": {}, "intervals": {}}
    for t, sign in (("high", 1.0), ("low", -1.0)):
        out["heads"][t] = {}
        for split, st in (("VAL", val), ("TEST", test)):
            b = st["battery"]
            out["heads"][t][split] = vb.head_vs_baselines(b[f"pred_{t}"].values, b[f"fut_{t}"].values,
                                                           st["ranges"], b["end_time"].values, b["coin"].values, sign)
        side = {split: {"y": st["battery"][f"fut_{t}"].values, "pred": st["battery"][f"pred_{t}"].values,
                        "std": st["battery"][f"sigma_{t}"].values, "base": st["ranges"]}
                for split, st in (("VAL", val), ("TEST", test))}
        out["intervals"][t] = vb.interval_vs_baseline(side["VAL"], side["TEST"])
    lc = np.asarray(run.dataset["last_candles"])
    prior_map = _coin_prior(run.dataset, run.idx["train"], _realised_log_range(lc, run.idx["train"]))
    feats = {}
    for split, st in (("VAL", val), ("TEST", test)):
        b = st["battery"]
        f = {k: np.log(np.maximum(v, vb.EPS)) for k, v in st["ranges"].items()}
        f["model"] = np.log(np.maximum(b["pred_high"].values - b["pred_low"].values, vb.EPS))
        f["coin prior"] = np.log(np.maximum(b["coin"].map(prior_map).fillna(prior_map["__all__"]).values, vb.EPS))
        feats[split] = (f, np.log(np.maximum(_realised_log_range(lc, st["idx"]), vb.EPS)))
    har = ["bar1", "mean6", "mean32"]
    sets = {"bar-1": ["bar1"], "ATR-14": ["atr14"], "HAR (bar-1, mean-6, mean-32)": har,
            "HAR + coin prior": [*har, "coin prior"], "model alone": ["model"], "model + HAR": ["model", *har],
            "model + HAR + coin prior": ["model", *har, "coin prior"]}
    out["har"] = vb.har_table(*feats["VAL"], *feats["TEST"], test["battery"]["coin"].values, sets)
    return out


def row_coins(dataset: dict, idx: np.ndarray) -> np.ndarray:
    """Coin name of each artifact row ``idx`` (from ``asset_bounds``)."""
    bounds = sorted(dataset["asset_bounds"], key=lambda b: b["start"])
    starts = np.array([b["start"] for b in bounds])
    names = np.array([b["name"] for b in bounds], dtype=object)
    return names[np.searchsorted(starts, np.asarray(idx), side="right") - 1]


def _coin_prior(dataset: dict, train_idx: np.ndarray, real_tr: np.ndarray) -> dict:
    """Coin -> mean realised (log) range over its TRAIN rows; ``__all__`` -> the train mean (coins unseen there)."""
    s = pd.Series(real_tr).groupby(row_coins(dataset, train_idx)).mean()
    return {**s.to_dict(), "__all__": float(real_tr.mean())}


def coin_info(run) -> pd.DataFrame:
    """Per coin: ``stale_share`` (share of TRAIN windows whose last bar has high == low: an illiquidity proxy, the
    artifact has no volume), ``rows`` (windows in the artifact), ``gaps`` (consecutive windows > one bar apart)."""
    ds = run.dataset
    tr = run.idx["train"]
    last = run.x["train"][:, -1, :].numpy()
    fo = list(ds["feature_order"])
    flat = (last[:, fo.index("high")] - last[:, fo.index("low")]) <= 1e-12
    stale = pd.Series(flat).groupby(row_coins(ds, tr)).mean()
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    bar = pd.Timedelta(ds["base_tf"]).value
    rec = {}
    for b in ds["asset_bounds"]:
        d = np.diff(ts[b["start"]:b["end"]])
        rec[b["name"]] = {"rows": b["end"] - b["start"], "gaps": int((d > bar).sum())}
    info = pd.DataFrame.from_dict(rec, orient="index")
    info["stale_share"] = stale.reindex(info.index).fillna(np.nan)
    return info


def error_frame(run, st: dict, har_w: dict | None, info: pd.DataFrame) -> pd.DataFrame:
    """The ``eval.error_map`` frame of one stage (label space; HAR fitted on VAL when ``har_w`` is given)."""
    ds, idx, tab, lab = run.dataset, st["idx"], st["table"], st["labels"]
    b = st["battery"]
    starts = {x["name"]: x["start"] for x in ds["asset_bounds"]}
    coins = row_coins(ds, idx)
    f = pd.DataFrame({"coin": coins, "end_time": b["end_time"].values, "vol": b["vol"].values,
                      "age_bars": idx - np.array([starts[c] for c in coins]),
                      "stale_share": info["stale_share"].reindex(coins).to_numpy(dtype=np.float64)})
    f["day"] = f["end_time"].values.astype("datetime64[D]")
    f["month"] = f["end_time"].dt.strftime("%Y-%m")
    f["hour"] = f["end_time"].dt.hour
    for t in run.targets:
        spec = head_spec("nig_regression", t)
        nu, al, be = (tab[spec.output_keys[k]].to_numpy(dtype=np.float64) for k in ("nu", "alpha", "beta"))
        mu, y = tab[spec.output_keys["mu"]].to_numpy(dtype=np.float64), np.asarray(lab[spec.label_key], np.float64)
        f[f"mu_{t}"], f[f"real_{t}"] = mu, y
        f[f"std_{t}"], f[f"half_{t}"] = nig_predictive_std(nu, al, be), nig_interval_halfwidth(nu, al, be, 0.8)
        f[f"nll_{t}"] = nig_nll(y, mu, nu, al, be)
        if har_w is not None and t in har_w:
            f[f"har_{t}"] = vb._ols_predict(st["ranges"], har_w[t], HAR)
    f["market_day"] = f.groupby("day")["real_close"].transform("mean") if "real_close" in f else 0.0
    return f


def evidence(rows: list[dict]) -> list[dict]:
    """Per use: the model's VAL / TEST mean and CI, the best control per split, and whether the model is evidence
    (beats every control AND CI above 0 on BOTH splits, same sign)."""
    out = []
    for use in dict.fromkeys(r["use"] for r in rows):
        per = {}
        for split in ("VAL", "TEST"):
            mine = [r for r in rows if r["use"] == use and r["split"] == split]
            model = next((r for r in mine if r["control"] == "model"), None)
            ctrl = [r for r in mine if r["control"] != "model" and r.get("avg") is not None]
            best = max(ctrl, key=lambda r: r["avg"]) if ctrl else None
            per[split] = {"model": model, "best_control": best}
        mv, mt = per["VAL"]["model"], per["TEST"]["model"]
        if not mv or not mt or "avg" not in mv or "avg" not in mt:
            continue
        beats = all(per[s]["best_control"] is None or per[s]["model"]["avg"] > per[s]["best_control"]["avg"]
                    for s in per)
        ci_pos = mv["ci_lo"] > 0 and mt["ci_lo"] > 0
        out.append({"use": use, "val": mv, "test": mt, "best_control_val": per["VAL"]["best_control"],
                    "best_control_test": per["TEST"]["best_control"], "same_sign": np.sign(mv["avg"]) ==
                    np.sign(mt["avg"]), "beats_controls": beats, "ci_above_zero_both": ci_pos,
                    "evidence": bool(beats and ci_pos and mv["avg"] > 0 and mt["avg"] > 0)})
    return out


def _book_lines(rows: list[dict]) -> list[str]:
    lines = [f"{'use':<38}{'control':<26}{'split':<6}{'n':>8}{'win%':>7}{'hit%':>7}{'avg%':>9}"
             f"{'CI95 lo':>9}{'CI95 hi':>9}{'total':>8}"]
    for r in rows:
        if "avg" not in r:
            lines.append(f"{r['use']:<38}{r['control']:<26}{r['split']:<6}{r['n']:>8}  (too few trades)")
            continue
        hit = f"{r['hit'] * 100:>7.1f}" if r.get("hit") is not None else f"{'-':>7}"
        lines.append(f"{r['use']:<38}{r['control']:<26}{r['split']:<6}{r['n']:>8,}{r['win'] * 100:>7.1f}{hit}"
                     f"{r['avg'] * 100:>+9.3f}{r['ci_lo'] * 100:>+9.3f}{r['ci_hi'] * 100:>+9.3f}"
                     f"{r['total_pts']:>+8.0f}")
    return lines


def reading_step(run, out_dir, cost: float = 0.0008) -> dict:
    """The whole reading of ``run`` (model holding the best weights) into ``out_dir``; returns the JSON dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val, test = stage_data(run, "val"), stage_data(run, "test")
    c1 = c1_section(run, val, test)
    rows = book(val["battery"], test["battery"], cost)
    ev = evidence(rows)
    lines = ["RULE C1 - volatility baselines (rank IC by timestamp; within-coin Spearman; model net of each)"]
    for t, by in c1["heads"].items():
        for split, d in by.items():
            lines.append(f"  {t:<4} {split:<4} model IC {d['model']['rank_ic']:+.4f} within-coin "
                         f"{d['model']['within_coin']:+.4f} | " + " | ".join(
                             f"{b}: IC {d[b]['rank_ic']:+.4f} wc {d[b]['within_coin']:+.4f} "
                             f"net {d[b]['model_net_of_it']:+.4f}"
                             for b in BASELINES))
    lines.append("HAR range table (OLS on VAL, scored on TEST): R2 / Spearman / within-coin Spearman")
    lines += [f"  {k:<32}{v['r2']:>8.3f}{v['spearman']:>9.3f}{v['within_coin']:>9.3f}" for k, v in c1["har"].items()]
    lines.append("80% intervals, calibrated on VAL (one scale each), scored on TEST")
    for t, iv in c1["intervals"].items():
        for name in ("model", "volatility_baseline"):
            m = iv[name]
            lines.append(f"  {t:<4} {name:<20} cov VAL {m['coverage_val']:.3f} TEST {m['coverage_test']:.3f}  mean "
                         f"halfwidth {m['mean_halfwidth_test'] * 100:.3f}%  share std<|move| "
                         f"{m['share_std_below_move_test']:.3f}")
    lines += ["", f"STRATEGY BOOK - cost {cost:.4%} per side x2; CI = day-block bootstrap 95%", *_book_lines(rows), "",
              "EVIDENCE per use (beats every control AND CI95 > 0 on BOTH splits, same sign)"]
    lines += [f"  {e['use']:<40} VAL {e['val']['avg'] * 100:+.3f}% TEST {e['test']['avg'] * 100:+.3f}%  same sign "
              f"{e['same_sign']}  beats controls {e['beats_controls']}  CI>0 both {e['ci_above_zero_both']}  -> "
              f"{'EVIDENCE' if e['evidence'] else 'no'}" for e in ev]
    long_all = {s: st["battery"].assign(day=st["battery"]["end_time"].values.astype("datetime64[D]"))
                .groupby("day")["y_ret"].mean() for s, st in (("VAL", val), ("TEST", test))}
    diag = {}
    for e in ev:
        if e["val"]["avg"] > 0 or e["test"]["avg"] > 0:
            for split in ("VAL", "TEST"):
                r = next(r for r in rows if r["use"] == e["use"] and r["split"] == split and r["control"] == "model")
                side = float(np.sign(r["trades"]["side"].mean())) or 1.0
                d = sd.diagnose(r["trades"], side * long_all[split], long_all[split])
                diag[f"{e['use']} | {split}"] = d
                lines += ["", sd.render(f"{split}  {e['use']}", d)]
    (out / REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = coin_info(run)
    har_w = {t: vb.ols_fit(val["ranges"], np.asarray(val["labels"][head_spec("nig_regression", t).label_key]), HAR)
             for t in run.targets if t != "close"}
    emap = em.error_map({"VAL": error_frame(run, val, har_w, info), "TEST": error_frame(run, test, har_w, info)},
                        heads=run.targets, coin_info=info)
    (out / ERROR_REPORT).write_text(em.render(emap) + "\n", encoding="utf-8")
    res = {"cost_per_side": cost, "c1": c1, "book": [{k: v for k, v in r.items() if k != "trades"} for r in rows],
           "evidence": ev, "diagnosis": diag, "error_map": emap}
    (out / JSON).write_text(json.dumps(res, indent=1, default=_json), encoding="utf-8")
    return res


def _json(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.generic):
        return v.item()
    return str(v)
