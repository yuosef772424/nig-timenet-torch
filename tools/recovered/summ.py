# Recovered from a past session (scratch file, not a repo module). Ran on the Colab runtime (put to /content/summ.py): summarises every <arm>_s<seed>/panel_report.json under the given out dirs into /content/look_summary.json.
# Original name: scratchpad/summ.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
import glob, json, os, sys
out = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*_s[0-9]/panel_report.json")):
        arm = os.path.basename(os.path.dirname(f))
        r = json.load(open(f)); t = r["test"]; u = r["uncertainty"]
        g = lambda sec, k: {x: t[sec][k]["gross"].get(x) for x in ("n_days", "win_share", "win_lo", "win_hi", "mean")} | {"net": t[sec][k]["net"].get("mean"), "net_lo": t[sec][k]["net"].get("mean_lo"), "net_hi": t[sec][k]["net"].get("mean_hi")}
        row = {"abs": {k: g("absolute", k) for k in t["absolute"]}, "neu": {k: g("neutral", k) for k in t["neutral"]},
               "abs_rel_sign": {x: t["absolute_relative_sign"]["gross"].get(x) for x in ("win_share", "win_lo", "win_hi", "mean")},
               "beats": [t["absolute"]["model"]["beats_every_baseline"], t["neutral"]["model"]["beats_every_baseline"]],
               "pairs": {m: {k: {x: v["model_minus_this"].get(x) for x in ("d_win", "d_win_lo", "d_win_hi", "d_mean", "d_mean_lo", "d_mean_hi", "wins")} for k, v in t[m].items() if k != "model"} for m in ("absolute", "neutral")},
               "ric": t["relative_rank_ic"], "sd": u["sd_ranks_error"], "cal": u["sharp_and_calibrated"], "bias": u["direction_bias"],
               "sel": {k: {"share": v["rows_share"], "m1": v["absolute_gross"].get("win_share"), "m1_net": v["absolute_net"].get("mean"), "m2": v["neutral_gross"].get("win_share"), "m2_days": v["neutral_gross"].get("n_days")} for k, v in u["confidence_selection"].items()},
               "val_m2": r["val"]["neutral"]["model"]["gross"].get("win_share"), "val_m1": r["val"]["absolute"]["model"]["gross"].get("win_share")}
        a = json.load(open(os.path.dirname(f) + "/baseline_summary.json"))
        row["best_epoch"], row["epochs_done"] = a["best_epoch"], a["epochs_done"]
        row["epoch_s"] = a.get("epoch_seconds_median")
        tr = json.load(open(os.path.dirname(f) + "/test_report.json"))
        row["ic"] = {k: tr["targets"][k]["skill"]["rank_ic_by_timestamp"] for k in tr["targets"]}
        out[f"{os.path.basename(os.path.dirname(d))}/{arm}"] = row
json.dump(out, open("/content/look_summary.json", "w"), default=float)
print(len(out), os.path.getsize("/content/look_summary.json"))
