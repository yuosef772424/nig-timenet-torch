# Recovered from a past session (scratch file, not a repo module). Ran on the sandbox: renders markdown tables (arms x seeds) from the look_summary.json that summ.py wrote; python tables.py SUMMARY.json real|e4.
# Original name: scratchpad/tables.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
"""Scratch: markdown tables from look_summary.json (arms x seeds)."""
import json, sys
s = json.load(open(sys.argv[1])); which = sys.argv[2]   # real | e4
rows = {}
for k, v in s.items():
    ds, arm = k.split("/"); name, seed = arm.rsplit("_s", 1)
    if ds.replace("_c", "") == which: rows.setdefault(name, {})[int(seed)] = v
order = ["per_coin", "relative", "relative_mean_block", "attention_block", "per_coin_rank_loss", "relative_ic_loss", "relative_rank_loss",
         "relative_rank_loss_ls", "relative_center", "relative_rank_loss_center"]
pc = lambda x: "n/a" if x is None else f"{x * 100:.1f}"
print("| arm | seed | best / epochs | metric 1 winning days [CI] | mean day net | metric 2 winning days [CI] | mean day gross / net | relative rank IC | rank IC high / low / close | beats every baseline (M1 / M2) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name in order:
    for seed in sorted(rows.get(name, {})):
        v = rows[name][seed]; a, n = v["abs"]["model"], v["neu"]["model"]
        print(f"| {name} | {seed} | {v['best_epoch']} / {v['epochs_done']} | {pc(a['win_share'])} % [{pc(a['win_lo'])}, {pc(a['win_hi'])}] | {a['net'] * 100:+.4f} % | "
              f"{pc(n['win_share'])} % [{pc(n['win_lo'])}, {pc(n['win_hi'])}] | {n['mean'] * 100:+.4f} % / {n['net'] * 100:+.4f} % | {v['ric']['model']:+.4f} | "
              f"{v['ic']['high']:.3f} / {v['ic']['low']:.3f} / {v['ic']['close']:.3f} | {'yes' if v['beats'][0] else 'no'} / {'yes' if v['beats'][1] else 'no'} |")
any_ = next(iter(next(iter(rows.values())).values()))
print("\nBaselines (TEST, the same days, gross):")
for m, sec in (("metric 1", "abs"), ("metric 2", "neu")):
    print(f"- {m}: " + "; ".join(f"{k} {pc(v['win_share'])} % [{pc(v['win_lo'])}, {pc(v['win_hi'])}], mean day {v['mean'] * 100:+.4f} % (net {v['net'] * 100:+.4f} %)" for k, v in any_[sec].items() if k != "model") + ".")
print("- relative rank IC: " + ", ".join(f"{k} {v:+.4f}" for k, v in any_["ric"].items() if k != "model") + ".")
print("\n| arm | seed | sd vs abs error, close: across coins / across time | high | low | T-sharp close / high / low | coverage 80 % close / high / low | z sd close | predicted up (realised) | per-coin predicted-up range | one-sign timestamps | metric 1 with the relative sign |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name in order:
    for seed in sorted(rows.get(name, {})):
        v = rows[name][seed]; sd, c, b = v["sd"], v["cal"], v["bias"]
        f = lambda t: f"{sd[t]['across_coins']:+.2f} / {sd[t]['across_time']:+.2f}"
        print(f"| {name} | {seed} | {f('close')} | {f('high')} | {f('low')} | {c['close']['t_sharp']:.2f} / {c['high']['t_sharp']:.2f} / {c['low']['t_sharp']:.2f} | "
              f"{c['close']['coverage_80']:.2f} / {c['high']['coverage_80']:.2f} / {c['low']['coverage_80']:.2f} | {c['close']['z_sd']:.2f} | {b['share_pred_up']:.2f} ({b['share_real_up']:.2f}) | "
              f"{b['per_coin_pred_up_min']:.2f} .. {b['per_coin_pred_up_max']:.2f} | {b['one_sign_timestamps']:.2f} | {pc(v['abs_rel_sign']['win_share'])} % |")
print("\n| arm | seed | rows kept: all | top 20 % confidence | lowest 20 % sd | top 20 % abs mean | all rows ranked by mean / sd |")
print("|---|---|---|---|---|---|---|")
for name in order:
    for seed in sorted(rows.get(name, {})):
        v = rows[name][seed]["sel"]
        g = lambda k: f"{pc(v[k]['m1'])} / {pc(v[k]['m2'])}" if k in v else "n/a"
        print(f"| {name} | {seed} | {g('all_rows')} | {g('top_confidence')} | {g('low_sd')} | {g('top_abs_mu')} | {g('ranked_by_mu_over_sd')} |")
