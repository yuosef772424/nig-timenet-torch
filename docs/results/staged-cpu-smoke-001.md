# Staged training (ROADMAP ST1), CPU smoke check on E1

**This is a smoke check of the machinery, not a result (rules C4, C5).** The environment is the synthetic E1 (8 coins, 3,200
bars, 10,232 train / 1,144 validation rows), a small model (transformer, d_model 32, 1 layer, no feed-forward), 2 seeds, one growth arm (`stride`). It shows that the staged loop, the sign, the stage table and the
equal-step baselines run end to end and how the output reads. It does not answer the card's question on the real data and
the real model: no GPU, no real artifact, no battery.

Command (CPU, 349 s): `python docs/results/cpu_smoke_st1_f2.py WORK staged '{"step_cap_epochs": 12, "eval_every": 6,
"warmup_steps": 10, "bn_refresh_steps": 10}'`; every other `StageSettings` field is the card's default (start share 1/32,
the share doubles, gamma 0.5, lr floor 1e-6, `k` 3, `min_delta` 1e-3, 3-point mean, probe 2,000 rows). The step cap is 12
passes (the card's 20 would need an hour on a CPU), the evaluation every 6 steps (the card's "at least 100" would leave stage 0
with no evaluation). A first run with a cap of 4 passes ended every stage by the cap and fired no sign (its output was not kept
as a result: too short to say anything).

Verdict: no gain by the accept rule (seeds beating B1 and B2 by more than the B1 seed spread 0.0005364: 0/2; test rank IC not lower on 0/2); see the per-stage tables.

| arm | seed | steps | best val | test rank IC targets.close.skill.rank_ic_by_timestamp | test rank IC targets.high.skill.rank_ic_by_timestamp | test rank IC targets.low.skill.rank_ic_by_timestamp |
|---|---|---|---|---|---|---|
| staged | 0 | 900 | -5.1855 | 0.0187 | 0.2294 | 0.2918 |
| staged | 1 | 654 | -5.1481 | 0.0169 | 0.2225 | 0.2963 |
| b1 | 0 | 900 | -5.2636 | 0.0513 | 0.2490 | 0.2985 |
| b1 | 1 | 654 | -5.2631 | 0.0613 | 0.2555 | 0.3126 |
| b2 | 0 | 900 | -5.2191 | 0.0653 | 0.2402 | 0.3135 |
| b2 | 1 | 654 | -5.2180 | 0.0701 | 0.2473 | 0.3107 |
| b3 | 0 | 900 | -4.7912 | 0.0137 | 0.2094 | 0.0716 |
| b3 | 1 | 654 | -4.8129 | 0.0026 | 0.2180 | 0.0201 |

## Stage table, staged arm, seed 0

| stage | rows | indep. labels | lr | steps | steps to sign | ended by | best val | gap at end | rank IC close | rank IC high | rank IC low |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | 320 | 1.00e-03 | 36 | - | cap | -4.7678 | 0.8287 | -0.0041 | 0.2071 | 0.0905 |
| 1 | 640 | 640 | 5.00e-04 | 60 | - | cap | -4.9446 | 0.6503 | 0.0097 | 0.2298 | 0.2202 |
| 2 | 1279 | 1279 | 2.50e-04 | 120 | - | cap | -5.0636 | 0.4084 | 0.0075 | 0.2386 | 0.2706 |
| 3 | 2558 | 2558 | 1.25e-04 | 198 | 198 | sign | -5.1318 | 0.2108 | 0.0201 | 0.2508 | 0.3141 |
| 4 | 5116 | 5116 | 6.25e-05 | 384 | 384 | sign | -5.1796 | 0.2296 | 0.0154 | 0.2701 | 0.3264 |
| 5 | 10232 | 10232 | 3.13e-05 | 102 | 102 | sign | -5.1855 | 0.0697 | 0.0168 | 0.2719 | 0.3277 |

## Stage table, staged arm, seed 1

| stage | rows | indep. labels | lr | steps | steps to sign | ended by | best val | gap at end | rank IC close | rank IC high | rank IC low |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | 320 | 1.00e-03 | 36 | - | cap | -4.8129 | 0.8891 | 0.0310 | 0.2358 | 0.0207 |
| 1 | 640 | 640 | 5.00e-04 | 60 | - | cap | -4.9488 | 0.6843 | 0.0450 | 0.2345 | 0.2019 |
| 2 | 1279 | 1279 | 2.50e-04 | 120 | - | cap | -5.0536 | 0.4456 | 0.0431 | 0.2376 | 0.2618 |
| 3 | 2558 | 2558 | 1.25e-04 | 198 | 198 | sign | -5.1200 | 0.2346 | 0.0443 | 0.2427 | 0.3010 |
| 4 | 5116 | 5116 | 6.25e-05 | 168 | 168 | sign | -5.1430 | 0.1836 | 0.0384 | 0.2505 | 0.3099 |
| 5 | 10232 | 10232 | 3.13e-05 | 72 | 72 | sign | -5.1481 | 0.1883 | 0.0388 | 0.2514 | 0.3121 |

What the smoke shows, and what it does not:
- The sign fires only in the last three stages (the larger the stage, the earlier the validation loss stops improving while
  the eval-mode train probe keeps falling); stages 0-2 end by their cap (36, 60, 120 steps), so "steps to the sign" against
  rows has three censored points and three measured ones here (198, 384, 102 steps and 198, 168, 72 for the rows 2,558,
  5,116, 10,232). Two seeds with different totals (900 and 654 steps) show that the budget of a run depends on where the
  signs fire; the baselines get the same total per seed (the table shows equal step counts per seed).
- By the accept rule of the card the staged arm shows no gain here: its best validation loss (-5.19 / -5.15) is worse than
  B1's (-5.26 / -5.26) and B2's (-5.22 / -5.22) on both seeds, and its test rank ICs are not higher. B3 (the 1/32 rows only)
  is the floor and is far behind on the low head (rank IC 0.07 / 0.02), as expected. With 2 seeds and one environment this is
  a look at the mechanics: the cap of 12 passes cuts the early stages short, the lr falls by half at each of 5 boundaries
  (3e-5 in the last stage), and the whole budget is 654-900 steps.
- Not verified here: the real artifact, the GPU, the card's cap of 20 passes, the arms `coins` and `time`, `new_weight` > 1,
  the effect of each of the improvements 1, 3 and 8 switched off, the battery (T-profit) of the winner.
