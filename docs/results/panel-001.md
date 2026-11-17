# panel-001: the panel study on the full artifact, flat split (exploratory, rule C5)

Job `docs/jobs/panel-001.json` (entry `panel`): arms `per_coin` (the baseline arm), `relative`, `relative_rank_loss`,
`relative_rank_loss_center`, seeds 0 and 1, the 100 longest-listed coins as whole timestamps, GRU, lr 3e-4, selection
`val_rank_ic_reg`, patience 6. Run 2026-11-16 on a Colab L4 from branch `ccr-841b9809-6hngwg`, one process per seed,
next to other jobs (`docs/handoff/2026-11-16-panel-session.md` section 3; the commit is not recorded on Drive).
Metrics and accept rule were written first: `docs/audit/2026-11-16-panel/PREREGISTERED.md`.

How this note was made: a reading agent copied the numbers from the report files on Drive
(`MyDrive/nig_timenet_runs/panel-001/train/<arm>_s<seed>/`, file ids in the tables); the lead session checked the row
`rank_loss_center s1` against its `panel_report.md`. The study-level files (`study.json` / `study.md`) were written to
the runtime's local disk and are lost.

Written before rule C7: this note is a first reading of the report files. Its failure reading (the error map by
groups of rows, the failed samples, bias, selective use, the named cause, the next hypotheses) is owed and is done
on the pooled walk-forward days (ROADMAP N1).

## Verdict

- **No arm is profitable after costs and no arm passes the accept rule.** Every TEST mean day net is negative on
  metric 1 and metric 2 in all 8 runs (metric 2: the whole CI below 0); no run beats every baseline; the only TEST
  wins are against `momentum_1d`.
- **The relative inputs and the rank term raise the relative rank IC beyond the seed spread**: per_coin 0.060 / 0.068
  < relative 0.086 / 0.096 < relative_rank_loss 0.107 / 0.108; `cross_center` adds nothing to the IC (0.108 / 0.109)
  but removes the level bet from the sign (metric 1 63.6 % on both seeds against 45.5 .. 55.6 % without it).
- **The gain in IC does not reach the days**: metric 2 gross is +0.06 .. +0.08 % per day against 0.16 % of cost
  (under half of the cost; bottleneck B-074).
- The battery's "tradeable=True" is one post-hoc cell of a grid, not profit (section 2e and oddity 4).

## Results (flat time split)

Cost 0.0008 per side; CI 95% from 5-day blocks. TEST: 49,791 rows, 85 coins, 588 timestamps, 99 days. VAL: 54,717 rows, 88 coins, 633 timestamps, 107 days. "net" = mean day net of cost, in %. Source files: `panel_report.md` ids below (numbers also cross-checked against `arm.json` for the model rows, they match). Diff columns are "model minus baseline, winning days, pp [CI]" as printed in the report.

Best baseline by winning days: TEST metric 1 = last_bar_asymmetry 62.6 % [53.3, 73.5] (net -0.1202 %); TEST metric 2 = reversal_1d 65.7 % [58.6, 73.5] (net -0.1081 %). VAL metric 1 = last_bar_asymmetry 57.9 % [49.3, 66.4] (net -0.0992 %); VAL metric 2 = last_bar_asymmetry 66.4 % [58.7, 74.2] (net -0.0993 %). Best baseline by net, metric 1 TEST = always_long -0.0242 % [-0.2342, +0.1969]; VAL = last_bar_asymmetry -0.0992 %. (Baseline numbers are the same in all 8 reports up to small CI differences for the paired rows.)

### 2a. TEST

| run (panel_report.md id) | M1 win gross [CI] | M1 net [CI] | M2 win gross [CI] | M2 net [CI] | rank IC | M1 diff vs last_bar_asym | M2 diff vs reversal_1d | WIN marks printed |
|---|---|---|---|---|---|---|---|---|
| per_coin s0 (11XrFVkJi_b0P3DaBWJZjNvnqph5W1RWL) | 55.6 [44.0, 69.6] | -0.1472 [-0.2634, -0.0088] | 59.6 [50.0, 69.6] | -0.0965 [-0.1379, -0.0437] | +0.0684 | -7.1 [-16.8, +3.0] | -6.1 [-17.0, +5.0] | M2 vs momentum_1d +25.3 [+12.0, +40.5] |
| per_coin s1 (1Xm59vlU_E7LyoNTH0QBD0Dt-lgpy9lOt) | 61.6 [51.7, 71.0] | -0.1034 [-0.1828, -0.0220] | 62.6 [53.0, 73.9] | -0.0840 [-0.1321, -0.0254] | +0.0604 | -1.0 [-10.1, +8.7] | -3.0 [-14.0, +8.2] | M2 vs momentum_1d +28.3 [+15.0, +42.7] |
| relative s0 (1Zsttu-wi-aVEHkVdgZ4q95jffKSrizqE) | 50.5 [38.7, 63.5] | -0.1765 [-0.3372, +0.0087] | 62.6 [54.3, 71.9] | -0.0975 [-0.1473, -0.0349] | +0.0955 | -12.1 [-24.8, +0.0] | -3.0 [-11.1, +5.2] | M2 vs momentum_1d +28.3 [+15.0, +42.4] |
| relative s1 (1LlMA95XVsHAzagzc7L2LI90Gkv8Ktydz) | 45.5 [34.8, 55.6] | -0.2471 [-0.4300, -0.0805] | 58.6 [50.7, 66.2] | -0.0864 [-0.1329, -0.0353] | +0.0861 | -17.2 [-25.4, -8.1] | -7.1 [-16.2, +2.0] | M2 vs momentum_1d +24.2 [+12.5, +36.9] |
| relative_rank_loss s0 (1__vBrje95h8YH7WxFEeF6MabZSKTMq5e) | 55.6 [44.9, 67.5] | -0.1794 [-0.3190, -0.0238] | 64.6 [54.5, 74.6] | -0.0949 [-0.1391, -0.0397] | +0.1067 | -7.1 [-17.2, +2.0] | -1.0 [-12.2, +10.1] | M2 vs momentum_1d +30.3 [+17.1, +45.0] |
| relative_rank_loss s1 (1bniis8xAMAyFYG9Lo2LnkVXtgfU8CcaC) | 45.5 [35.4, 56.1] | -0.2285 [-0.4034, -0.0788] | 63.6 [55.0, 72.8] | -0.0962 [-0.1453, -0.0398] | +0.1078 | -17.2 [-27.0, -8.0] | -2.0 [-12.0, +7.1] | M2 vs momentum_1d +29.3 [+16.6, +43.4] |
| rank_loss_center s0 (1LHgHjEQCpFFSRTUDn81ozKIDS3Jvb5qG) | 63.6 [56.3, 70.0] | -0.1141 [-0.1418, -0.0889] | 64.6 [56.6, 73.0] | -0.0997 [-0.1441, -0.0488] | +0.1075 | +1.0 [-13.4, +14.0] | -1.0 [-11.0, +8.2] | M2 vs momentum_1d +30.3 [+18.3, +43.4] |
| rank_loss_center s1 (1VwqBs9uWNX4lVLtYgRWnWDqIdw8RZ2i9) | 63.6 [54.3, 72.6] | -0.1108 [-0.1385, -0.0755] | 67.7 [59.1, 77.0] | -0.0829 [-0.1271, -0.0325] | +0.1092 | +1.0 [-15.3, +16.4] | +2.0 [-6.1, +11.1] | M2 vs momentum_1d +33.3 [+20.2, +48.5] |

No WIN mark on metric 1 for any TEST run; no WIN against reversal_1d or last_bar_asymmetry for any TEST run. Mean day gross (not net) for reference, M1 / M2: pc s0 +0.0128 / +0.0635; pc s1 +0.0566 / +0.0760; rel s0 -0.0165 / +0.0625; rel s1 -0.0871 / +0.0736; rrl s0 -0.0194 / +0.0651; rrl s1 -0.0685 / +0.0638; center s0 +0.0459 / +0.0603; center s1 +0.0492 / +0.0771.

### 2b. VAL

| run | M1 win gross [CI] | M1 net [CI] | M2 win gross [CI] | M2 net [CI] | rank IC | M1 diff vs last_bar_asym | M2 diff vs last_bar_asym | WIN marks printed |
|---|---|---|---|---|---|---|---|---|
| per_coin s0 | 60.7 [52.3, 70.1] | -0.0856 [-0.1594, -0.0043] | 60.7 [50.9, 71.0] | -0.1238 [-0.1799, -0.0689] | +0.0749 | +2.8 [-7.7, +15.0] | -5.6 [-19.6, +9.7] | M1 vs yesterday_market_sign +21.5 [+10.0, +36.6]; M2 vs reversal_1d +8.4 [+0.4, +15.5] |
| per_coin s1 | 68.2 [60.8, 75.6] | -0.0613 [-0.1430, +0.0271] | 62.6 [53.7, 71.9] | -0.1170 [-0.1713, -0.0664] | +0.0732 | +10.3 [+2.8, +20.6] | -3.7 [-15.5, +9.3] | M1 vs yesterday_market_sign +29.0 [+15.2, +41.1]; M2 vs reversal_1d +10.3 [+2.0, +20.3] |
| relative s0 | 57.9 [49.3, 69.2] | -0.0774 [-0.1883, +0.0458] | 56.1 [44.9, 67.3] | -0.1187 [-0.1894, -0.0431] | +0.0961 | +0.0 [-13.1, +14.2] | -10.3 [-26.7, +6.6] | M1 vs yesterday_market_sign +18.7 [+2.8, +35.7] |
| relative s1 | 52.3 [44.9, 58.8] | -0.1401 [-0.2618, -0.0307] | 59.8 [49.5, 72.4] | -0.1119 [-0.1729, -0.0423] | +0.0881 | -5.6 [-15.5, +5.9] | -6.5 [-20.7, +9.8] | M2 vs reversal_1d +7.5 [+0.4, +15.0] |
| relative_rank_loss s0 | 56.1 [47.7, 66.8] | -0.0864 [-0.1642, +0.0049] | 59.8 [49.1, 71.1] | -0.1172 [-0.1844, -0.0515] | +0.1085 | -1.9 [-12.1, +10.0] | -6.5 [-23.1, +9.6] | M1 vs yesterday_market_sign +16.8 [+0.9, +34.5] |
| relative_rank_loss s1 | 54.2 [46.4, 62.6] | -0.1195 [-0.2116, -0.0339] | 58.9 [49.0, 69.7] | -0.1158 [-0.1838, -0.0453] | +0.1079 | -3.7 [-12.7, +6.0] | -7.5 [-21.7, +8.9] | M1 vs yesterday_market_sign +15.0 [+2.8, +27.2] |
| rank_loss_center s0 | 56.1 [46.3, 65.9] | -0.1336 [-0.1803, -0.0909] | 59.8 [48.4, 72.1] | -0.1166 [-0.1850, -0.0472] | +0.1075 | -1.9 [-16.4, +13.6] | -6.5 [-24.4, +12.0] | M1 vs yesterday_market_sign +16.8 [+2.8, +32.3] |
| rank_loss_center s1 | 57.9 [48.8, 67.3] | -0.1360 [-0.1816, -0.0885] | 57.0 [47.7, 68.3] | -0.1120 [-0.1811, -0.0374] | +0.1095 | +0.0 [-15.0, +16.3] | -9.3 [-23.2, +7.6] | M1 vs yesterday_market_sign +18.7 [+6.0, +32.7] |

Unprinted-WIN rows whose CI excludes 0 (see section 6): per_coin s1 VAL M1 vs always_long +15.0 [+5.0, +24.4] and vs last_bar_asymmetry +10.3 [+2.8, +20.6] (the table above shows the latter), relative s1 VAL M1 vs yesterday_market_sign +13.1 [+0.0, +24.9].

### 2c. Direction bias and uncertainty (TEST, thresholds from VAL), from the same panel_report.md files

| run | predicted up (realised 0.493) | hit rate | per-coin share predicted up (range) | per-day corr | timestamps with one sign for every coin | T-sharp close / high / low | coverage 80 % close / high / low |
|---|---|---|---|---|---|---|---|
| per_coin s0 | 0.416 | 0.509 | 0.32 .. 0.51 | -0.60 | 0.01 | 0.000 / 0.021 / 0.134 | 0.897 / 0.885 / 0.936 |
| per_coin s1 | 0.610 | 0.513 | 0.44 .. 0.83 | -0.61 | 0.00 | 0.000 / 0.269 / 0.025 | 0.888 / 0.898 / 0.925 |
| relative s0 | 0.229 | 0.512 | 0.04 .. 0.50 | -0.17 | 0.23 | 0.000 / 0.007 / 0.440 | 0.898 / 0.886 / 0.925 |
| relative s1 | 0.057 | 0.502 | 0.01 .. 0.09 | -0.09 | 0.70 | 0.000 / 0.002 / 0.270 | 0.890 / 0.855 / 0.943 |
| relative_rank_loss s0 | 0.306 | 0.518 | 0.02 .. 0.94 | -0.38 | 0.00 | 0.000 / 0.020 / 0.530 | 0.898 / 0.888 / 0.927 |
| relative_rank_loss s1 | 0.121 | 0.510 | 0.01 .. 0.31 | -0.19 | 0.20 | 0.000 / 0.010 / 0.423 | 0.893 / 0.857 / 0.945 |
| rank_loss_center s0 | 0.594 | 0.518 | 0.02 .. 1.00 | -0.46 | 0.00 | 0.000 / 0.032 / 0.359 | 0.898 / 0.886 / 0.924 |
| rank_loss_center s1 | 0.597 | 0.519 | 0.03 .. 0.99 | -0.18 | 0.00 | 0.000 / 0.018 / 0.108 | 0.890 / 0.854 / 0.940 |

Correlation of per-coin predicted-up share with realised share: pc s0 -0.34, pc s1 +0.03, rel s0 +0.21, rel s1 -0.54, rrl s0 +0.34, rrl s1 +0.19, center s0 +0.24, center s1 +0.22. Confidence-filtered rows (all_rows / top_confidence / top_abs_mu), metric 1 mean day net: pc s0 -0.1472 / -0.0041 / +0.0225; pc s1 -0.1034 / +0.0513 / +0.0302; rel s0 -0.1765 / -0.0460 / -0.0677; rel s1 -0.2471 / -0.1170 / -0.0708; rrl s0 -0.1794 / -0.0756 / -0.1130; rrl s1 -0.2285 / -0.0552 / -0.1194; center s0 -0.1141 / +0.0120 / -0.0156; center s1 -0.1108 / +0.0059 / -0.0175. The top_confidence rows keep 0.21 to 0.29 of rows. Positive mean day net (CI not given in the report): top_confidence in pc s1, center s0, center s1; top_abs_mu in pc s0, pc s1. All other cells negative.

### 2d. Selection and test-time rank IC per head (arm.json, `test_rank_ic`; best_value = val_rank_ic_reg)

| run (arm.json id) | best_value | test_rank_ic high / low / close |
|---|---|---|
| per_coin s0 (1j5umfzzF_hahO5gA2YqBOFfNKdkiFzrs) | 0.27840459198656015 | 0.30413238919150243 / 0.3989289045995473 / 0.06842341609787923 |
| per_coin s1 (1g94ovpNaCEj3censuEVca0ApxgO1cwgi) | 0.27242491172583894 | 0.30203699543933965 / 0.3935926189792327 / 0.06035140602455034 |
| relative s0 (1vdQa6cOGkk2kGSqejQvHwgzwq5EEtdnR) | 0.2940497709538186 | 0.3161991464476138 / 0.41584657049698515 / 0.09546992687902721 |
| relative s1 (1q5iMRIfDgjwOT3mRaDc_gtO3RESLktZP) | 0.29344298665274093 | 0.3156839619078958 / 0.4153493841760851 / 0.08609220790445006 |
| relative_rank_loss s0 (1I3jwT3nzJx0kIp8pesrDJoVnDXenhWQI) | 0.30129199764961406 | 0.31894298480037997 / 0.4230074288886955 / 0.10668485141173377 |
| relative_rank_loss s1 (1MmtO90V5ZVT1lYIGe84dKHk-qhCjPNiH) | 0.30227818695191666 | 0.3188217000598561 / 0.42331328826695613 / 0.10775778684905582 |
| rank_loss_center s0 (1ZyFA1IUP_ixGPCnP6aBmjYTmN34CSLrL) | 0.30138845144194226 | 0.3194998248538337 / 0.42383870069057655 / 0.10752323124956356 |
| rank_loss_center s1 (1nJA0DnsTP3yVEQlk1SbKriwcQpx-2Pkz) | 0.30258558021735543 | 0.3198448299527743 / 0.42320025129150657 / 0.10916713026196492 |

All 8 arm.json: `beats_every_baseline` false for both absolute and neutral; `battery_tradeable` true; `selection_metric` "val_rank_ic_reg"; `finished` true.

### 2e. Battery verdict lines (battery_report.txt; quoted verbatim, same Arabic line in all 8: "الحكم: قابل للتداول نظرياً (Sharpe>0.10)")

| run (battery_report.txt id) | BEST line | weekly line | T-PROFIT "model (close p_dir)" line (TEST, after cost) | last line |
|---|---|---|---|---|
| per_coin s0 (119pDC1-dN6Bp1ocd6MsSAOpKvsKOK7yK) | BEST: LONG top5% Sharpe=+0.130 PF=1.57 n=1,810 avg=+0.4576% | 20 of 37 weeks positive | model (close p_dir) n= 49,312 avg=-0.0734% sharpe=-0.027 CI98.8%=[-0.1991%, +0.0422%] | tradeable=True marginal=False |
| per_coin s1 (1te8Y4n51lL_y3lxyeLHkre5AUp7sTs9c) | BEST: LONG top5% Sharpe=+0.116 PF=1.47 n=653 avg=+0.6005% | 22 of 39 weeks positive | avg=-0.1961% sharpe=-0.072 CI98.8%=[-0.3574%, -0.0389%] | tradeable=True marginal=False |
| relative s0 (1TC2O-Im2NV5TchY4YL_i6sxsTcIk4-ys) | BEST: LONG top5% Sharpe=+0.152 PF=1.47 n=578 avg=+0.4044% | 13 of 32 weeks positive | avg=-0.1898% sharpe=-0.070 CI98.8%=[-0.3320%, -0.0250%] | tradeable=True marginal=False |
| relative s1 (1yexUaJK8xyj1RSb4TlZUvnzQ3jPoVG9J) | BEST: LONG top5% Sharpe=+0.197 PF=1.65 n=516 avg=+0.5365% | 14 of 40 weeks positive | avg=-0.1559% sharpe=-0.057 CI98.8%=[-0.2893%, +0.0135%] | tradeable=True marginal=False |
| relative_rank_loss s0 (1mqtQ5geUeqqcb6Z9diZEbobJ-Xt11odQ) | BEST: LONG top5% Sharpe=+0.219 PF=1.77 n=599 avg=+0.5133% | 16 of 39 weeks positive | avg=-0.1730% sharpe=-0.064 CI98.8%=[-0.3043%, -0.0140%] | tradeable=True marginal=False |
| relative_rank_loss s1 (18oSVR6SK32MdGxaeKmzKkmwnGhwLiycM) | BEST: LONG top5% Sharpe=+0.244 PF=1.92 n=511 avg=+0.6347% | 16 of 38 weeks positive | avg=-0.1562% sharpe=-0.058 CI98.8%=[-0.2831%, +0.0015%] | tradeable=True marginal=False |
| rank_loss_center s0 (1tMMmmggo4muq85PbUwcNT1cHMb4Siy7C) | BEST: LONG top10% Sharpe=+0.103 PF=1.37 n=1,276 avg=+0.2048% | 17 of 46 weeks positive | avg=-0.1621% sharpe=-0.060 CI98.8%=[-0.2822%, -0.0195%] | tradeable=True marginal=False |
| rank_loss_center s1 (1hZjXF6XDpNUCDZggr4mqQ_wWCf8T4aA4) | BEST: LONG top20% Sharpe=+0.141 PF=1.51 n=2,130 avg=+0.3099% | 15 of 39 weeks positive | avg=-0.1423% sharpe=-0.052 CI98.8%=[-0.2656%, +0.0079%] | tradeable=True marginal=False |

Same baseline rows in every battery report: always up avg=-0.0281% [-0.2484%, +0.2078%]; always down avg=-0.2919% [-0.5278%, -0.0716%]. Shuffled direction avg between -0.1029% (pc s0) and -0.2340%. Other battery lines worth copying: SANITY TEST IC(mu, y_ret): +0.0501 / +0.0460 / +0.0622 / +0.0733 / +0.0652 / +0.0656 / +0.0693 / +0.0721 (order of the table); anti-cowardice mean |mu| / mean |y_ret| = 0.043 / 0.039 / 0.057 / 0.092 / 0.062 / 0.095 / 0.029 / 0.026; battery "MARKET-NEUTRAL PORTFOLIO (TEST) long top20% / short bottom20% (VAL choice)", n=588, avg% (win%, PF) in table order: -0.2118 (40.65, 0.60) / -0.1777 (42.52, 0.66) / -0.2125 (44.39, 0.62) / -0.1903 (44.73, 0.64) / -0.2005 (44.56, 0.64) / -0.2001 (42.86, 0.63) / -0.2051 (44.05, 0.63) / -0.1822 (43.71, 0.66); 4 sides of cost per period, all negative.

### 2f. Reading (5 lines)
1. Best arm by metric (TEST): rank IC, relative_rank_loss and relative_rank_loss_center (+0.1067 to +0.1092) over relative (+0.0861, +0.0955) over per_coin (+0.0604, +0.0684); metric 2 winning days, center s1 67.7 % and center s0 / rrl s0 64.6 %; metric 1 winning days, center 63.6 % in both seeds and per_coin s1 61.6 %; VAL metric 1 best is per_coin s1 68.2 %.
2. Arm gap vs seed gap: IC arm gaps (+0.026 per_coin to relative, +0.017 relative to rrl) exceed seed gaps (<= 0.0094, relative); rrl vs center (+0.0011 on average) is inside the seed spread (0.0011 to 0.0017). Metric 1 winning days: seed spread within an arm is 6.0 pp (per_coin), 5.0 (relative), 10.1 (rrl) and 0.0 (center), so only the center advantage in absolute direction is larger than its own seed spread, and its CI vs last_bar_asymmetry (+1.0 [-13.4, +14.0]) includes 0.
3. Net profit: none. Every metric-1 and metric-2 TEST mean day net is negative; the metric-2 CIs are entirely below 0 in all 8; the metric-1 CI upper bound is above 0 only for relative s0 (+0.0087). VAL net is negative in all 16 cells, with CI upper bound above 0 in per_coin s1 M1 (+0.0271), relative s0 M1 (+0.0458), rrl s0 M1 (+0.0049). No run beats every baseline (arm.json `beats_every_baseline` false x16); the only WIN marks on TEST are against momentum_1d.
4. TEST vs VAL: agree that nothing is net profitable, that rank IC is positive and orders per_coin < relative < rrl ~ center, that metric 2 net is below -0.08 %. They disagree on metric 1: per_coin is best on VAL (60.7, 68.2) but center on TEST (63.6, 63.6); relative and rrl are 45.5 to 55.6 on TEST but 52.3 to 58.0 on VAL; the only WIN marks differ (TEST vs momentum_1d, VAL vs yesterday_market_sign / reversal_1d).
5. Battery marks every arm "tradeable=True" on a single post-hoc best cell of a 6-cell grid (a different cell per arm), while the full-set after-cost line (T-PROFIT model) has CI below or straddling 0 and avg below the always-up row in all 8; do not read the battery verdict as profit.

## Config facts

Read from: `arm.json` (8 runs, ids in 2d), `baseline_summary.json` of per_coin seed 0 ONLY (id 1tyFsxVlIrNYtiq9CaC33aj_PBFk0no5b; the other 7 baseline_summary.json are 34.9 to 56.4 KB each, not opened; they probably repeat the epoch history and add arm-specific settings), `test_report.json` of per_coin seed 0 (id 1NGN3YJlKuR76ISZQvOM99dY_1qX9lsZG), and best_meta.json files.

| item | value | source |
|---|---|---|
| commit sha / repo_ref / code_sha | not on Drive (no such key in arm.json, baseline_summary.json (per_coin s0), test_report.json (per_coin s0)) | |
| artifact name | not on Drive | |
| data_fingerprint | c572b1b36960f7b026d969af18f83f8d502b2fcef3c66eeeec146fdaf590eea2 (identical in all 8 arm.json) | arm.json x8 |
| identity_fingerprint | 756b2f06e9f0bbf1118e4b03471b9f0dc89dfbc5c09a108aeea5c1546e48c6cf (per_coin s0 only; others not read) | baseline_summary.json 1tyFsxVlIrNYtiq9CaC33aj_PBFk0no5b |
| device / GPU name | not on Drive. Only `peak_memory_mb`: per_coin s0 183.66943359375, s1 184.45751953125; relative s0 196.88330078125, s1 196.8349609375; rrl s0 196.8984375, s1 196.8505859375; center s0 196.8984375, s1 196.8505859375 | arm.json |
| rows per split (windows) | train 1113463, val 54717, test 49791 | arm.json |
| coins per split | VAL 88, TEST 85 (train: not on Drive); TEST 588 timestamps / 99 days; VAL 633 timestamps / 107 days | panel_report.md |
| split dates | train_end 2026-02-24 04:00:00+00:00; val_end 2026-06-15 08:00:00+00:00; battery integrity line: VAL max 2026-06-15 00:00:00 / TEST min 2026-06-20 20:00:00, overlap 0 | baseline_summary.json (per_coin s0); battery_report.txt |
| batch size | 512 (per_coin s0) | baseline_summary.json |
| lr | 0.0003 in every epoch of the history (per_coin s0, epochs 1 to 7) | baseline_summary.json history |
| epochs | per_coin s0: requested 25, done 7, stopped_early true, best epoch 1; all 8 runs: epochs done and best epoch in section 1a | baseline_summary.json; arm.json |
| selection metric | val_rank_ic_reg | arm.json x8 |
| seconds | per epoch: per_coin s0 median 153.62026019800032, step_seconds_per_batch 0.07063000468873579; per-run total seconds in section 1a (1081.995 to 2501.420) | baseline_summary.json; arm.json |
| targets / input | targets [high, low, close]; input_key X_4h; reg_target_mode return; label_smoothing 0.1; max_rows null | baseline_summary.json (per_coin s0) |
| model overrides (per_coin s0) | d_model 64, num_layers 2, head_hidden 64, class_head_hidden 32, dropout 0.25, input_clip 4.0, stats_mode symlog, input_noise_std 0.1, feature_dropout 0.1, linear_path_l2 0.001, level_passthrough true, level_norm batch, encoder gru, bn_warmup true, order_detach_close true, nig_scale_fit train_std, nig_trunk_grad 0.3, input_scaler train_std | baseline_summary.json |
| cost | 0.0008 per side (battery: 0.0800% per side, 0.1600% per round trip; market-neutral: 4 sides) | panel_report.md; battery_report.txt |
| run_dir (per_coin s0) | /content/drive/MyDrive/nig_timenet_runs/panel-001/train/per_coin_s0/ckpt | baseline_summary.json |
| rolling jobs | rows, coins, batch size, lr, epochs, fingerprint: not on Drive (only best_meta.json with best value, best epoch, metric) | |

Timing of the rolling jobs from file times only: rolling-base-001 fold 0 seed 0 checkpoint written 16:35:33, last file 18:06:03; rolling-001 fold 0 seed 0 first best_meta 16:36:09, last 18:51:21. Both jobs ended with checkpoint files only; whether an evaluation step ran is not determinable from Drive.

## Open questions and oddities (from the reading)

1. Rolling jobs have no evaluation output on Drive (see 1b/1c/3). Either the evaluation step was not run or its outputs were written elsewhere (or were not kept; the root folder shows modifiedTime 19:28:43 and the same-named shared-drive folder 17ciUfduPTmGToNdpC2IMYVECkBDFE-Li 19:30:41 is empty).
2. The only rolling number is the selection metric, and its value (0.2460 to 0.3117) is in the range of the high/low-head rank IC of panel-001 test (0.30 to 0.42) and well above the close-head IC (0.06 to 0.11); the files do not say which heads it averages, so do not compare it to the panel rank IC.
3. WIN mark rule unclear. Printed WIN appears next to one baseline row only per table in most reports, yet other rows have a CI above 0 without a WIN: per_coin s1 VAL M1 vs always_long +15.0 pp [+5.0, +24.4] and vs last_bar_asymmetry +10.3 [+2.8, +20.6] (WIN printed only on yesterday_market_sign +29.0 [+15.2, +41.1]); relative s1 VAL M1 vs yesterday_market_sign +13.1 [+0.0, +24.9] has no WIN, rrl s1 VAL M1 vs yesterday_market_sign +15.0 [+2.8, +27.2] has WIN; rrl s0 VAL M1 vs yesterday_market_sign +16.8 [+0.9, +34.5] has WIN. The rule (maybe a multiple-comparison correction, maybe a different threshold) is not in the file.
4. Battery verdict vs after-cost profit. All 8 battery reports print "tradeable=True marginal=False" ("الحكم: قابل للتداول نظرياً (Sharpe>0.10)"), based on the best single grid cell (LONG top5% in 6 runs, top10% in center s0, top20% in center s1), while the same file's T-PROFIT "model (close p_dir)" line has avg -0.0734% to -0.1961% and its CI never lies above 0 (lower bound always below 0, upper bound above 0 in per_coin s0, relative s1, rrl s1, center s1); the "always up" row (-0.0281%) is higher than the model avg in all 8. The file itself says "Profit = model beats every baseline row AND its CI excludes 0".
5. Direction bias differs by source. Battery SANITY P(p_dir>0.5) TEST: per_coin s0 0.735, s1 0.265, relative s0 0.234, s1 0.209, rrl s0 0.243, s1 0.208, center s0 0.259, s1 0.216; panel_report "predicted up" TEST: 0.416, 0.610, 0.229, 0.057, 0.306, 0.121, 0.594, 0.597. For per_coin and the center arm the two disagree strongly (e.g. center s0 0.259 vs 0.594; per_coin s1 0.265 vs 0.610) and per_coin seeds flip sign in both sources. The battery grid agrees with the flip (per_coin s0 LONG top5% n=1,810, SHORT n=655; s1 LONG n=653, SHORT n=1,812). Different definitions are likely (close class head vs sign of mu) but the files do not say.
6. Equal winning-day share in different arms: relative s1 and relative_rank_loss s1 both have metric 1 TEST 45.5 % (45 of 99 days) with different gross days (-0.0871 vs -0.0685) and different CIs, so not duplicate files; coincidence of a day count.
7. In the center arm the "model" row and the "model, sign of (mean - timestamp mean) (information)" row are equal in TEST (63.6 / 63.6) and VAL (56.1 / 56.1 for s0; 57.9 / 57.9 for s1); in other arms they differ. Expected if the center head removes the timestamp mean, but it means center's metric 1 is the relative signal only.
8. Battery "WEEKLY STABILITY" lines are labelled "week 2026-06-24", "week 2026-06-25", ... on consecutive dates, 32 to 46 "weeks" over a 99-day TEST; the unit is not weeks. Also the battery text says "With the czar loss ..." (anti-cowardice header), unrelated to the arm names here.
9. Metric 2 winning days gross ~58.6 to 67.7 % (panel report) vs battery market-neutral win% 40.65 to 44.73 (4 sides of cost); different cost basis, not a contradiction, but the two sections are easy to confuse.
10. arm.json `test_rank_ic.close` equals the panel "cross-sectional rank IC of the relative move" to 4 digits (e.g. 0.06842341609787923 vs +0.0684) and `relative_rank_ic` in arm.json is 0.06842399488713873 (differs in the 6th digit): two slightly different computations of the same number.
11. per_coin best epoch is 1 or 2 (best_value 0.2784 / 0.2724) and training ran to epoch 7 / 8 (stopped_early true for s0, the only baseline_summary read); in the rolling base job best epochs are 1 to 8 with no pattern.
12. Two-session overlap: file times show panel-001 and the rolling jobs writing at the same time (e.g. rolling-base fold 0 at 16:29:31 and panel-001 per_coin_s1 at 16:30:21), so the jobs overlapped in time (separate sessions or processes); whether epoch seconds are comparable across jobs is unknown.
13. Not opened: 7 of 8 baseline_summary.json, all panel_report.json, battery_profit.json, parquet frames, all `.pt`, smoke reports. The per-arm loss/regularisation settings (the thing that distinguishes relative / rank_loss / center) are likely in those baseline_summary files and I did not read them.
