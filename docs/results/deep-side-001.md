# Deep side — is "the side where price goes deeper" predicted? (`baseline-002-s0`, seed 0)

Recorded 2026-11-09. Exploratory reading (rule C5), read-only: no training, no default changed. Owner task of
2026-11-08. Numbers are never edited afterwards; a correction goes in a new file that names this one.

## 0. Definitions and method

- **Target.** `deep = up_real + down_real`, with `up_real = log(future_high_max / last_close)` and
  `down_real = log(future_low_min / last_close)` (negative), over the 2-bar horizon. The binary direction is
  `sign(deep)`: "down" when the drop is deeper than the rise, whatever the close does.
- **Prediction.** `deep_pred = mu_high + mu_low`, decoded with the artifact's `logret_close` codec (B-051 fixed,
  D-017): `log1p(pred_high) + log1p(pred_low)`, which equals `mu_high + mu_low` in label space.
- **Rows.** The whole VAL (239,297) and TEST (239,772) splits of the artifact `c572b1b36960`, run restored with
  `workflow.run.restore` (best epoch 1).
- **Measures.** Rank IC is the Spearman of `deep_pred` vs `deep` per timestamp (cross-section), then the mean. Its
  95 % CI resamples whole days (500 draws). The AUC is computed on `deep > 0`, also with a day-block CI.
  "model − it" is the paired per-timestamp IC difference, with a day-block CI.
- **Baselines** (same rows):
  - last-bar asymmetry: `log(H/C) + log(L/C)` of the last input bar, the last bar's own "deep" around its close;
  - close head mu (`log1p(mu_close)`) and close head p_dir;
  - last-bar close return;
  - ATR-14 of the window;
  - random scores;
  - the model's scores shuffled.

  Each baseline gets the sign of its VAL IC. The sign is fixed on VAL and applied unchanged to TEST, so last-bar
  return and ATR-14 enter as −1 × value.
- **Thresholds.** Volatility bucket edges are the VAL quintiles of ATR-14. The near-tie margin is the VAL 20 %
  quantile of |deep|.
- **Tool.** `eval/deep_side.py` and `workflow/deep_side_run.py`. They write `deep_side_report.txt` and
  `deep_side.json` (local only, `/home/user/work/t5r_data/deep_side/`, about 3.5 minutes on CPU).
  Rule for "evidence": on BOTH splits, the model's IC CI is above 0, its AUC CI is above 0.5, and the paired
  (model − baseline) CI is above 0 for every baseline.

## 1. Label clip (B-054): `deep` is a sum of two labels clipped at |log| = 1

| split | rows | high clipped | low clipped | high or low | both | close clipped |
|---|---|---|---|---|---|---|
| VAL | 239,297 | 12 | 37 | 49 (0.02 %) | 0 | 22 |
| TEST | 239,772 | 11 | 16 | 27 (0.01 %) | 0 | 17 |

The clipped rows are too few to move a rank IC. Every deep value of a clipped row is a lower bound of its true size.

## 2. Skill and baselines (items 1-2)

| score | VAL rank IC [CI95] | VAL AUC [CI95] | VAL model − it | TEST rank IC [CI95] | TEST AUC [CI95] | TEST model − it |
|---|---|---|---|---|---|---|
| **model (mu_high + mu_low)** | **+0.0163 [+0.0086, +0.0242]** | **0.506 [0.492, 0.521]** | | **+0.0146 [+0.0054, +0.0228]** | **0.509 [0.494, 0.524]** | |
| last-bar asymmetry | +0.0287 [+0.0216, +0.0379] | 0.517 [0.504, 0.529] | **−0.0124 [−0.0254, −0.0006]** | +0.0273 [+0.0200, +0.0355] | 0.514 [0.502, 0.527] | **−0.0127 [−0.0261, −0.0004]** |
| close head mu | +0.0166 [+0.0054, +0.0275] | 0.508 [0.495, 0.521] | −0.0004 [−0.0077, +0.0062] | +0.0155 [+0.0053, +0.0255] | 0.503 [0.489, 0.515] | −0.0010 [−0.0097, +0.0072] |
| close head p_dir | +0.0224 [+0.0122, +0.0310] | 0.518 [0.503, 0.531] | −0.0061 [−0.0150, +0.0025] | +0.0133 [+0.0051, +0.0217] | 0.504 [0.491, 0.516] | +0.0013 [−0.0114, +0.0127] |
| last-bar return (−1×) | +0.0239 [+0.0166, +0.0326] | 0.516 [0.502, 0.530] | −0.0076 [−0.0207, +0.0046] | +0.0259 [+0.0188, +0.0330] | 0.511 [0.497, 0.524] | −0.0114 [−0.0253, +0.0018] |
| ATR-14 (−1×) | +0.0143 [+0.0024, +0.0256] | 0.505 [0.491, 0.519] | +0.0020 [−0.0068, +0.0097] | +0.0183 [+0.0072, +0.0290] | 0.500 [0.486, 0.514] | −0.0037 [−0.0134, +0.0060] |
| random | −0.0003 [−0.0041, +0.0035] | 0.499 [0.497, 0.502] | +0.0166 [+0.0073, +0.0254] | +0.0018 [−0.0020, +0.0052] | 0.500 [0.499, 0.502] | +0.0128 [+0.0028, +0.0214] |
| shuffled model | +0.0018 [−0.0021, +0.0058] | 0.501 [0.499, 0.503] | +0.0145 [+0.0065, +0.0225] | +0.0004 [−0.0042, +0.0048] | 0.499 [0.497, 0.502] | +0.0142 [+0.0043, +0.0232] |

**Reading.**
- The model's IC is above 0 on both splits (CI excludes 0) and the sign agrees on VAL and TEST. It does beat random
  and shuffled scores.
- The model's AUC CI contains 0.5 on both splits.
- A trivial baseline does better than the model on BOTH splits: the last bar's own asymmetry (+0.029 / +0.027). Its
  paired CI is below 0 on both splits.
- The model ties the close head's mu and p_dir and ATR-14 (paired CIs contain 0).

## 3. Is it the close direction renamed? (item 3)

| | VAL | TEST |
|---|---|---|
| Spearman(deep, sign of the true close move), pooled / per timestamp | +0.725 / +0.620 | +0.712 / +0.638 |
| same sign of deep and of the close move | 84.0 % | 83.5 % |
| Spearman(deep_pred, close mu), pooled / per timestamp | +0.612 / +0.597 | +0.616 / +0.610 |
| share of rows with deep > 0: true / predicted | 49.2 % / 14.7 % | 50.5 % / 17.5 % |
| model net of close head mu | +0.0031 [−0.0030, +0.0095] | +0.0041 [−0.0062, +0.0131] |
| model net of close head p_dir | +0.0056 [−0.0017, +0.0131] | +0.0094 [−0.0017, +0.0193] |
| model net of last-bar asymmetry | +0.0192 [+0.0114, +0.0268] | +0.0175 [+0.0091, +0.0252] |
| model net of last-bar return | +0.0197 [+0.0120, +0.0273] | +0.0185 [+0.0104, +0.0259] |
| model net of ATR-14 | +0.0083 [+0.0020, +0.0146] | +0.0059 [−0.0028, +0.0135] |
| model net of all five baselines at once | +0.0055 [−0.0005, +0.0116] | +0.0099 [+0.0006, +0.0182] |

**Reading.**
- The target is mostly the close direction under another name: deep and the close move share their sign on 84 % of
  rows.
- The prediction is mostly the close head: its rank correlation with the close mu is 0.61.
- Net of the close mu, nothing is left (CI contains 0 on both splits).
- Net of all baselines at once, the CI contains 0 on VAL and is barely above 0 on TEST. A sign or CI that holds on
  one split only is not evidence.
- The model's side is biased: it calls "down deeper" on 83–85 % of rows, while the truth is about 50 %. This is the
  same low-larger-than-high bias of the mu means as B-063.

## 4. Volatility buckets (item 4; ATR-14, VAL quintile edges 2.05 / 2.60 / 3.31 / 4.63 %)

| bucket (calm -> violent) | VAL n | VAL model IC [CI95] | VAL best baseline | VAL model − best | TEST n | TEST model IC [CI95] | TEST best baseline | TEST model − best |
|---|---|---|---|---|---|---|---|---|
| 0 | 47,860 | −0.0132 [−0.0279, +0.0026] | asymmetry +0.060 | −0.073 [−0.097, −0.048] | 65,498 | −0.0168 [−0.0384, +0.0050] | asymmetry +0.046 | −0.063 [−0.098, −0.033] |
| 1 | 47,859 | −0.0046 [−0.0183, +0.0104] | asymmetry +0.023 | −0.028 [−0.049, −0.005] | 46,539 | −0.0153 [−0.0325, +0.0015] | asymmetry +0.032 | −0.047 [−0.068, −0.025] |
| 2 | 47,859 | +0.0053 [−0.0083, +0.0199] | asymmetry +0.027 | −0.022 [−0.046, +0.005] | 45,608 | +0.0044 [−0.0092, +0.0173] | asymmetry +0.029 | −0.025 [−0.045, −0.005] |
| 3 | 47,859 | +0.0237 [+0.0137, +0.0348] | asymmetry +0.027 | −0.003 [−0.020, +0.012] | 42,443 | +0.0053 [−0.0092, +0.0199] | last-bar return +0.033 | −0.028 [−0.049, −0.009] |
| 4 | 47,860 | +0.0371 [+0.0242, +0.0506] | close mu +0.041 | −0.004 [−0.011, +0.003] | 39,684 | +0.0513 [+0.0389, +0.0645] | close mu +0.061 | −0.010 [−0.018, −0.001] |

**Reading.**
- Inside a volatility bucket, the model ranks deep at about 0 in the three calmest quintiles. The sign is negative in
  the two calmest.
- It is positive only in the most violent quintile, and there the close head's mu does better.
- The model is below the best baseline in every bucket: in 2 / 5 VAL buckets and 5 / 5 TEST buckets the CI is below
  0.
- The pooled IC therefore comes from calm-vs-violent separation and from the violent fifth. That fifth is where the
  close head already ranks it.

## 5. Near ties (item 5): rows with |deep| < 0.689 % dropped (the VAL 20 % quantile)

| | VAL (kept 191,437) | TEST (kept 183,904) |
|---|---|---|
| model IC [CI95] | +0.0269 [+0.0188, +0.0345] | +0.0266 [+0.0171, +0.0350] |
| model AUC [CI95] | 0.508 [0.492, 0.526] | 0.516 [0.496, 0.533] |
| best baseline IC | close head p_dir +0.0236 | ATR-14 (−1×) +0.0273 |
| model − last-bar asymmetry | +0.0042 [−0.0096, +0.0169] | +0.0032 [−0.0100, +0.0154] |
| model − close head mu | +0.0041 [−0.0031, +0.0111] | +0.0010 [−0.0075, +0.0090] |
| model − ATR-14 | +0.0069 [−0.0024, +0.0157] | −0.0007 [−0.0102, +0.0094] |

**Reading.**
- Dropping near ties raises the model's IC from 0.016 to 0.027. It lowers the last-bar asymmetry's IC from 0.029 to
  0.023, so that baseline's edge sits in the small moves.
- Even so, no paired difference has a CI above 0, and the AUC CI still contains 0.5. The near ties do not hide a
  skill.

## 6. Profit after cost (item 6)

**Not reached.** The spec runs it only if the IC beats every baseline on both splits, and it does not (sections 2-5).
Trading a side whose ranking is worse than the last bar's own asymmetry would only re-test the strategy book's close
direction rule. That rule already loses after cost on both splits (`baseline-001.md` section 4).

## 7. Verdict

| question | answer | evidence |
|---|---|---|
| does the model predict the deeper side above every baseline, CI excluding 0 on both splits? | **no** | IC +0.016 / +0.015 > 0, but AUC CI contains 0.5; the last-bar asymmetry beats it on both splits (paired CI < 0); nothing left net of the close mu |
| is it the close direction renamed? | **yes, mostly** | deep and the close move share their sign on 84 % of rows; deep_pred vs close mu 0.61; net of close mu ≈ 0 |
| is the skill calm-vs-violent separation? | **mostly** | within volatility quintiles the IC is ≈ 0 or negative except in the most violent fifth, where the close mu is better |
| profit after cost | not reached | |

The idea is closed for this model: `docs/bottlenecks.csv` B-065. It may be reopened if a model's high / low heads beat
the HAR baselines on the move SIZE (rule C1, `baseline-001.md` section 3); until then their sum has no edge of its
own. Not verified:
- other seeds (C4: one seed is a look);
- a deep target trained directly as a head;
- other horizons;
- other cost levels.
