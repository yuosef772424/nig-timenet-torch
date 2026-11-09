# E1 fixes — where the mu scale and the close signal are lost, and the switches that fix it (FX1)

Recorded 2026-11-09. Exploratory (rule C5): CPU only, the synthetic E1 environment (D-018 generator, 64,000 windows,
32 coins; the true conditional mean of every label is known), short runs (3 epochs = 300 steps, and 10 epochs).
No default changed (D-022). Numbers are never edited afterwards; a correction goes in a new file that names this one.
Every row: `docs/results/e1-fixes-001-arms.csv` (round, arm, epochs, seed, per head mean ratio / slope / corr /
T-sharp, blocking gate fails). Tool: `python -m nig_timenet_torch.workflow.e1_arms` (arms, gate rows, mu calibration,
head readout gap); batch 512, test split.

## 0. Measures

- `mean_ratio` = mean mu / mean oracle mean (ideal 1). `slope` = regression slope of the oracle mean ON mu (ideal 1;
  > 1 = mu compressed towards its average). `corr` = Pearson(mu, oracle mean). T-sharp: gate C `t_sharp_vs_oracle`.
- Trivial reference (rule C1): a constant mu has corr 0 and no slope; the oracle itself has ratio 1, slope 1, corr 1.
- Arms judged on 3 seeds with seed-paired differences (rule C4); single-seed arms are "looks".

## 1. Diagnosis (baseline = default model with the anti-memorization preset, 3 epochs, seed 0)

| probe (no retraining unless said) | high slope / ratio / corr | low slope / ratio / corr |
|---|---|---|
| model mu, test | 1.51 / 0.55 / 0.92 | 1.66 / 0.81 / 0.92 |
| model mu, train rows | 1.55 / 0.55 / 0.92 | 1.69 / 0.81 / 0.92 |
| least-squares readout of the LABEL from the head's hidden layer (eval mode) | 1.00 / 1.00 / 0.94 | 0.95 / 1.00 / 0.94 |
| same readout fitted on TRAIN-MODE features (noise, dropout, BN batch stats on) | 1.91 / 0.88 / 0.91 | 1.96 / 0.89 / 0.91 |
| ... only BatchNorm in train mode | 1.48 / 0.82 | 1.64 / 0.76 |
| ... only trunk_drop / only the head dropout | 1.12 / 1.15 | 1.17 / 1.27 |
| ... only input noise + feature dropout / only GRU dropouts | 1.06 / 1.04 | 1.05 / 1.01 |
| BatchNorm moving statistics set to the train population | **0.91 / 0.71 / 0.97** | **0.93 / 0.99 / 0.97** |
| loss-only probe: NIG head on the PERFECT features (oracle mean, sd), same loss, 300 steps | 1.04 / 0.91 / 0.99 | 0.97 / 0.84 / 1.00 |

Signal trace (`model.taps.signal_trace`, ridge R^2 of the oracle mean): raw input 0.97, `level_bn` 0.97,
`final_norm` 0.95, `readout_fc` 0.89, trunk / head hidden 0.87 (high; low alike). `instance_norm` alone 0.49 and
`stats_bn` 0.43: the per-window normalisation erases the level, and the level passthrough path restores it — no layer
of the default path erases the signal; the information reaches the heads (R^2 0.87, readout slope 1).

**Cause 1 (verified): BatchNorm moving statistics.** `level_bn` / `stats_bn` (Keras momentum 0.99, init mean 0 /
variance 1) after 300 updates still hold 0.99^300 = 5 % of their init: moving variance 0.042 for a level feature whose
population variance is 2e-4, so eval divides the level path by sqrt(0.042 + 1e-3) = 0.21 instead of 0.035 (6x
smaller). Training is unaffected (batch statistics). Fix: `bn_warmup` — the trained weights are identical (training
never reads the moving stats) and the eval numbers equal the population probe exactly (bnw arm, table 2).

**Cause 2 (verified): ordered means couple close to high / low.** `mu_high = mu_c + softplus(raw_h)`,
`mu_low = mu_c - softplus(raw_l)`: the high and low losses push mu_close. The offsets live in softplus's exponential
tail at label scale 0.02 (raw_h mean -4.3, slope sigmoid(-4.3) = 0.013), so mu_c is the fast path. Close corr with its
true mean -0.19 (seed 0). Fix: `order_detach_close` (offsets added to mu_c.detach()); `enforce_order: false` also
recovers it (+0.59) but loses the order guarantee.

**Cause 3 (verified, residual after cause 1): dropout regression dilution.** At 10 epochs (BN converged) the
train-mode readout probe gives 1.32 / 1.31, of which trunk_drop 1.12 / 1.17 and the head dropout 1.14 / 1.13 (BN 1.01 /
0.98): a readout learned on dropped-out features is shrunk at eval. Switch: `readout_dropout` (trunk_drop + NIG
head dropout rate; class heads keep `dropout`).

**Cause 4 (verified on E1, B-052): unit-ful head constants at label scale 1.** beta_min 0.01 is ~25x the label
variance: predicted sd / planted sd 1.11-1.29, coverage_80 above range. Switch: `nig_scale` (heads work in label
units / s; s = 0.01 = TF's reg_target_scale 100 inside the model).

## 2. Arms (3 epochs, seed 0 unless n = 3; means over seeds)

| arm | n | high slope / ratio / corr | low slope / ratio / corr | close corr | blocking fails | verdict |
|---|---|---|---|---|---|---|
| baseline | 3 | 1.79 / 0.81 / 0.92 | 1.73 / 0.65 / 0.92 | +0.41 | 7.7 | reference |
| **bundle** = bn_warmup + order_detach_close | 3 | **1.06 / 1.01 / 0.96** | **0.92 / 0.70 / 0.97** | +0.65 | 7.3 | slope fixed (every seed) |
| **bundle + nig_scale 0.01** | 3 | 1.22 / 0.89 / 0.96 | 1.32 / 0.85 / 0.97 | **+0.83** | **5.7** | best close, sd, coverage, gates |
| bundle + readout_dropout 0 | 3 | see section 3 | | | | |
| bnw (bn_warmup) | 1 | 0.91 / 0.71 / 0.97 | 0.93 / 0.99 / 0.97 | -0.12 | 7 | cause 1 |
| detach (order_detach_close) | 1 | 1.48 / 0.70 / 0.92 | 1.28 / 0.68 / 0.92 | +0.72 | 5 | cause 2 |
| no_order (enforce_order false) | 1 | 2.26 / 1.02 / 0.91 | 1.65 / 0.77 / 0.91 | +0.59 | 8 | cause 2 (loses order) |
| offset scale 0.02 (intermediate switch, superseded by nig_scale) | 1 | 1.90 / 1.00 / 0.90 | 1.65 / 0.67 / 0.91 | +0.68 | 6 | close only |
| nig01 (nig_scale 0.01) | 1 | 2.26 / 0.78 / 0.94 | 2.22 / 0.82 / 0.94 | +0.81 | 7 | close, sd |
| E1 at reg_target_scale 100 (data side) | 1 | 2.48 / 0.79 / 0.92 | 2.44 / 0.75 / 0.93 | +0.50 | 8 | label scale is not the slope cause |
| beta-NLL 0.5 / 1.0 (normalized) | 1 | 1.50 / 1.59 slope | 1.80 / 1.85 slope | -0.31 / -0.25 | 9 | rejected |
| lambda_reg_end 0 | 1 | 1.48 / 0.55 / 0.92 | 1.63 / 0.83 / 0.92 | -0.21 | 8 | rejected |
| beta_min 1e-5 | 1 | 1.85 / 0.85 / 0.91 | 2.06 / 1.02 / 0.91 | +0.05 | 8 | mean only, slope worse |
| no input noise / feature dropout | 1 | 1.59 / 0.73 / 0.90 | 1.82 / 0.92 / 0.89 | -0.45 | 9 | rejected |
| dropout 0.05 (all) | 1 | 1.94 / 0.87 / 0.91 | 1.82 / 0.73 / 0.93 | +0.53 | 7 | no slope gain at 3 epochs (BN dominates) |
| class_loss_weight 5, no Kendall | 1 | 1.55 / 0.69 / 0.90 | 1.50 / 0.85 / 0.88 | +0.20 | 5 | A5 still fails (B-060) |
| grad_clip_norm 1.0 | 1 | 4.00 / -0.14 / 0.69 | 5.22 / 1.04 / 0.77 | +0.45 | 19 | harmful at label scale 1 |
| keras init | 1 | 1.80 / 0.60 / 0.92 | 1.97 / 0.82 / 0.90 | +0.19 | 11 | no gain |
| EMA 0.999 (with warm-up) | 1 | 1.81 / 0.69 / 0.91 | 1.91 / 0.67 / 0.93 | +0.43 | 6 | no gain |
| bundle + TR1 (clip 1, keras, EMA, wd groups) | 1 | 1.32 / 0.09 / 0.32 | 2.94 / 0.46 / 0.75 | +0.43 | 17 | harmful (clip) |
| TR1 round 1 (EMA without warm-up) | 1 | broke mu (sd 30x) | | | 21 | fixed by the EMA warm-up (D-021) |

Seed-paired, 3 epochs (|x - 1| for slope and ratio, difference bundle - baseline, mean [min, max] over seeds):
bundle: high slope -0.63 [-1.00, -0.43], low slope -0.64 [-0.88, -0.46], high / low corr +0.04 / +0.05 (every seed),
close corr +0.24 [-0.19, +0.92]. Bundle + nig_scale: slope -0.57 / -0.41 (every seed), mean ratio -0.14 / -0.20
(every seed), close corr +0.42 [+0.04, +1.03] (every seed).

## 3. Longer runs (10 epochs, 3 seeds) — training length is a confounder

| arm | high slope / ratio / corr | low slope / ratio / corr | close corr | blocking fails |
|---|---|---|---|---|
| baseline | 1.31 / 0.93 / 0.96 | 1.18 / 0.91 / 0.96 | +0.11 | 7.3 |
| bundle | 1.23 / 0.97 / 0.96 | 1.15 / 0.85 / 0.96 | **+0.67** | 6.3 |

Paired: slope -0.08 [-0.34, +0.09] / -0.01 [-0.13, +0.13] (no longer different: 1000 steps let the Keras moving
stats converge), close corr **+0.56 [+0.15, +0.83] (every seed)**. So bn_warmup matters for short runs (gates, smokes,
fractional arms) and changes nothing once the moving statistics converged; order_detach_close helps at both lengths.
The residual slope 1.15-1.3 at 10 epochs is cause 3 (dropout dilution).

Bundle + nig_scale and bundle + readout_dropout 0: runs in progress at the time of this commit (added below when done).

## 4. Gates (3 seeds, E1 + E2; E3 NOT run: no real artifact on this machine)

| configuration | E1 A / B / C | E2 A / B / C / canary | E1 failed checks (seeds) |
|---|---|---|---|
| baseline (fef8d2fa) | fail / pass / fail | fail / pass / pass / **pass** | A3 2, A5 3+3, A6 2+3, A7 2+1, coverage_80 high 1 low 1, T-sharp high 2 low 3 |
| bundle (a5904c8f) | fail / pass / fail | fail / pass / pass / **pass** | A3 1, A5 3+2, A6 2+3, A7 1, coverage_80 low 1, std_scale high 2 low 2, T-sharp high 2 low 3 |
| bundle + nig_scale 0.01 | fail / pass / fail | fail / **fail** / pass / **pass** | A5 3+3, A6 3+3, T-sharp high 2 low 3; E2 B gap_ci_contains_0 low seed 0, close seed 2 |

The leakage canary passes for all three on every seed (shuffled labels learn nothing). None passes the protocol (E3
missing; A5 / A6 / T-sharp fail everywhere). With nig_scale, E2 gate B failed on 2 of 9 head x seed rows (the
confidence gap on permuted labels has a CI just off 0: [0.008, 0.22], [-0.20, -0.01]) — an alarm to read before that
switch is used. The bundle raises std_scale_vs_planted to 1.22-1.29 (just over 1.25 on 2 seeds): with the BN fix the
level path is no longer shrunk, which widens the predicted sd against the beta floor (cause 4).

## 5. Verdicts

| hypothesis | verdict |
|---|---|
| NIG NLL itself compresses mu | rejected (perfect-feature probe: slope 1.0; beta-NLL does not help) |
| evidence regulariser lambda_reg | rejected |
| label scale / softplus offsets cause the slope | rejected for the slope (scale 100 worse); verified for the close coupling and sd scale |
| beta floor (B-052) | verified: over-wide intervals; beta_min 1e-5 fixes the mean only |
| BatchNorm moving stats at short runs (B-062) | **verified** (population probe, bnw arm, 3 seeds paired) |
| ordered means coupling (B-063) | **verified** (detach / no_order, 3 seeds paired at 3 and 10 epochs) |
| dropout regression dilution (residual slope) | verified by the readout probe; switch effect: section 3 |
| training length | confounder: 10 epochs fixes cause 1 by itself |
| TR1 switches (clip, keras init, EMA) | no gain on E1; grad clip 1.0 harmful at label scale 1 (TF clipped labels x 100) |

## 6. Recommended bundle

`model_overrides: {"bn_warmup": true, "order_detach_close": true}` — no gate regression except std_scale (cause 4),
slope fixed at short runs, close signal recovered at every length. Add `nig_scale: 0.01` when intervals / coverage
matter (best close corr, mean ratio, sd scale, fewest fails) but read its E2 gate-B alarm first. `readout_dropout`:
section 3. Defaults stay off (D-022); making any of them the default is an owner decision.

## 7. Not verified

- No GPU, no real data, no E3: none of these is shown on the real artifact. On real data an epoch is thousands of
  steps, so cause 1 is probably absent there; B-058 (real-data shrinkage) is not explained by this study. Cheap check:
  compare moving vs population BN variance in the baseline-002 checkpoint.
- Short runs only (3 / 10 epochs on 51,168 training windows); single seeds for every arm but the bundle rows.
- No different architecture was tried (step 4 skipped: the bundle's gates still fail on A5 / A6 / T-sharp, so the
  structure is not stable yet).
- B-060 (class heads starve, A5) and the A6 cosine failures are untouched.
