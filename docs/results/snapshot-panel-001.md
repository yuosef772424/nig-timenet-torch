# Snapshot panel model ("100 x 3") against today's panel model: walk-forward on the full artifact (2026-11-18)

Exploratory (rule C5). Pre-registration: `docs/audit/2026-11-18-snapshot-panel/PREREGISTERED.md` (written before any
number); design `docs/audit/2026-11-18-snapshot-panel/DESIGN.md`; known-answer look on E4
`docs/audit/2026-11-18-snapshot-panel/E4_READING.md`. Jobs `docs/jobs/snapshot-rolling-{base,joint,hybrid}-001.json`,
one commit (a2096fe), Colab L4, artifact `c572b1b36960`, 7 expanding folds (first train window 3 years, test and step
180 days, embargo 2 days), the 100 coins with most TRAIN rows of each fold, all their rows, 2 seeds, batch 3200 rows,
12 epochs / patience 3, selection `val_rank_ic_reg`, the cross-section rank term in every arm. Outputs on Drive
`nig_timenet_runs/snapshot-rolling-<arm>-001/out` (`rolling_summary.md`, `pooled_days_seed_<s>/panel_report.md`).
Number of variants in this comparison: 3 arms. "seed a / b" = the two seeds' pooled reports of an arm (the file
names on Drive do not say which seed is which in this session's listing; a / b is not matched across arms).

## Result against the accept rule

| arm | params | close relative rank IC, mean of 7 folds (sd over folds) | high | low | pooled TEST days, cross-sectional rank IC of the relative move (seed a / b) | market-neutral top / bottom 20 %: winning days gross, mean day net (seed a / b) |
|---|---|---|---|---|---|---|
| base (today's panel model) | 0.14 M | **0.087 (0.017)** | **0.361 (0.016)** | **0.443 (0.022)** | **+0.087 / +0.088** | 58.9 % / 58.9 %, -0.117 % / -0.117 % |
| joint (the moment as one vector) | 1.41 M | 0.013 (0.004) | 0.158 (0.016) | 0.200 (0.026) | +0.016 / +0.010 | 54.1 % / 53.5 %, -0.149 % / -0.157 % |
| hybrid (joint + shared per-coin path) | 1.47 M | 0.058 (0.012) | 0.281 (0.027) | 0.359 (0.016) | +0.058 / +0.059 | 57.2 % / 58.4 %, -0.135 % / -0.125 % |

Per fold, base is above both snapshot arms on every head in every one of the 7 folds; the seed sd per fold is <= 0.003
for base on close and <= 0.008 for the snapshot arms, so every gap is many seed spreads wide. Accept rule (a) fails for
joint (-0.074) and hybrid (-0.029) on both seeds; (b) fails too (high -0.20 / -0.08, low -0.24 / -0.08). Neither arm
replaces base. No arm is profitable after costs on all rows (B-074: the gross edge of a 4 h book is below its cost).

## What the model learned

- The snapshot arms DID learn something above their trivial baselines: hybrid's relative rank IC (+0.058) beats
  momentum (-0.048), reversal (+0.048) and the last-bar asymmetry (+0.038) on the pooled days of both seeds; its high /
  low rank IC (0.28 / 0.36) are far above zero. The joint arm alone reads almost nothing on close (+0.013, below every
  trivial ranker but momentum) and about half of base's high / low skill.
- The joint arm's close IC grows with the training size of the fold (0.010, 0.010, 0.014, 0.011, 0.014, 0.011, 0.021
  for folds 0..6, about 3,700 to 13,000 training moments): it is learning slowly, sample by sample.
- The uncertainty of every arm ranks the error (sd vs |error| +0.19..+0.38 pooled); the snapshot arms' standardised
  errors are too wide (z sd 1.8-2.5 for joint, 1.0-1.5 for hybrid, against 0.95-1.19 for base), so their sd is too
  small for their errors although the 80 % coverage looks closer to 0.80.
- E4 (known answer): base reaches the oracle on close (0.127 of 0.130); joint 0.01; hybrid 0.09.

## Where it is right and where it fails

Measured per fold (the table above, `rolling_summary.md`): the ranking of the arms is the same in all 7 folds (2023-01
.. 2026-06: calm, trending and volatile half-years), so the failure is not one regime. Per head: the loss is largest on
high / low (the volatility ranking that the shared per-coin reading does best) and on close for the joint arm.
NOT measured: the error map by coin group (history length, volatility), by confidence decile and by event class: the
prediction tables of the folds (25-30 MB each) are on Drive, out of reach of this session's Drive tool (10 MB) and the
Colab runtime ended before the reading. That map is the first step of the next session (`eval/error_map.py` on the
`test_table.parquet` of each fold, on Colab).

## The samples it failed on

Not read in this session (same reason: the row tables are on Drive). What is known without them: the joint arm's
TEST direction is balanced (49.6 % / 49.4 % of rows predicted up against 48.6 % realised) and its hit rate is 49.1 % /
49.4 %, i.e. its close mean carries no sign information on any large group of rows; its per-coin share predicted up
spans 0.19 .. 0.77 with a correlation of +0.07 / +0.16 with the realised share: coin-level offsets learned from the
coin embedding / per-coin weights that do not generalise to the test half-year.

## Bias

- base: 46.8 % / 42.1 % of rows predicted up (realised 48.6 %): a mild down lean, per-day correlation -0.15 / -0.21.
- hybrid: 36.3 % / 40.5 % predicted up: a clear down lean; per-coin share 0.01 .. 0.81 (one seed) with -0.23 / +0.05
  correlation to the realised share: the per-coin terms push some coins to one side for the whole test window.
- joint: balanced on average (49.6 % / 49.4 %) but with no skill.
A VAL-fitted re-centring was not run (it does not change a rank IC, the primary metric).

## Selective use

The panel report's rules (thresholds fitted on VAL, applied to TEST; metric 1 = equal-weight sign portfolio, net of
0.0008 per side):

| rows kept | base (seed a / b) | hybrid | joint |
|---|---|---|---|
| all rows | -0.078 % / -0.096 % | -0.170 % / -0.124 % | -0.166 % / -0.140 % |
| top confidence (~21 %) | **+0.027 % / +0.041 %** | -0.078 % / +0.011 % | -0.129 % / -0.125 % |
| top \|mu\| (~21 %) | **+0.034 % / +0.061 %** | -0.104 % / +0.014 % | -0.108 % / -0.082 % |
| low sd (~20 %) | -0.166 % / -0.167 % | -0.200 % / -0.179 % | -0.207 % / -0.194 % |

Only base keeps a positive mean net day on both seeds when it trades its most confident or largest-move fifth (the
rule set existed before this run; 4 rules x 3 arms were looked at, and no CI is printed for these rows): a hypothesis
for the next walk-forward, not a result. The snapshot arms do not reach it on both seeds.

## Why it failed

**Sample efficiency (signal there but not read).** The data hold the relative signal (base reads 0.087 on the same
rows and folds; on E4 base reaches the oracle), so "no signal" is excluded. The snapshot model gives every coin its
own input and output weights over a 512-wide moment vector: one independent example per moment for each coin's
weights (about 3,700 to 13,000 moments per fold) against ~100 examples per moment for the shared per-coin function.
Evidence: (1) E4, where only a per-coin beta could add anything, joint reads 0.01 against 0.127; (2) on the real folds
the joint close IC rises with the number of training moments but stays near 0.01-0.02; (3) no memorisation signature
(on E4 train and val NLL move together; on the real folds the best epoch is late), so it is under-learning, not
over-fitting; (4) adding the joint path to the shared path (hybrid) LOWERS the shared reading (0.058 against 0.087):
the joint features enter the readout as noise that the shared path then has to work around.

## Next hypotheses

| # | hypothesis | measurement that would refute it | cheapest test |
|---|---|---|---|
| H1 | the cost is the per-coin weights, not the view of the whole moment: a cross-coin block with SHARED weights (attention of each coin over the coins of its moment, `cross_coin_layers`) keeps base's IC and can add the cross-coin part | base + `cross_coin_layers: 1` is below base on the pooled close relative IC on both seeds | one more arm on the same 7 folds (one job, ~1 h on an L4) |
| H2 | the joint path can only help as a residual on a converged shared model: gate it at 0 (like `CrossCoin.gate`) and / or train it after the shared path | hybrid with a zero-init gate on the joint features stays below base on both seeds | a gated hybrid arm on E4 first (minutes), then the folds |
| H3 | base's top-confidence / top-\|mu\| fifth is profitable after costs | the pre-registered rule (top 20 % by confidence, fitted on VAL) has a day-block CI that includes 0 on either seed in a new walk-forward with `n_variants` counted | the same base job with the rule written before the run and the CI printed |
| H4 | the error map will show where the snapshot arms lose (thin / young coins, through the per-coin weights) | the loss of joint against base is the same in every coin-history group | `eval/error_map.py` on the saved fold tables (Colab, no training) |
