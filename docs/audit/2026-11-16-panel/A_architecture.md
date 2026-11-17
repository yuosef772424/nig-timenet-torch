# Report A: architecture of the panel model (panel audit 2026-11-16)

Metrics and accept rule: `PREREGISTERED.md` (written before these numbers). Every number here is **a look, not a
result** (rule C5): small slices, 2 seeds per arm (rule C4), one model class (GRU encoder, lr 3e-4).

## What was read

- `model/cross_coin.py`: `RelativeInputs` (`cross_input: relative`: each coin's scaled window becomes [own, own -
  timestamp mean, (own - mean) / timestamp sd, mean], bar by bar), `CrossCoin` (`cross_coin_layers`, kinds
  `attention` / `mean`, a zero-initialised gate after the trunk), the timestamp id as one extra input channel.
- `model/builder.py` `NigTimeNetV2.forward`: trunk -> (cross-coin block) -> NIG heads -> `ordered_means` -> outputs;
  the class heads read the same trunk.
- `trainer/groups.py`: whole-timestamp batches and evaluation chunks. `workflow/flat_run.py`: `time_ids`.
- Encoder use (B-070, N-20): NOT re-measured. The doctor's block-use card does not run on a cross-section run (N-30,
  now B-076). N-20 stands as measured on 2026-11-14 (all encoder blocks off moves the label rank IC by <= 0.02).

## Arms

All arms: default model (preset + `MODEL_FIRST_FIXES`) with the GRU encoder, whole-timestamp batches, 12 epochs at
most, patience 4, selection `val_rank_ic_reg` (the `_ls` arm: `val_ls_sharpe_close`), batch 256, seeds 0 and 1.
Run on the Colab L4 through the bridge with the entry `panel` (the same code the full job runs), because the 16-20
arm runs per dataset would have taken the 4-core sandbox about 4 hours; the 60,000-row slice ran on the sandbox CPU
(`docs/results/panel-cpu-001.md`).

| arm | what it changes |
|---|---|
| per_coin | nothing: the per-coin model on whole-timestamp batches (the baseline arm) |
| relative | `cross_input: relative` |
| relative_mean_block | + `cross_coin_layers: 1`, kind `mean` |
| attention_block | per-coin inputs + `cross_coin_layers: 1`, kind `attention` |
| per_coin_rank_loss | per-coin model + `cross_section_loss: rank_ic` (report B) |
| relative_ic_loss / relative_rank_loss | `relative` + `cross_section_loss: ic` / `rank_ic` |
| relative_rank_loss_ls | the same, epoch selected by `val_ls_sharpe_close` |
| relative_center / relative_rank_loss_center | + `cross_center` (the close mean minus its timestamp mean) |

## E4: the known answer (180,000 rows, 50 coins, planted specific IC 0.15; TEST = 116 days)

Oracle (the exact conditional mean, scored by the same code): metric 1 **87.1 %** winning days [80.1, 92.7], metric 2
**91.4 %** [87.5, 95.5], relative rank IC **0.1435**. The market share of the close variance is 0.58 (real: 0.4-0.6).

<!-- TABLE:e4:main -->

## Real data: the long slice (240,000 rows, 50 longest-listed coins, 2023-12 .. 2026-02, inside the full run's training era; TEST = 115 days)

Fingerprint `a1de68a4a4a6` (a slice of `c572b1b36960` built by `nig-envs e3p --coins 50 --max-rows 240000`), split
70 / 15 / 15 by time: train 167,900 / val 34,300 / test 34,400 rows.

<!-- TABLE:real:main -->

## Findings

1. **The model can read a relative signal when there is one.** On E4 the arms with relative inputs reach the
   oracle's market-neutral book (87-90 % winning days against 91.4 %; relative rank IC 0.120-0.131 against 0.1435).
   The per-coin arm stops at 81 % and IC 0.089: it sees the signal under the market noise of its own window.
2. **The trunk-level blocks add nothing over the relative inputs; attention alone hurts.** `relative_mean_block` =
   `relative` on both data sets (as N-28 / section 9 found). `attention_block` without relative inputs is the worst
   arm on E4 (78 %, IC 0.086, below the per-coin arm) and the lowest IC on the real slice.
3. **The close mean's LEVEL is the defect on metric 1.** On E4 every arm without `cross_center` wins 47-66 % of days
   in metric 1 although its ranking is at the oracle's level; with `cross_center` the same model wins 81-86 %
   (oracle 87.1 %). The head puts a common level into the mean (report B, direction bias): the sign of the mean is
   then a bet on the market, which nothing predicts.
4. **On real data no arm separates on the day metrics.** Metric 2 is 55-64 % winning days for every arm with CIs of
   +-10 points; metric 1 is 47-57 %. What does move, on both seeds where both exist, is the relative rank IC:
   relative inputs +0.02 .. +0.03 over per-coin, the rank term a further +0.01 .. +0.03 (report B).
   `cross_center` does not raise metric 1 on real data (49-55 %): here the relative sign of ALL coins has no edge,
   only the tails of the ranking have a small one.
5. **No arm passes the accept rule on real data**: none beats every baseline with a paired CI that excludes 0, on
   either metric. On E4 the centred arms pass metric 2 against momentum and the asymmetry on some seeds only (the
   baselines are strong there by construction: the planted signal is the last three bars).
6. **The architecture is not the limit on real data; the inputs are** (as N-08 / B-070 said for the per-coin model).
   The same arms that reach 91 % of the oracle on E4 reach a rank IC of 0.06-0.10 on real data. Three log-move
   channels of the coin and of its timestamp hold no more.

## Proposals

| # | Proposal | Evidence | Built as |
|---|---|---|---|
| A1 | Train panel runs with `cross_input: relative` | E4: 81 % -> 87 % (metric 2), IC 0.089 -> 0.120; real: IC +0.02 .. +0.03; full artifact (section 9 of the 2026-11-14 audit): close IC 0.055 -> 0.085 | exists (D-028); arm `relative` of `panel-001` |
| A2 | Centre the close mean inside its timestamp | E4: metric 1 47-58 % -> 81-86 %; real: no change | model option `cross_center` (off by default), arm `relative_rank_loss_center` |
| A3 | Do not use the attention block; do not stack the mean block on relative inputs | finding 2 | no change (both stay available, off) |
| A4 | Give the model inputs the window does not hold: volume / taker flow / funding per coin, and a longer horizon label | finding 6, report C finding 6 (cost) | not built: needs a new artifact from the data pipeline (B-055, B-059, B-074) |
| A5 | Make the doctor and the health report read a cross-section run before any panel model becomes a default | N-30, B-076 | not built |

Not verified: the transformer encoders on the panel route (one encoder here); more than one cross-coin layer; the
panel arms on the full artifact with the new switches (job `panel-001`).
