# Report C: normalisation of the close label and of the inputs (panel audit 2026-11-16)

Metrics and accept rule: `PREREGISTERED.md` (written before these numbers). Every number here is **a look, not a
result** (rule C5): one artifact, no model, a ridge on 8 / 16 simple features.

Tool: `normalisation_look.py` in this folder (numpy only; `python normalisation_look.py ARTIFACT OUT --top-coins N`).

## No leakage: what was fitted where

- Rows: the run's own time split with whole timestamps (`SplitSettings(cross_section=True, top_coins=...)`, purged by
  label end). Every timestamp is on ONE side of the split.
- Fitted on TRAIN rows only: `beta_i` (OLS slope of the coin's label on the market's), `sigma_i` (sd of the coin's
  residual), the feature mean / sd, the ridge weights. Scored on VAL rows. The full run's TEST rows were never read.
- The timestamp mean used inside a target (`y_i - mean_j y_j`) is the mean of the labels of the SAME timestamp. It
  defines what is traded (a move relative to the market); it is label-side and is never an input.
- The relative INPUT features are the coin's own window features minus the mean over the coins of the timestamp at
  the same bars (all known at prediction time).

## Data

| look | artifact | coins | train rows | val rows | val days | train end | val end |
|---|---|---|---|---|---|---|---|
| full artifact (Colab, numpy on the CPU of the runtime) | `c572b1b36960` | 100 longest-listed | 1,113,463 | 54,717 (106 days) | 106 | 2026-02-24 | 2026-06-15 |
| CPU slice `e3p` (this sandbox) | `4416868f4483` (slice of `c572b1b36960`) | 50 | 41,900 | 7,300 | 25 | 2025-12-25 | 2026-01-24 |

Label = `log(future_close / last_close)` over the horizon (2 bars of 4 h); realised return for the day metrics =
`future_close / last_close - 1`. Market = the equal-weight mean of the coins of the timestamp unless stated.

## Full artifact, 100 coins (the table to read)

Market share of the close label's variance on VAL: 0.39. Train beta: median 1.04, range 0.54 .. 1.24.

| target variant | cross-coin corr | effective coins (of 100) | ridge rank IC of its own target: own features / own + relative | rank IC of the relative move: own / own + relative | metric 1 winning days [CI] | metric 2 winning days [CI] | metric 2 mean day, gross / net |
|---|---|---|---|---|---|---|---|
| raw | +0.373 | 2.6 | +0.012 / +0.088 | +0.012 / +0.088 | 51.9 % [43, 61] | 60.4 % [51, 72] | +0.029 % / -0.131 % |
| minus the timestamp mean | -0.012 | 86.4 | +0.073 / +0.087 | +0.073 / +0.087 | 60.4 % [50, 70] | 59.4 % [51, 69] | +0.025 % / -0.135 % |
| minus beta_i x market | -0.012 | 86.4 | +0.069 / +0.085 | +0.072 / +0.086 | 57.5 % [47, 68] | 58.5 % [51, 67] | +0.019 % / -0.142 % |
| (r - beta x market) / sigma_i | -0.011 | 86.4 | +0.076 / +0.085 | +0.081 / +0.088 | 58.5 % [48, 69] | 58.5 % [50, 69] | +0.038 % / -0.122 % |
| cross-sectional rank | -0.012 | 86.4 | +0.102 / +0.108 | +0.102 / +0.108 | 57.5 % [48, 68] | 57.5 % [49, 68] | +0.044 % / -0.116 % |
| minus beta x (mean of the 10 longest-listed coins) | +0.081 | 10.9 | +0.037 / +0.085 | +0.037 / +0.086 | 49.1 % [39, 59] | 55.7 % [48, 65] | +0.021 % / -0.140 % |
| minus beta x BTC | +0.173 | 5.5 | +0.007 / +0.089 | +0.009 / +0.089 | 50.0 % [41, 60] | 59.4 % [52, 69] | +0.031 % / -0.130 % |

The day-metric columns are the ridge with own + relative features; metric 1 of a relative target is the sign of a
relative prediction (the equal-weight book is then market-neutral by construction).

Baselines on the same 106 VAL days (gross): metric 1: always-long 51.9 %, yesterday's market sign 35.8 %, last-bar
asymmetry 62.3 % (mean day +0.059 %); metric 2: momentum 50.9 %, reversal 49.1 %, last-bar asymmetry **68.9 %** (mean
day +0.066 %, rank IC of the relative move +0.059). Rank IC of the relative move: reversal +0.050, momentum -0.050.

## CPU slice, 50 coins (25 VAL days: too few days to rank anything; the correlations are the stable part)

| target variant | cross-coin corr | effective coins (of 50) | ridge rank IC of its own target: own / own + relative |
|---|---|---|---|
| raw | +0.386 | 2.5 | +0.011 / -0.004 |
| minus the timestamp mean | -0.020 | 50.0 | +0.002 / -0.002 |
| minus beta_i x market | -0.020 | 50.0 | -0.016 / -0.016 |
| (r - beta x market) / sigma_i | -0.011 | 50.0 | +0.027 / +0.016 |
| cross-sectional rank | -0.020 | 50.0 | +0.083 / +0.082 |
| minus beta x top-10 mean | +0.052 | 14.2 | +0.018 / -0.010 |
| minus beta x BTC | +0.201 | 4.6 | -0.024 / -0.018 |

`doctor.data_checks.overlap()` on the slice's train rows (the formula used above): `cross_coin_corr` 0.568,
`effective_coins` 1.73 of 50 (the train period of the slice is more correlated than its VAL month).

## Findings

1. **The raw close label is almost one coin.** Pairwise correlation 0.37 on the full artifact: 100 coins are worth
   2.6 independent ones. Taking out the equal-weight market brings 86 back. A market built from the 10 longest-listed
   coins leaves 10.9, BTC alone 5.5: **the equal-weight mean of the traded universe is the market to remove**; a
   top-coin or BTC market leaves most of the common move in.
2. **beta_i adds nothing over the plain mean** here (effective coins equal, ridge IC equal or lower): the train betas
   sit in 0.54 .. 1.24 around 1.04, so `y - mean` and `y - beta x mean` are nearly the same variable. Dividing the
   residual by `sigma_i` helps a little (IC of the relative move 0.081 vs 0.072 with own features).
3. **Where the signal is.** With the raw target a ridge on the coin's own features reads nothing (0.012); the same
   ridge with the features RELATIVE to the timestamp reads 0.088. With a relative target the own features already read
   0.073. So the information is the coin against its timestamp, and it can enter either through the inputs
   (`cross_input: relative`, already built) or through the target side. N-27 said this on 150,000 rows; it holds on
   1.1 M.
4. **The rank target is the best target for a linear reader**: 0.108 against 0.087-0.088 (+0.02) with the same
   features, 0.102 with own features only. Ranks remove the heavy tails of 8 h returns from the fit.
5. **Rank IC is not the day metric.** The 1-day reversal ranking has a rank IC of +0.050 and LOSES in the top /
   bottom 20 % book (49.1 % winning days, mean -0.013 %): the middle of the cross-section reverts, the tails keep
   going. The ridge with the highest IC (0.108) wins fewer days (57.5 %) than the single last-bar asymmetry feature
   (68.9 %, IC 0.059). A model selected by rank IC is selected on the wrong thing for metric 2.
6. **No variant is near profit after cost.** The best gross mean of the market-neutral book is +0.044 % per period
   (+0.066 % for the asymmetry baseline) against 0.16 % of cost (2 sides per unit of gross exposure): every net mean
   is -0.09 % .. -0.14 %. At a 4 h rebalance the gross edge is a quarter to a third of the cost.
7. **A trivial baseline is strong.** The last-bar asymmetry of DS1 wins 68.9 % of days in metric 2 and 62.3 % in
   metric 1 on VAL, gross. Any model has to beat that, not 50 %.

## Proposals (each measured by an arm in report A / B; nothing here changes a default)

| # | Proposal | Evidence here | Built as |
|---|---|---|---|
| C1 | Keep the label as it is for the NIG heads (their outputs must stay returns); add a term that fits the mean to the label's RANK inside the timestamp | finding 4 (rank target +0.02 IC), finding 1 (the raw loss spends its gradient on 2.6 coins' worth of common move) | `TrainSettings.cross_section_loss = "rank_ic"` (off by default) |
| C2 | Let the close mean be a move relative to the timestamp | findings 1-3: only the relative part is readable; metric 1 with a relative sign 57-60 % vs 51.9 % with the raw sign (ridge) | model option `cross_center` (off by default) |
| C3 | Select the epoch by the book of metric 2, not by rank IC | finding 5 | selection metric `val_ls_sharpe_close` |
| C4 | Do not build a beta-neutral or top-coin-market variant | findings 1-2 | not built |
| C5 | Report every panel arm against the last-bar asymmetry | finding 7 | `eval/day_metrics.py` baselines |

Not done here: the same table on the full run's TEST rows (kept unseen on purpose); volume- or liquidity-weighted
markets (the artifact holds no volume: B-055); other horizons (the artifact has one label, 2 bars).
