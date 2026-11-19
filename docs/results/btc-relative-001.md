# btc-relative-001: BTC-relative inputs, GRU, 2 seeds (owner 2026-11-19; exploratory, rule C5)

Job `docs/jobs/rolling-btcrel-001.json` on Colab L4, commit 34d574a, checkout `/content/nig_br`, after its smoke
(`rolling-btcrel-smoke-001`, rule E5). Inputs: `data_io/btc_relative.py` (11 channels per bar: the coin's log high /
low / close over the previous close minus BTC's of the same bar; RSI 14 and stochastic %K / %D of close_coin /
close_BTC; the coin's relative volume; BTC's own log moves and RSI). Labels unchanged (the coin's own return).
Otherwise arm R of `rolling-ind-001`: single-coin GRU 64 x 2, lr 3e-4, 8 epochs, patience 3, 400k train rows, top 100
coins, 7 expanding folds (TEST 2023-01 .. 2026-06), seeds 0 and 1. Pre-registration
`docs/audit/2026-11-19-arch-indicators/PREREGISTERED_BR.md` (commit 34d574a, before any number). Reading
`read_arms.py` (long and short books apart); outputs and charts on Drive `nig_timenet_runs/analysis-ind-001/btcrel/`.

## Verdict

**FAILS.** The primary number, the long book minus BTC with both legs' costs, is below 0 with a CI below 0 on both
seeds, and below R's:

| arm | seed | long book, mean day [CI 97.5 %] | without best 5 % | minus BTC, 2 costs [CI] | beta to BTC | close rank IC (mean of folds) |
|---|---|---|---|---|---|---|
| R | 0 | +0.173 % [+0.042, +0.302] | -0.072 % | -0.184 % [-0.285, -0.084] | 1.35 | 0.019 (both seeds) |
| R | 1 | +0.215 % [+0.072, +0.355] | -0.022 % | -0.130 % [-0.240, -0.023] | 1.35 | |
| BR | 0 | +0.097 % [-0.057, +0.253] | -0.169 % | -0.217 % [-0.339, -0.089] | 1.29 | 0.032 (both seeds) |
| BR | 1 | +0.139 % [-0.007, +0.280] | -0.107 % | -0.207 % [-0.315, -0.090] | 1.25 | |

## What the model learned

- **The coin ranking improved.** Close rank IC by timestamp, mean of the 2 seeds per fold, BR against R: 0.018 / 0.013,
  0.031 / 0.023, 0.027 / 0.017, 0.023 / 0.009, 0.031 / -0.002, 0.044 / 0.029, 0.047 / 0.040: higher in all 7 folds,
  +0.013 on average (about +70 %). Seeing coins against BTC helps the model order coins inside a timestamp.
- The beta of the long book to BTC fell a little (1.35 -> 1.25-1.29).
- The short book read apart (information only): minus the equal-weight market it loses -0.02 % per day with BR
  (CI contains 0) against -0.10 % with R (CI below 0): the relative skill shows on the short side too.

## Where it is right and where it fails

Right in ranking (all folds). Not enough for a book: the long book's gross over BTC (about +0.10 % per trade, from
`long_btc` = -0.05 % with one cost) is under one round trip. Per fold the long book (net per trade, mean of seeds)
still follows the market: fold 3 (2024-06 rally) +0.42 %, folds 5-6 (2025-06 .. 2026-06) -0.03 / -0.20 %.

## The samples it failed on

As R (`docs/results/btc-hedge-long-001.md`): the hedged losses are the market drops, where the chosen high-beta
coins fall more than BTC; per-row tables are on Drive (`fold_*/battery_table_test.parquet`), not broken down here.

## Bias

Long only by construction; beta to BTC 1.25-1.29. The short book's day mean is positive (+0.19 % per day, CI above 0
at 97.5 %) but negative without the best 5 % of days and about 0 once the market move is removed: it is the market's
falls, not a coin edge. Information only (not pre-registered as a candidate).

## Selective use

One rule (the pre-registered VAL 90th percentile long book). No rule chosen on TEST.

## Why it failed

**Signal smaller than the cost:** the relative inputs raise the coin-ranking skill in every fold, but the extra
gross per trade of the top decile over BTC (about 0.10 %) is below the 0.16 % round trip, and hedging adds a second
one. The cost wall again (B-074; `docs/results/b1-btc-timing-001.md`).

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| BR1 | A label relative to BTC (log move of coin / BTC over the horizon) aims the loss at what the inputs now show: the BTC-hedged top decile's gross per trade rises above 0.16 % | BR1's `long_btc` gross per trade stays at or below BR's on both seeds | the btc_relative artifact with a re-targeted label; same job, 2 seeds |
| BR2 | The rank term (`cross_section_loss: rank_ic`, time blocks) on BTC-relative inputs adds to the IC gain | close rank IC not above BR's in at least 5 of 7 folds on both seeds | arm IR's reading (running) shows the rank term on absolute inputs first |
| BR3 | A longer horizon (24-48 h) lets the ranking pay the round trip (the IC gain is small per 8 h) | a ridge on the BTC-relative inputs at 24-48 h has gross per round trip below 0.16 % | ROADMAP N3, model-free first |
