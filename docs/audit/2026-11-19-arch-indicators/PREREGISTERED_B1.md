# Pre-registration B1: buy BTC at the times the model is bullish (owner 2026-11-19, "start with B1")

Written before any B1 number is computed. Exploratory (rule C5). Hypothesis B1 of `docs/results/btc-hedge-long-001.md`.

## Data
The saved VAL and TEST tables (`battery_table_{val,test}.parquet`) of arm R of `rolling-ind-001` (seeds 0, 1) and of
`rolling-single-002` (seeds 2, 3): 7 expanding folds per seed. No training.

## Signal (per timestamp, from the model's outputs only)
Per fold: q = the 90th percentile of mu_close on the fold's VAL table (as the long book). For every timestamp t,
`long_share(t)` = the share of that timestamp's rows with mu_close >= q.

## The rule (primary, the only one judged)
Per fold: L = the 80th percentile of `long_share` over the VAL timestamps. On TEST, buy BTCUSDT at every timestamp
with `long_share(t) >= L` (about the top 20 % of timestamps), hold to the label horizon (8 h), net per trade =
BTC y_ret - 0.0016 (0.08 % per side). Day book = the mean net of the day's trades; days pooled over the 7 folds per seed.

## Baselines on the same TEST days
- BTC always long: buy BTC at every timestamp (the same holding), mean day net.
- BTC at the complementary timestamps (long_share below L): shows whether the selection, not BTC, carries the result.

## Accept rule (all on EVERY one of the 4 seeds)
1. The mean day net has a day-block bootstrap CI above 0 at level 1 - 0.05 / 4.
2. The mean day net without the best 5 % of days is above 0.
3. The mean day net is above BTC always long on the same days.

## Information only (not judged; listed so that nothing is chosen after the fact)
The same rule with L at the 70th and the 90th percentile; the signal `mean mu_close over the timestamp's rows`
instead of `long_share` (L at its VAL 80th percentile); per-fold net per trade; BTC's mu_close alone (its own row's
mu >= its VAL 80th percentile). If the primary rule fails, the failure reading (rule C7) is written before any of
these is turned into a candidate; a variant that looks good here is a hypothesis for a new pre-registration.
