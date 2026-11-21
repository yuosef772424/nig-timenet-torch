# Pre-registration BR: BTC-relative inputs, GRU, 2 seeds (owner 2026-11-19)

Written before any number of the run. Exploratory (rule C5).

## Owner's request
A quick GRU training on 2 seeds whose inputs are normalised by subtracting BTC: per bar of the window
log(high / previous close) of the coin minus the same value of BTC for the same bar, likewise low and close; RSI and
stochastic on the coin price divided by BTC's (as an ALT/BTC pair); BTC added as a feature of every sample.

## Data
`/content/art_btcrel` = artifact `c572b1b36960` with its channels replaced by `data_io/btc_relative.py` (11 channels:
rel_high / rel_low / rel_close, ratio RSI 14, ratio stochastic %K / %D on ratio closes, coin relative volume, BTC
log high / low / close moves, BTC RSI 14). Labels unchanged (the coin's own return).

## Arm and its reference
BR = arm R of `rolling-ind-001` with only the inputs changed: single-coin GRU 64 x 2, lr 3e-4 (the R configuration,
so that one thing changes; "quick": 8 epochs, patience 3; lr 1e-5 needs more than 16 epochs, see the stage-1 look),
400k train rows, top 100 coins, the same 7 expanding folds, seeds 0 and 1, rows batches, no rank term.
Reference: R (seeds 0, 1, already finished on commit 458bcdb).

## The question and the accept rule
Does a model that sees coins against BTC pick coins that beat BTC? Read with `read_arms.py` (long book: VAL 90th
percentile of mu_close; long and short apart) on TEST pooled over the 7 folds per seed:
1. Primary: the long book minus BTC with both legs' costs (`long_btc_real`) has a day-block CI above 0 at level
   1 - 0.05 / 2 on BOTH seeds and stays above 0 without the best 5 % of days.
2. And it is above R's `long_btc_real` on both seeds.
Reported next to it: the unhedged long book, beta to BTC, the long book minus the equal-weight market, close rank IC
by timestamp (the coin-picking skill itself) against R's, and the short book (information only).
If BR fails, the failure reading of rule C7 is written before any next arm.
