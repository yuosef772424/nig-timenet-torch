# Pre-registration M3: market residual (method 3), a LOCAL CPU look (owner 2026-11-19)

Written before the residual artifact is built and before any number. Exploratory (rule C5); one machine-size look,
NOT the owner's artifact (public Binance futures klines, `data_io/vision_slice.py`: the 60 default coins, 4h,
2022-01 .. 2025-12, stride 2, 252,562 rows).

## Method 3 (`data_io/market_residual.py`)
Market = equal-weight close log move of the 60 coins per bar; beta per coin per bar from its last 180 bars (>= 60,
bars <= k only). Inputs per bar: the coin's log high / low / close moves minus beta_k x market; RSI / stochastic of the
residual path; relative volume; the market move and its RSI; beta. Close label = own label - beta_t x market move over
the label span (high / low labels unchanged).

## Question
Does a model trained on market-residual inputs and labels pick coins that beat the market by their beta, better
than the same model on raw inputs / own labels?

## Arms (one commit, `jobs.rolling` on CPU)
- A: the raw artifact (3 channels, own label).
- M: the residual artifact (10 channels, residual label).
Both: single-coin GRU 64 x 2, lr 3e-4, 5 epochs, patience 2, 60k train rows, all 60 coins, expanding folds with a
730-day first train window, 180-day test windows, 4 folds (TEST 2024-01 .. 2025-12), embargo 2 days, seeds 0 and 1.

## Model-free bound first (same rows)
Ridge (features of the last bars: residual / raw moves summed over 1, 6, 18 bars, the bar range, RSI) fitted on each
fold's train rows, scored on its test rows: rank IC by timestamp against the residual label and against the own
label. If the ridge finds no residual ranking (IC <= 0.01), M is not expected to either.

## Reading (TEST, pooled over the folds, per seed; the rows of A and M are the same rows, joined by coin and time)
1. Close rank IC by timestamp against the RESIDUAL label: M (its mu) vs A (its mu).
2. Top decile (threshold = the 90th percentile of mu on the fold's VAL): mean residual return per trade, M vs A
   (gross), with a day-block CI.
"Feasible" = M above A on 1 AND 2 on both seeds. "Pays" = M's top-decile residual gross per trade above 0.16 % (one
round trip; a long-only trader keeps the market exposure) and above 0.32 % (with a beta hedge). A look on 60 coins and
2 years of tests: a pass sends method 3 to the full artifact on the GPU (2 seeds, 7 folds); a fail gets the C7 reading.
