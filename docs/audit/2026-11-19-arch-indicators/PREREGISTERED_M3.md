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

## Amendment 1: M3a, longer horizons, model-free (written before these numbers)
The same 60-coin data rebuilt with label horizons 6 bars (24 h) and 12 bars (48 h), stride = horizon (labels do not
overlap), then `market_residual`. Ridge (the residual feature set, fitted on the residual label of each fold's train
rows), the same 4 test windows: rank IC by timestamp and the per-timestamp top 10 % residual gross per trade, against
8 h (horizon 2). "Can pay" = top 10 % residual gross per trade above 0.16 % at some horizon, in at least 3 of 4 folds.

## Amendment 2: putting the market back (owner 2026-11-19, before these numbers)
Question: does M's coin selection, with the market added back, make a better LONG-ONLY book (own returns, cost
0.16 % per trade) than arm A alone? Same local runs (A, M; seeds 0, 1; 4 folds). Variants, fixed here:
- V0 (today's way): A's long book, VAL 90th percentile of A's mu.
- V1 (market back): score = M's mu + beta_t x A's mean mu over the timestamp's coins (A's market forecast; beta_t
  from the residual artifact's last bar); VAL 90th percentile of the score.
- V2 (timing x selection): at timestamps whose mean A mu is at or above its VAL 80th percentile, buy the top 10 % of
  that timestamp's coins by M's mu.
- Baseline: always long (every row).
Read per seed on TEST pooled over folds: net per trade, mean day with a day-block CI (95 %), without the best 5 % of
days, win days. "Better" = above V0 on mean day AND on the mean without the best 5 % of days, on both seeds.

## Amendment 3: exit threshold / hysteresis (owner 2026-11-19, idea from another model; before these numbers)
Idea: enter a coin when its mu reaches the top 10 %, keep it while it stays in the top 25-30 %, exit below. Fewer
round trips, the same selection. Local runs A and M, seeds 0 and 1, 4 folds, TEST. Per fold the levels are VAL
quantiles of mu: entry = 90th percentile; exit = 90th (H0 = today's book: one 8 h period per entry), 75th (H75),
70th (H70). Rows of a coin are consecutive 8 h periods (stride = horizon); a missing period forces an exit. A held
coin earns each period's simple return; 0.08 % is charged on the entry period and 0.08 % on the exit period. Book
return per timestamp = mean over held coins; day = mean of the day's timestamps with a position.
Read per arm and seed: mean day with a day-block CI (95 %), without the best 5 % of days, periods held per entry,
entries. "Better" = H75 or H70 above H0 on the mean day AND without the best 5 % of days, on both seeds of the arm.
"Pays" = its CI above 0.
