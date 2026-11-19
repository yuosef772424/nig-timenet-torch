# Pre-registration: architecture and size, indicator channels, rank loss, lr 1e-5 (owner task 2026-11-19)

Written before any number of the runs below is read. Exploratory (rule C5).

## The owner's request

Choose the best architecture and number of units; train with more than one loss (the rank loss and the others); add
RSI and the stochastic oscillator (and volume, "if useful", owner follow-up); learning rate 1e-5; launch training.
One of the models will be tested in live trading, LONG ONLY, because the long side showed skill
(`docs/results/rolling-single-002.md`: long top 10 % |mu| had a day CI above 0 on seeds 2 and 3, not pre-registered,
negative without the best 5 % of days). Charts of a small sample saved to Drive.

## Data

`/content/art_ind` = the artifact `c572b1b36960` (535 coins, 4h, 32-bar windows) plus four channels per bar from
`data_io/indicator_channels.py`: `rsi_14`, `stoch_k_14_3`, `stoch_d_3`, `volume_rel_30` (Binance futures 4h candles,
100 bars of warm-up; a window with a missing bar or a close that disagrees with the artifact gets zeros, counted in
`derivation_indicator_channels.json`).

## Losses of every arm except R

The NIG NLL of close / high / low (with the evidence and calibration terms already in the run default) + the three
binary class heads (BCE, Kendall weights) + the cross-section rank term `cross_section_loss: rank_ic` (weight 1.0,
1 - the correlation inside each timestamp of every NIG mean with the rank of its label, D-032). The rank term needs
whole timestamps in a batch: `batch_order: time_blocks`, `time_block_bars: 8` (about 400-500 rows per batch with 100
coins and the row cap). Known cost (`docs/results/batch-by-wf-002.md`): time-block batches alone worsened the test
raw loss; arm I (below) separates the channels from the loss recipe.

## Stage 1: architecture and size look (`arch-ind-look-001`)

Arms (all: `art_ind`, the loss recipe above, lr 1e-5, 10 epochs, patience 3, 200k train rows, top 100 coins, the
first 2 walk-forward folds (test 2023-01 .. 2023-12), seed 0):

| arm | encoder | d_model | num_layers |
|---|---|---|---|
| g64 | gru | 64 | 2 (the current model) |
| g128 | gru | 128 | 2 |
| g128l3 | gru | 128 | 3 |
| tcn64 | tcn | 64 | 2 |
| tr64 | transformer | 64 | 2 |
| ct64 | conv_transformer | 64 | 2 |

Choice rule (VAL only; TEST of these folds is not read for the choice): the arm with the highest mean over the 2
folds of the VAL long book (below) net per trade; an arm within 0.02 % per trade of the best and smaller is preferred
(fewer parameters). The VAL close rank IC by timestamp is reported next to it. One seed: a look, not a result (C4).

## Stage 2: walk-forward (`rolling-ind-001`)

7 expanding folds (as `rolling-single-002`), 400k train rows, top 100 coins, seeds 0 and 1, the chosen architecture:

| arm | channels | losses / batches | lr | epochs (patience) |
|---|---|---|---|---|
| R | 3 (art1) | NIG + class, random rows of 512 | 3e-4 | 8 (3): the rolling-single configuration |
| I | 7 (art_ind) | NIG + class, random rows of 512 | 1e-5 | 16 (4) |
| IR (candidate) | 7 (art_ind) | NIG + class + rank term, time blocks of 8 | 1e-5 | 16 (4) |

## The long book (the use the owner will trade)

Per fold and seed: the threshold q = the 90th percentile of mu_close on that fold's VAL table; on TEST, buy every row
with mu_close >= q, hold to the label horizon; net per trade = exp(y_ret) - 1 - 0.0016 (0.0008 per side). The day
book = the mean net of the day's trades; days are pooled over the 7 folds per seed. Also reported: the same book minus
the equal-weight market move of the same timestamp (all rows of that timestamp) = the market-hedged long (hypothesis
N1 of rolling-single-002: is the edge only market beta?), and always-long on all rows as the trivial baseline.

## Accept rule for the candidate IR (all must hold on BOTH seeds)

1. The mean day net of the long book has a day-block bootstrap CI above 0 at level 1 - 0.05 / 3 (three arms).
2. The mean day net without the best 5 % of days is above 0.
3. The mean day net is above that of R and above always-long on the same days.

If IR fails, the failure reading of rule C7 is written before any next arm. Arm I is read the same way (it shows
what the channels alone do). The live-trading candidate is whichever arm passes; if none passes, none is
recommended for live money and the note says so.

## Amendment 1 (owner 2026-11-19, before any number of the arms was read)

Every reading separates buys from sells: the long book above is read alone; a short book (the 10th percentile of
mu on VAL, sell every row at or below it, net = -y_ret - 0.0016) is reported in its own columns, information only.
No number mixes the two sides. Reading `rolling-single-002` with this script (a reference, already read in its own
note) gave: long top 10 % mean day +0.218 % / +0.158 % (CI above 0, seeds 2 / 3), but the long book minus the
equal-weight market move of the same timestamp is -0.125 % / -0.156 % per day: the long trades sit at timestamps
where the whole market rose (+0.30 % on average over the horizon), i.e. the model times the market, it does not pick
coins. Both readings are added to the accept rule as information; the rule itself is unchanged.

## Amendment 2 (owner 2026-11-19: "the free test first: re-read the current long trades minus the BTC return")

Written before the BTC-hedged numbers are computed. On the saved TEST tables of arm R (seeds 0, 1) and of
`rolling-single-002` (seeds 2, 3), the long book above is re-read with BTC as the hedge: per trade
`y_ret - y_ret(BTCUSDT, same end_time) - 0.0016` (one hedge leg, no extra cost) and the same minus a second 0.0016
(the BTC short's own round trip, the realistic cost of a hedged trade). Trades whose timestamp has no BTCUSDT row
are dropped and counted. Accept "an edge beyond BTC": the realistic BTC-hedged mean day net has a day-block CI above 0
(level 1 - 0.05 / 4, four seeds read) on every seed AND stays above 0 without the best 5 % of days. If it fails, the
long edge is BTC / market direction (timing), not coin choice. Also reported: beta to BTC of the long book's trade
returns (OLS slope) and the BTC-hedged book with that beta.
