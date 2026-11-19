# btc-hedge-long-001: the long book minus BTC (owner 2026-11-19, the free test; exploratory, rule C5)

No training. The saved TEST tables of arm R of `rolling-ind-001` (seeds 0, 1; single-coin GRU 64, 3 channels, lr 3e-4,
commit 458bcdb) and of `rolling-single-002` (seeds 2, 3; the same configuration, commit cc3b011), 7 expanding folds
each, about 1200 test days per seed. Rule written before the numbers: amendment 2 of
`docs/audit/2026-11-19-arch-indicators/PREREGISTERED.md`. Script `docs/audit/2026-11-19-arch-indicators/read_arms.py`
(commit 4db5a9d); outputs on Drive `nig_timenet_runs/analysis-ind-001/btc_hedge/`.

The long book: per fold, buy every TEST row whose mu_close is at or above the 90th percentile of mu_close on that
fold's VAL table; hold to the label horizon (8 h); cost 0.08 % per side. BTC hedge: the BTCUSDT row of the same
timestamp (every trade had one).

## Verdict

**FAILS: there is no edge beyond BTC.** The BTC-hedged book, with the BTC short's own round trip charged, loses on
every seed, with a CI below 0 (98.75 %):

| arm | seed | long book, mean day | BTC long at the same times | minus BTC, 1 cost | minus BTC, 2 costs [CI] | without best 5 % days | beta to BTC | beta-hedged, per trade |
|---|---|---|---|---|---|---|---|---|
| R | 0 | +0.173 % | +0.036 % | -0.024 % | -0.184 % [-0.296, -0.076] | -0.384 % | 1.35 | -0.293 % |
| R | 1 | +0.215 % | +0.025 % | +0.030 % | -0.130 % [-0.253, -0.011] | -0.325 % | 1.35 | -0.303 % |
| ref002 | 2 | +0.218 % | +0.047 % | +0.011 % | -0.149 % [-0.274, -0.024] | -0.358 % | 1.31 | -0.291 % |
| ref002 | 3 | +0.158 % | +0.024 % | -0.026 % | -0.186 % [-0.299, -0.079] | -0.353 % | 1.34 | -0.261 % |

Winning days of the realistic hedged book: 41-44 %.

## What the model learned

- WHEN to be long: at the times the model picks, BTC itself returns about +0.19 % gross over 8 h (+0.02 to +0.05 %
  net of one round trip), and the chosen coins about 0.16 % more than BTC, gross.
- The chosen coins move 1.31-1.35 x BTC: the book is a leveraged bet on the market's direction at those times.

## Where it is right and where it fails

Right on the market's direction at the selected timestamps (BTC long at those times is positive net on all four
seeds, small). Wrong as a coin picker: once BTC is removed, what is left (+0.16 % gross per trade over BTC) is about
one round trip of cost and does not pay a second one; removed with the measured beta (1.35) it is -0.26 to -0.30 %
per trade. The per-fold reading of `docs/results/rolling-single-002.md` (fold 3, the late-2024 rally, carries the
long book) is the same fact seen from the market side.

## The samples it failed on

The hedged book's losses are the trades in market drops: a beta of 1.35 makes the chosen coins fall more than BTC.
The worst unhedged days (`rolling-single-002`, "The samples it failed on") are 8 h pumps on the short side and
market-wide drops on the long side. A row table of the worst hedged days is not made yet (it needs the day table of
`book_days`, saved by the script: next step if this reading is used).

## Bias

The book is long only by construction; its bias is the market beta (1.31-1.35), on every seed.

## Selective use

The rule is the pre-registered top 10 % of mu_close (threshold fitted on VAL). No other rule was tried on the hedged
book (one variant per seed; CI level for 4 seeds read).

## Why it failed

**Read but biased toward the market: the long edge is the market's direction at the chosen times, levered by
high-beta coins, not a coin-level edge.** Evidence: beta 1.35 on all four seeds; minus BTC the gross is one cost;
with the second cost and with the beta it is negative with a CI below 0; `long_hedged` (minus the equal-weight market)
is negative too (-0.11 to -0.16 % per day).

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| B1 | The model's long signals time BTC itself: buy BTC (one leg, lowest cost) at the timestamps where many coins are in the long book | the BTC-at-signal book has a day CI containing 0, or is negative without the best 5 % of days, on the 4 seeds | the saved tables: group the long trades by timestamp, trade BTC when the share of long rows exceeds a VAL-fitted level (pre-register first) |
| B2 | A low-beta long book (top 10 % of mu among coins with beta to BTC below 1 on VAL) keeps the timing without the leverage of the losses | its mean day without the best 5 % is still below 0 | the saved tables plus a per-coin beta fitted on each fold's VAL |
| B3 | The rank term (arm IR of `rolling-ind-001`) adds a coin-level edge: its BTC-hedged book beats R's | IR's BTC-hedged mean day is not above R's on both seeds | the same script on `out_IR_s{0,1}` when the arm finishes |
