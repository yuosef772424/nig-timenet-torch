# b1-btc-timing-001: buy BTC at the times the model is bullish across coins (owner 2026-11-19; exploratory, rule C5)

No training. Hypothesis B1 of `docs/results/btc-hedge-long-001.md`, pre-registered in
`docs/audit/2026-11-19-arch-indicators/PREREGISTERED_B1.md` (commit ed69c64, before any number); script
`b1_btc_timing.py` in the same folder; outputs on Drive `nig_timenet_runs/analysis-ind-001/b1/`. Tables: arm R of
`rolling-ind-001` (seeds 0, 1) and `rolling-single-002` (seeds 2, 3), 7 expanding folds, 7322 TEST timestamps and
1225 TEST days per seed (2023-01 .. 2026-06).

Rule: per fold, `long_share(t)` = share of the timestamp's rows with mu_close at or above the VAL 90th percentile;
buy BTC (8 h, 0.08 % per side) at every TEST timestamp whose long_share is at or above its VAL 80th percentile.

## Verdict

**FAILS the accept rule on all four seeds** (condition 1: CI contains 0; condition 2: negative without the best 5 % of
days). Condition 3 holds: it beats BTC always-long on the same days.

| seed | B1 mean day [CI 98.75 %] | without best 5 % | win days | per trade | BTC always (all days) | BTC at the other times | BTC always, B1's days |
|---|---|---|---|---|---|---|---|
| 0 | +0.019 % [-0.096, +0.142] | -0.161 % | 45.9 % | -0.033 % | -0.115 % | -0.195 % | -0.187 % |
| 1 | +0.028 % [-0.087, +0.150] | -0.139 % | 46.8 % | -0.010 % | -0.115 % | -0.204 % | -0.175 % |
| 2 | +0.070 % [-0.048, +0.188] | -0.097 % | 48.6 % | +0.005 % | -0.115 % | -0.198 % | -0.153 % |
| 3 | +0.014 % [-0.097, +0.122] | -0.142 % | 47.7 % | -0.062 % | -0.115 % | -0.190 % | -0.187 % |

Information only (not judged, listed in the pre-registration): L at the VAL 70th percentile is weaker
(-0.03 to -0.09 % per trade); L at the 90th percentile is +0.02 to +0.08 % per trade on all four seeds but its CI
contains 0 and it is negative without the best 5 % of days; the mean-mu signal and BTC's own mu do not do better.

## What the model learned

The cross-coin bullishness ranks BTC's next 8 h: at the selected timestamps BTC returns about +0.15 % gross per trade,
at the other timestamps about +0.02 % gross (net -0.13 to -0.14 %); the gap (about 0.13 % per trade, 0.2 % per day)
has the same sign on all four seeds. The selection is real; its size is about one round trip of cost.

## Where it is right and where it fails

Per fold, net per trade (seed 0 / 1 / 2 / 3): the first folds (2023) and fold 4 (2024-12 .. 2025-06) are mostly
positive and above BTC always; folds 5 and 6 (2025-06 .. 2026-06, a falling BTC: BTC always -0.19 / -0.21 % net per
trade) lose on every seed (-0.13 to -0.27 %), as much as or more than BTC always. The timing does not keep the book
out of a falling market.

## The samples it failed on

The losing days are in the last year of tests (folds 5-6) on every seed; a list of the worst days is in
`b1_seeds.csv` / `b1_folds.csv` on Drive (per fold), not yet broken down by day (next step if B1 is continued).

## Bias

Long only by construction. The signal fires on 19-20 % of TEST timestamps (1393-1439 of 7322), as the VAL level
intended: no drift of the firing rate between VAL and TEST.

## Selective use

One primary rule (judged), four information variants (listed before the numbers). No rule chosen on TEST.

## Why it failed

**Signal smaller than the cost:** the selection lifts BTC's 8 h gross return from about +0.045 % (all timestamps) to
about +0.15 %, which is about the 0.16 % round trip; net it is around 0 with a CI around 0. The same cost wall as
B-074 (`docs/results/panel-001.md`), now on one coin with the lowest spread.

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| B1a | Holding BTC while the signal stays on (one entry and one exit per run of signal timestamps, not a round trip every 8 h) keeps the gross and cuts the cost | the position book's mean day without the best 5 % is still below 0 on the 4 seeds | BTC's 4 h closes from `last_candles` of the artifact plus the saved signals; pre-register |
| B1b | A longer label horizon (24-48 h) raises the move per round trip above the cost (ROADMAP N3) | a ridge on the same inputs at 24-48 h shows a gross per round trip below 0.16 % | N3 of ROADMAP "Now" (model-free first) |
| B1c | A regime filter (BTC above its own 7-day mean, fitted on VAL) removes the falling-market folds 5-6 | the filtered book still loses in folds 5-6 on 3 of 4 seeds | saved tables plus BTC closes; pre-register |
