# short-side-001: why the "top short" rows lose in every run (exploratory, rule C5)

Owner question 2026-11-19: "why was top short a loser in all the runs? Check the loss function: is the short loss
computed exponentially?" Read from the saved prediction tables of `rolling-single-001` (the single-coin model, 7
expanding folds x seeds 0 and 1, 101 coins, 1225 pooled TEST days per seed, 2023-01-07 .. 2026-06-14), downloaded
from Drive `nig_timenet_runs/rolling-single-001/fold_00k_seed_s/out/battery_table_test.parquet`. Scripts and their
full output: `docs/audit/2026-11-19-short-side/` (`OUTPUT.txt`). Cost 0.0008 per side (0.16 % per round trip).
"Top x % |mu|" = rows whose |predicted close move| is above the VAL quantile; the side is the close class head
(`p_dir > 0.5` = long), as in the battery's trading grid. Thresholds of fold k come from its VAL table for folds 0-1
and from the TEST table of fold k-1 otherwise: VAL tables above 10 MB cannot be fetched through the Drive connector.
That earlier period is never the rows being judged.

## Answer in short

1. **The training loss is not exponential, and it is symmetric in up and down.** The close head is an NIG regression
   on `y = log(future close / last close)` (codec `logret_close`, artifact `c572b1b36960`, clip |y| <= 1, B-054):
   `trainer/tasks.py::evidential_task_loss` uses `(y - mu)^2` inside a Student-t NLL plus `|y - mu|` x evidence. The
   class head is a plain BCE on the sign. No term weights a row by its sign (`direction_balance` is off by default and
   symmetric). Over-predicting and under-predicting a log move cost the same.
2. **The P&L is exponential, and that is correct.** The battery decodes the label back to price
   (`eval/battery_adapter.py::_head`, `y_ret = exp(y) - 1`) and a short is paid `-y_ret = 1 - e^y`. A rise of +l costs
   `e^l - 1 > l`; an equal log fall gains only `1 - e^-l < l`. This is not a bug: it is what a short of a fixed
   notional earns. The mismatch is that the model is trained in a currency (log) that hides it.
3. **What it costs the short, measured.** Split each trade into the part a log forecast is paid (`side x log(1+y)`)
   and the exp() part (`side x (y - log(1+y))`): on the pooled walk-forward the short top 5 % |mu| rows are RIGHT in log
   units (+0.280 % / +0.117 % per trade, seeds 0 / 1), the exp() part takes -0.145 % / -0.152 %, the round trip takes
   0.16 %, and the net is -0.026 % / -0.195 %. The long top 5 % gets the exp() part as a GIFT (+0.103 % / +0.108 %). The
   model's own sigma predicts this part (-0.130 % / -0.140 % from `s^2 / 2`), so it is knowable before the trade.
4. **Why it hits the TOP rows hardest.** |mu| ranks volatility: the short top 5 % rows have a realised sd of 5.5-5.6 %
   per 8 h against 3.0 % on all rows, and the exp() part grows as `s^2 / 2`. In the most uncertain sigma quintile the
   short pays -0.10 % per trade for exp() against -0.013 % in the calmest.
5. **And the direction itself is a market call.** Hedging each trade with the equal-weight market move of its
   timestamp turns BOTH sides negative (top 10 %: long +0.27 % -> -0.32 %, short +0.01 % / -0.09 % -> -0.27 % / -0.26 %).
   The short side's day P&L has a correlation of -0.44 / -0.41 with the market day: +2.3 % a day in the most falling
   fifth of days, -1.37 % in the most rising fifth.

## Confirmation with the real VAL thresholds (re-read on Colab, 2026-11-19)

`battery_from_folder` was re-run on the 14 fold folders on Colab, with each fold's own VAL table, so it prints the new
sections. The reports are in `rolling-single-001/out/fold_*/ss1/battery_report.txt`. "LONG vs SHORT PAYOFF" rows, top
5 % |mu| per fold and seed:

| fold | SHORT win % | SHORT log % | SHORT exp % (pred) | SHORT net % | SHORT L/W | LONG exp % | LONG net % |
|---|---|---|---|---|---|---|---|
| 0 s0 / s1 | 48.5 / 43.9 | +0.231 / -0.108 | -0.089 (-0.094) / -0.087 (-0.100) | -0.018 / -0.355 | 0.95 / 1.02 | +0.049 / +0.073 | +0.169 / +0.291 |
| 1 s0 / s1 | 45.0 / 51.0 | -0.065 / +0.123 | -0.081 (-0.092) / -0.117 (-0.110) | -0.306 / -0.154 | 1.05 / 1.15 | +0.071 / +0.061 | +0.421 / +0.201 |
| 2 s0 / s1 | 53.8 / 50.6 | +0.354 / +0.088 | -0.123 / -0.106 | +0.071 / -0.178 | 1.12 / 1.15 | +0.084 / +0.107 | +0.166 / +0.024 |
| 3 s0 / s1 | 55.6 / 55.5 | +0.500 / +0.369 | -0.137 / -0.140 | +0.203 / +0.069 | 1.12 / 1.20 | +0.156 / +0.150 | +1.642 / +1.856 |
| 4 s0 / s1 | 56.3 / 51.5 | +0.609 / +0.368 | -0.186 / -0.157 | +0.263 / +0.050 | 1.12 / 1.03 | +0.151 / +0.130 | +0.607 / +0.323 |
| 5 s0 / s1 | 56.6 / 57.9 | +0.131 / +0.206 | -0.273 (-0.188) / -0.274 (-0.213) | -0.303 / -0.229 | 1.47 / 1.51 | +0.098 / +0.069 | +0.454 / +0.562 |
| 6 s0 / s1 | 61.0 / 60.7 | +0.383 / +0.529 | -0.289 (-0.180) / -0.273 (-0.169) | -0.066 / +0.096 | 1.61 / 1.49 | +0.088 / +0.083 | -0.148 / -0.024 |

- The short's exp() part is negative in 14 of 14 runs (-0.08 to -0.29 %), the long's positive in 14 of 14. The
  short's log part is positive in 12 of 14.
- Folds 5 and 6 (2025-06 .. 2026-06) are the clearest case of "wins more often and still loses": the short wins 57-61 %
  of its trades with losses 1.5-1.6 x its wins. There the realised exp() part (-0.27 .. -0.29 %) is larger than the
  model's sigma predicts (-0.17 .. -0.21 %), so the tails are heavier than the predictive distribution.
- The best scorecard row of each fold varies: `top 20 % confidence | LONG` is best in 3 of the first 4 folds, and 5 of
  14 folds pass the verdict. Each pass is the best of 27 rows on ~175 days, so it counts only as a look; the
  pre-registered test is `rolling-single-002`.

## What the model learned

- **Direction in log units, on both sides of the top rows.** Log part per trade, top 10 % |mu|: long +0.352 % /
  +0.339 %, short +0.284 % / +0.179 % (seeds 0 / 1); positive for the short in 11 of 14 fold x seed runs. The trivial
  baseline (no skill) scores 0 on this part (known-answer test
  `tests/eval/test_battery_scorecard.py::test_asymmetry_known_answer_short_pays_half_the_variance`).
- **Confidence ranks the hit rate.** Close-head confidence quintiles (VAL edges): hit 0.502 -> 0.542 (seed 0) and
  0.501 -> 0.536 (seed 1), monotone.
- **Sigma ranks the size of the move and the exp() cost.** Mean |y| 1.15 % -> 2.98 % from the calmest to the most
  uncertain quintile; the short's exp() cost -0.014 % -> -0.103 % in the same order; the predicted exp() part matches
  the realised one within 10-20 % on every selection.

## Where it is right and where it fails

Pooled walk-forward, net per trade (seed 0 / seed 1):

| selection | LONG net | SHORT net | SHORT log part | SHORT exp() part | SHORT loss / win |
|---|---|---|---|---|---|
| all rows | -0.089 / -0.089 | -0.106 / -0.113 | +0.099 / +0.091 | -0.046 / -0.044 | 1.06 / 1.06 |
| top 5 % abs(mu) | +0.459 / +0.509 | -0.026 / -0.195 | +0.280 / +0.117 | -0.145 / -0.152 | 1.13 / 1.14 |
| top 10 % abs(mu) | +0.274 / +0.266 | +0.011 / -0.090 | +0.284 / +0.179 | -0.112 / -0.109 | 1.13 / 1.12 |
| top 20 % abs(mu) | +0.121 / +0.119 | -0.043 / -0.069 | +0.202 / +0.170 | -0.085 / -0.079 | 1.12 / 1.12 |
| top 20 % confidence | +0.099 / +0.010 | -0.037 / -0.087 | +0.178 / +0.130 | -0.055 / -0.058 | 1.01 / 1.05 |
| low 20 % sigma | -0.146 / -0.146 | -0.162 / -0.149 | +0.012 / +0.024 | -0.014 / -0.013 | 1.08 / 1.07 |

- Per fold (top 10 %): the short's exp() part is negative in 14 of 14 runs (-0.069 .. -0.200 %), the long's positive in
  14 of 14 (+0.050 .. +0.148 %); the short net is positive in 5 of 14, the long net in 12 of 14.
- The long side's pooled "profit" lives in one regime: fold 3 (2024-06-30 .. 2024-12-21, the late-2024 rally) gives
  +1.47 % / +1.36 % per trade; fold 6 is negative on both seeds. Hedged against the market it is -0.32 %.
- Low-sigma rows: hit rate 0.50, moves (1.15 %) too small for the 0.16 % cost on both sides: the cost wall (B-074).

## The samples it failed on

The 15 worst short top-10 % trades per seed (`OUTPUT.txt`, "worst short trades") are all 8 h pumps of +40 % to +88 %:
DENTUSDT 2026-02-26 04:00 (+88.5 %, twice in a row), DUSKUSDT 2026-01-19 (+64 %), TRXUSDT 2024-12-03 (+68 %),
ATAUSDT 2024-12-28 (+61 %), FILUSDT 2025-11-07 (+58 %), CTSIUSDT, API3USDT, RLCUSDT, GTCUSDT, ALPHAUSDT, EGLDUSDT.
What they share: the model's sigma rank is high (20 of the 30 rows of both seeds above the 0.88 quantile), the predicted move is small
(mu -0.3 .. -1.0 %) and the confidence is LOW (p_dir 0.37-0.49, mostly 0.42-0.47). Trades below -10 %: 2.6 % of the
short top-10 % trades; the worst 1 % of trades hold 16-17 % of all short losses. This is the squeeze pattern of
`docs/audit/2026-11-17-failure-charts/` (B-077) on 14 more runs.

## Bias

- Share of rows the class head calls short per fold: 0.37 .. 0.57 (seed 0) and 0.41 .. 0.56 (seed 1) against a realised
  share of falling rows of 0.48 .. 0.52: the lean changes sign from fold to fold and seed to seed; it is a market-level
  bet learned from the training window, not a stable preference for one side.
- Which side carries the loss: the short, on rising-market days (above) and on CROWDED days. The quarter of days with
  the most short top-10 % trades holds 71-74 % of those trades and its mean day is -0.07 % / -0.08 %; the quarter with
  the fewest trades (1-3 a day) has +1.0 % per day. An equal-weight day mean therefore reads the short side as
  +0.30 % a day with a CI above 0 while the trades lose: the scorecard now prints the busy-day mean beside it.
- A VAL-fitted correction was not run here (no VAL table for folds 2-6); see the next hypotheses.

## Selective use

Rules written before this reading (the selections of `eval/battery_scorecard.py`, VAL thresholds), 27 rows x 2 seeds,
CI 99.81 % (Bonferroni over 27): no row passes the scorecard verdict (mean-day CI above 0, winning days above 50 %,
positive without the best 5 % of days AND on busy days).

| row (seed 0 / seed 1) | mean day [CI] | winning days | w/o best 5 % days | busy days |
|---|---|---|---|---|
| short top 10 % abs(mu) | +0.301 [+0.060, +0.604] / +0.292 [+0.064, +0.572] | 55.9 / 55.0 % | -0.073 / -0.059 | -0.010 / -0.032 |
| long top 5 % abs(mu) | +0.208 [-0.124, +0.554] / +0.231 [-0.069, +0.621] | 53.5 / 53.1 % | -0.222 / -0.219 | +0.193 / +0.186 |
| long top 20 % confidence | +0.195 [+0.002, +0.420] / +0.200 [-0.035, +0.380] | 56.1 / 56.1 % | -0.037 / -0.035 | +0.101 / +0.131 |
| both sides, all rows | -0.096 [-0.157, -0.041] / -0.100 [-0.166, -0.049] | 42.4 / 43.9 % | -0.181 / -0.181 | -0.090 / -0.097 |

Two more variants were looked at on TEST and count as variants (hypotheses only): the mean-corrected rule (trade only
when the predicted log move beats `s^2/2` plus cost: long +0.010 % / +0.032 %, short -0.053 % / -0.050 % per trade) and
a stop at 1-3 x sigma on the realised high (short top 10 % -0.002 .. -0.124 %: it cuts the worst 1 % from -25 % to
-10 % but not the mean).

## Why it failed

**Read, but paid in a skewed currency.** The model is trained on a log target with a symmetric loss and it does call
the direction of its top rows in log units, but a short is paid `1 - e^l`. On the volatile rows that a top-|mu| rule
picks, the exp() part (-0.11 .. -0.15 % per trade, predicted by the model's own sigma) plus the round trip (0.16 %)
is larger than the log edge (+0.12 .. +0.28 %); the long side receives the same exp() part as a gain. Evidence: the
log / exp split above (all 14 runs), the known-answer test (no skill: short -s^2/2, long +s^2/2), and the matching
predicted exp() part. Second, the direction of the top rows is mostly the market's: market-hedged, both sides lose.

## Next hypotheses

| # | Hypothesis | Measurement that refutes it | Cheapest test |
|---|---|---|---|
| H1 | A side-aware decision in PRICE units (score = predicted log move + side x s^2/2, a higher bar for shorts fitted on VAL) keeps the long top rows and drops the shorts that only pay exp() | on the next walk-forward with VAL tables, the VAL-fitted per-side thresholds give a TEST short net <= 0 and a both-sides net no better than "long only" on both seeds | rerun `battery_from_folder` (it now prints the asymmetry section) on runs whose VAL tables are reachable; then a rule fitted on VAL |
| H2 | The direction of the top rows is a market call; a market-relative target (the panel's relative inputs + rank term) is the edge that survives hedging | the hedged top-10 % net stays below -0.2 % for the panel model's relative score on the same folds | apply `hedge_and_stop.py` to the panel walk-forward tables (`panel-rolling-ref-003`) |
| H3 | Shorts on the most uncertain fifth (sigma q5) and low-confidence shorts carry the squeezes; excluding them for the SHORT side only removes most of the tail without removing the log edge | the short net of sigma q1-q4 x confidence q3-q5 is not above the short net of all rows on TEST, both seeds | a VAL-fitted filter on saved tables, counted as a variant |
| H4 | Training the close head on the simple return (or adding `s^2/2` to the decoded mean) makes mu rank the price payoff instead of the log move | the IC of mu with the simple return does not rise on E1 / the real slice | a switch on the target codec, E1 first (protocol gates) |

What is kept: the battery now prints the day scorecard, the uncertainty buckets and this log / exp split for every
run (`eval/battery_scorecard.py`), so the question is answered by every future report and by `battery_from_folder`
on old runs.
