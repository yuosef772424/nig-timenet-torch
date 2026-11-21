# independent-review-001: an outside audit of the "skill but no profit after cost" reading (owner 2026-11-21; exploratory, C5)

Owner request 2026-11-21: audit the assumptions, the model and the evaluation; look for code errors or a bias in the
Claude readings that say (a) "the model has skill but does not pay after fees" and (b) "without the best 5 % of days the
long book loses: the profit lives in a few days". Data: the 28 tables of the latest control arm on Drive,
`nig_timenet_runs/rolling-dw1-r-001/out_s{0,1}/fold_00k_seed_s/battery_table_{test,val}.parquet` (arm R: single-coin
GRU, art1, 101 coins, 7 expanding folds, TEST 2023-01-07 .. 2026-06-14, 654,487 TEST rows per seed). Scripts and their
full output: `docs/audit/2026-11-21-independent-review/` (`a1` .. `a6`, `OUT_*.txt`). Cost 0.08 % per side. "Long
book" = the rows with `mu >= VAL 90th percentile of mu` of the fold, long (the book of `dw1_read.py`).

Failure reading: not applicable (an audit of the evaluation and of earlier readings, no new arm was trained).

Nothing here is pre-registered except what a1 reproduces; the breadth gate of a4 is fitted on VAL only, but it is one
more variant. Two seeds of one arm. Slippage and funding are not modelled.

## Verdict

1. **The recorded numbers are right.** a1 reproduces `dw1-001.md` to the third decimal: mean day +0.173 / +0.215 %,
   without the best 5 % of days -0.072 / -0.022 %, always long -0.147 %. No duplicate (coin, time) rows, folds do not
   overlap, `y_ret(t) = last_close(t + 8 h) / last_close(t) - 1` exactly (no shift, no look-ahead in the label).
2. **(b) is mostly an artefact of a one-sided trim.** Dropping only the best 5 % of days makes ANY long book on crypto
   negative: always-long goes -0.147 -> -0.279 %, BTC -0.115 -> -0.220 %. Trimmed on BOTH sides (best and worst 5 %),
   the model book stays positive: +0.169 / +0.231 % a day; its median day is +0.144 / +0.208 %. The verdict condition
   `mean_wo_best > 0` (`eval/battery_scorecard.py::scorecard_verdict`, also `dw1_read.py`) rejects positive skew, not
   a lack of edge, and no benchmark is put through the same test.
3. **(a) is a sizing / accounting result, not a property of the signal.** The "mean day" is an equal-weight mean of
   the TRADES of a day; it is not the return of a capital account. Put into a real account (two 8 h sub-books, half the
   capital each, compounded, fee on the actual turnover):
   - one equal share of the book per timestamp (all capital in 1 coin when 1 coin is picked): x0.01 / x0.06, the
     account is ruined. Per timestamp the book LOSES (-0.15 % per 8 h); the per-trade +0.13 % comes only from the
     752 timestamps with 31+ picks (+0.225 %), the 85 % of timestamps with 1-30 picks lose (a3, `OUT_a3`).
   - a FIXED slot per coin (1 / 101 of the sub-book per pick, the rest in cash): **CAGR +13.6 / +14.3 %, Sharpe 0.92 /
     0.94, max drawdown -19.7 / -20.0 %**, years +15 +32 +5 -3 % / +16 +42 -0 -3 % (2026 = Jan-Jun), after fees
     (a4). Always long on the same slots: -6.0 %, drawdown -78.5 %.
   - the same with a breadth gate fitted on VAL (trade only when the share of coins above the threshold is at least
     the VAL-best level): +17.8 / +19.4 %, Sharpe 1.19 / 1.26, drawdown -18.3 / -17.0 %, every year positive on both
     seeds (a4).
   - fee sensitivity (fixed slots): 0.02 % per side +24.6 / +25.5 %, 0.05 % +19.0 / +19.8 %, 0.10 % +10.2 / +10.7 %,
     0.15 % +2.0 / +2.4 % (a5). It survives the current fee; it dies near 0.15 % per side.
4. **The money is market TIMING, not coin choice.** A placebo that buys RANDOM coins at the same timestamps and the same
   count earns the same: +13.1 .. +14.5 % CAGR, Sharpe 0.92 - 1.02 (a5); on the day metric the model beats the
   placebo in 57 % (seed 0) / 99.5 % (seed 1) of 200 draws (a2). Using the breadth (share of coins with mu above the
   VAL q90) as the exposure of the equal-weight basket gives +18.0 / +17.9 % CAGR, Sharpe 1.12 / 1.09, drawdown -19 %,
   at 10 % mean exposure (a6). This agrees with the sceptic reading that the edge is timing, and disagrees with "no
   profit after fees": timing is an edge, and hedging it against the market (the "minus the market" columns) removes it
   by construction.
5. **What it does NOT beat:** BTC buy-and-hold over the same 3.5 years (+45.5 % CAGR, Sharpe 1.34, drawdown -52.5 %).
   Timing BTC with the breadth is worse than holding it (Sharpe 0.87 / 0.94): the breadth signal helps the alt basket
   (it sits out the alt bear of 2025-2026) and not BTC.

## Errors and biases found

| # | where | what | effect |
|---|---|---|---|
| E1 | `scorecard_verdict`, `dw1_read.py`, every "w/o best 5 %" rule | one-sided trim, no benchmark under the same test | rejects any positively skewed long book; buy-and-hold fails it too |
| E2 | `day_profile` and every "mean day" | equal-weight mean of a day's trades, not an account return; no position sizing, no compounding, days with no trade dropped | the same signal reads +0.17 %/day, ruins an account at 1 share per timestamp and earns +14 %/y at fixed slots |
| E3 | `battery_lib.round_trip` in all books | a full round trip on every 8 h row, even when the coin is kept | over-counts fees; small here (a3: -76 -> -74 %), larger for hysteresis rules |
| E4 | "minus the equal-weight market / BTC" columns | subtracts the market move on the days the book trades | removes timing alpha by construction; it tests coin choice only |
| E5 | `battery_lib._scored` (TEST) | drops rows with `dir_true == 0`, chosen by the realised outcome | 16,208 of 654,487 TEST rows (2.5 %, mostly coarse-tick coins: CVC, CTK, LIT); small optimistic bias in every battery number |
| E6 | Bonferroni over 27 strongly correlated rows (99.81 %) | very conservative | a real but small edge cannot pass on ~1,200 days |
| E7 | `rank_ic_by_timestamp` (already found in `m3-improve-local-001.md`) | 2-row timestamps give +-1 | noisy skill metric |

No look-ahead was found in the label, the fold cuts, the VAL thresholds or the `top_coins` choice (ranked on TRAIN
rows). Delisted coins are present (18 of 101 end > 30 days before the TEST end), so the coin set is not survivor-only.

## What to change (proposals, each a switch + a test)

1. **Evaluate in a capital account.** Add an `account` section to the battery: fixed-slot sizing, cash for unused
   slots, fee on turnover, compounding, CAGR / Sharpe / drawdown / per-year, and the same numbers for a timing placebo,
   always-long and BTC. Judge rows by the account, not by the mean day.
2. **Replace the one-sided trim** by a symmetric trim or the median day, and print the benchmark's value beside it.
3. **Make timing the target.** The signal that pays is how many coins the model is bullish on at once. Train or read a
   market head (next 8 h - 3 d move of the equal-weight basket) and trade the basket / a few liquid coins with exposure
   = the forecast; fewer instruments and less turnover attack the 0.16 % wall.
4. **Coin choice needs a lower cost per unit of edge**: M3's residual IC (0.09) is the best coin-level signal so far
   but pays 0.03 - 0.05 % per 8 h; hold positions (hysteresis, already listed as C2), longer horizons, or maker fees.
5. **Long only** until a short rule passes: the short pays the convexity (exp() part) and fights the drift
   (`short-side-001.md`).
6. Before any real money: a third and fourth seed, slippage on alts, funding of perpetuals, and the breadth gate
   pre-registered on a fresh walk-forward.
