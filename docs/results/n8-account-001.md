# n8-account-001: every saved run on Drive read as a capital account (N8, owner 2026-11-22; exploratory, C5)

Owner request 2026-11-22: test the N8 account section on the rolling-dw1-r-001 tables and on the other models on
Google Drive. Code: `eval/account.py` at ecfc710 (PR #78), run on a Colab CPU runtime through the bridge.
Scan: `docs/audit/2026-11-22-n8-account/scan.py` over `MyDrive/nig_timenet_runs` (95 groups = one run folder and one
seed, its folds in ONE account, each fold cut at its own VAL q90 of mu); every number in
`docs/audit/2026-11-22-n8-account/scan_summary.tsv`; the full JSON (rows, stats, years) on Drive in
`nig_timenet_runs/_n8_account_scan/`. Cost 0.08 % per side on the turnover, fixed slots (1 / coins of TEST), 2
sub-books (8 h label on 4 h rows), 20 placebo draws. Nothing here is pre-registered: it is a re-reading of runs whose
arms were already read; 95 groups are many looks (best-of-N), so no single row is a result.

Failure reading: not applicable (a re-reading of saved runs through a new measurement, no new arm was trained).

## 1. The section reproduces the audit exactly (rolling-dw1-r-001)

| row | seed 0 | seed 1 | audit |
|---|---|---|---|
| model long book, 101 slots, turnover fee | CAGR +13.6 %, Sharpe 0.92, maxDD -19.7 % | +14.3 %, 0.94, -20.0 % | a4: the same |
| same, all capital on the picks | -73.6 % | -53.1 % | a3: the same |
| fee 0.02 / 0.05 / 0.10 / 0.15 % per side | +24.6 / +19.0 / +10.2 / +2.0 % | +25.5 / +19.8 / +10.7 / +2.4 % | a5: the same |
| basket x breadth (timing only) | +18.0 %, Sharpe 1.12, -19.1 % | +17.9 %, 1.09, -19.8 % | a6: the same |
| always long / BTCUSDT buy-and-hold | -6.0 % / +45.5 % (Sharpe 1.34, -52.5 %) | same | a4: the same |
| placebo (random coins, same timestamps and counts) | mean +13.9 %, p95 +14.5 %, book beats 15 % | +14.0 %, +14.5 %, 65 % | a5: +14.1..+14.5 |

The round trip every period (the battery's charge, E3) costs the book about 3 points a year (+10.5 / +11.2 %).
38 s per seed on the CPU runtime (654,487 TEST rows). Re-runs of one configuration give identical accounts
(rolling-single-001 = rolling-dw1-r-001 = rolling-ind-001 arm R; rolling-single-002 = rolling-r23-001 =
rolling-ls1-a0-001; panel-rolling-001 = ref-002 = ref-003).

## 2. All runs with the full 7-fold TEST window (2023-01 .. 2026-06, BTC +45.5 %, always long -6.0 %)

| family | runs (seed pairs) | CAGR | beats the placebo p95 on both seeds | basket x breadth |
|---|---|---|---|---|
| single-coin GRU (rolling-single, dw1, ind, indr, lr1, ls1, r23, btcrel) | 15 | -4.0 .. +17.4 % | none | +2 .. +5.5 points above the book, every run |
| panel, rank_ic loss (panel-rolling-001/ref, 2tf, ls, std) | 4 distinct | +14.7 .. +23.0 % | all 4 (100 % of draws, both seeds) | +0.5 .. +1.8 points above the book |
| panel snapshot base | 1 | +16.5 / +15.9 % | yes (100 %) | +0.3 / +0.5 points |
| panel base, usdt, snapshot hybrid / joint | 4 | -2.1 .. +12.6 % | on one seed only, or with a near-zero / negative book | above the book (except hybrid seed 1) |

Best: panel-rolling-std-001, +23.0 / +18.9 %, Sharpe 1.39 / 1.21, max drawdown -18.0 / -19.4 %, years +23 +56 +8 -2 % /
+27 +38 +6 -2 % (2026 = Jan-Jun);
placebo p95 +20.6 / +16.5 %. Nothing beats BTC buy-and-hold on CAGR (+45.5 %), at a third of its drawdown
(-52.5 %). Runs on other windows (6 folds, one fold, smoke) are in the TSV and not compared here.

## 3. Coin choice or lower turnover? The same books at zero cost

The placebo re-draws its coins every period and so pays more turnover than a book that keeps its coins; at 0 cost
that advantage is gone:

| run | seed | model | placebo mean / p95 | beats | basket x breadth |
|---|---|---|---|---|---|
| rolling-dw1-r-001 | 0 / 1 | +28.5 / +29.5 % | +29.3 / +29.9, +29.6 / +30.1 % | 10 / 45 % | +32.6 / +32.7 % |
| panel-rolling-std-001 | 0 / 1 | +38.1 / +33.9 % | +35.5 / +36.4, +31.1 / +32.0 % | 100 / 100 % | +39.6 / +34.8 % |
| panel-rolling-ls-001 | 0 / 1 | +34.3 / +28.6 % | +31.9 / +32.7, +25.7 / +26.6 % | 100 / 100 % | +35.8 / +28.6 % |
| snapshot-rolling-base-001 | 0 / 1 | +31.9 / +30.8 % | +28.6 / +29.9, +27.8 / +28.9 % | 100 / 100 % | +32.0 / +30.9 % |

So the panel models' coin choice is real before cost (+2.5 .. +3 points a year over random coins at the same
timing, both seeds), the single-coin model's is not (the audit's "timing, not coin choice" holds for it). And the
equal-weight basket held at the book's breadth beats every book, with or without cost: a few concentrated coins
compound worse than the whole basket (volatility drag) and pay more turnover, which eats the coin-choice edge.

## 4. Is the timing a skill, or the market the book happened to sit in? (`timing.py`, `OUT_timing.txt`)

The basket held at the book's breadth, against the SAME breadth series shifted in time by a random offset (keeps its
distribution and autocorrelation, breaks its link with the market; 50 draws) and against a constant exposure equal to
its mean (0.10), cost 0.08 % per side:

| run | seed | timing CAGR | constant 0.10 | shifted: mean / p95 | beats | timing IC by year 2023 / 24 / 25 / 26 |
|---|---|---|---|---|---|---|
| rolling-dw1-r-001 | 0 | +18.0 % | +1.1 % | -11.1 / -2.1 % | 100 % | +0.072 / +0.057 / +0.054 / +0.010 |
| rolling-dw1-r-001 | 1 | +17.9 % | +1.1 % | -10.0 / +1.8 % | 100 % | +0.102 / +0.079 / +0.045 / -0.013 |
| panel-rolling-std-001 | 0 | +24.2 % | +1.1 % | -10.4 / -0.6 % | 100 % | +0.089 / +0.099 / +0.054 / +0.024 |
| panel-rolling-std-001 | 1 | +20.0 % | +1.1 % | -9.6 / -1.3 % | 100 % | +0.109 / +0.073 / +0.043 / +0.034 |
| panel-rolling-ls-001 | 0 / 1 | +21.5 / +15.2 % | +1.1 % | p95 -2.6 / +1.1 % | 100 / 100 % | positive every year (2026: +0.005 / +0.020) |
| snapshot-rolling-base-001 | 0 / 1 | +16.8 / +16.4 % | +1.1 % | p95 -3.6 / -1.5 % | 100 / 100 % | positive every year (2026: +0.027 / +0.027) |

- The profit is NOT the market's direction: the market over TEST loses (always long -6.0 %), a constant 10 %
  exposure makes +1.1 %, and the same breadth at the wrong times loses about 10 % a year. The model's breadth says
  WHEN to hold the market, and it is right in every year on every seed: a timing skill of the model (IC +0.04 .. +0.11
  in 2023-2025).
- It weakens: the timing IC of 2026 (January-June, the last fold) is +0.00 .. +0.03 on every run (one seed -0.013).
- Concentration is that of crypto, not of the model: the best 5 % of days hold 200-320 % of the log growth of every
  book AND of BTC buy-and-hold (283 %); without them every book and BTC lose (-18 .. -25 % / -50 %). With the best AND
  worst 5 % of days removed every book stays positive (+14 .. +23 %), but so do always long (+24 %) and BTC (+28 %):
  that trim favours the most volatile series, it does not separate skill. What separates: months > 0 54-66 % for the
  books against 37 % always long and 54 % BTC; folds > 0 5-6 of 7 against 3 of 7 always long and 5 of 7 BTC.
- The book's own return, gross of cost, splits into the same exposure on the whole basket and coin choice: single-coin
  +28.5 = +29.2 - 0.7 and +29.5 = +29.5 + 0.0 points (no coin choice); panel std +38.1 = +35.4 + 2.7 and
  +33.9 = +31.1 + 2.8, ls +2.4 / +2.9, snapshot base +3.3 / +3.0 points (a small, real coin choice).

## 5. What it means for the next steps

- N8 is confirmed on real tables: the account section, run on any finished fold folders
  (`python -m nig_timenet_torch.workflow.battery_run <folders>`), gives the audit's numbers.
- The single-coin model is a market-timing signal (a real one: section 4); the panel (rank_ic) models time the market
  as well and add a small, real coin choice. The timing edge fell in 2026: N9 must read it per year.
- The best tested use is the breadth as exposure on a diversified basket (N9): every row of section 2 sits under its
  own basket x breadth line. N9 should read the panel models as well as the single-coin one, with the coin tilt as a
  second arm (basket weights tilted toward the top picks, not a few slots only).
- Not checked: slippage and funding; the panel tables have no `last_close`, so their 2 sub-books come from the
  artifact contract (8 h label on 4 h rows), not from the table; the tables of rolling-btcrel-lbl and rolling-mres*
  carry a relative / residual label (y_ret is not a price ratio), so the section skips them (correctly: an account
  needs a price return).
