# arm-i-001: arm I of rolling-ind-001 (7 indicator channels, lr 1e-5) read against R (exploratory, rule C5)

Job `docs/jobs/rolling-ind-001.json`, Colab L4, commit 458bcdb (2026-11-19). Arm I = single-coin TCN 64 x 2, 7 input
channels (`art_ind`: the 3 price channels + RSI 14, stochastic %K / %D, relative quote volume), NIG + class losses,
random rows of 512, lr 1e-5, 16 epochs (patience 4), 400k train rows, top 100 coins, 7 expanding folds (TEST 2023-01
.. 2026-06), seeds 0 and 1. Reference R = GRU 64 x 2, 3 channels, lr 3e-4, 8 epochs (patience 3), same folds.
Pre-registration `docs/audit/2026-11-19-arch-indicators/PREREGISTERED.md` (arm I is "read the same way" as the
candidate IR). Read 2026-11-20 from the saved tables: `read_arms.py` (Drive `nig_timenet_runs/analysis-ind-001/arm_I`)
and `docs/audit/2026-11-20-readings/read_detail.py` (`analysis-ind-001/arm_I_detail`; the error map, worst rows,
bias and selection tables below come from its CSVs).

## Verdict

**FAILS** rules 2 and 3 (rule 1 holds): the long book's day mean is above 0 with its CI, but negative without the
best 5 % of days and below R on both seeds. Hedged against the market or BTC it loses like R.

| arm | seed | long book, mean day [CI 98.3 %] | without best 5 % | minus market | minus BTC, 2 costs | beta to BTC | close rank IC (mean of folds) |
|---|---|---|---|---|---|---|---|
| I | 0 | +0.146 % [+0.022, +0.284] | -0.085 % | -0.154 % | -0.170 % | 1.36 | 0.0176 (both seeds) |
| I | 1 | +0.144 % [+0.036, +0.268] | -0.066 % | -0.151 % | -0.177 % | 1.35 | |
| R | 0 | +0.173 % [+0.034, +0.310] | -0.072 % | -0.159 % | -0.184 % | 1.35 | 0.0185 (both seeds) |
| R | 1 | +0.215 % [+0.060, +0.362] | -0.022 % | -0.110 % | -0.130 % | 1.35 | |

Always long on all rows: -0.147 % per day. Short book (information only): I +0.053 % / +0.061 % per day (CI contains
0), R +0.133 % / +0.137 %.

## What the model learned

- **The ordering of coins inside a timestamp, weakly, as R does.** Close rank IC by timestamp above 0 in all 14
  fold x seed runs (0.005 .. 0.030; R 0.007 .. 0.044 with one negative fold), mean 0.0176 against R's 0.0185: the
  four indicator channels add no measurable ranking skill at this training budget.
- **The bullish tail is real in gross terms.** Inside the long book, the top fifth of mu earns +0.55 % / +0.73 % per
  trade gross against +0.14 % / +0.04 % for the bottom fifth (R: +0.88 % / +0.77 % against -0.01 % / +0.14 %).
- **It did not finish learning.** The best epoch is the LAST one (16 of 16) in all 14 runs; R stops at epochs 2-8 of
  8. At lr 1e-5 the arm was still improving when the budget ended (the stage-1 look said the same: idea LR).
- The channels change WHAT is bought, not the result: only 50 % / 58 % of arm I's long trades are also R's trades.

## Where it is right and where it fails

Error map of the long book (pooled TEST rows of 7 folds, per seed; net per trade after 0.16 %; "minus market" = net
minus the equal-weight move of the same timestamp; baseline = always long on the same rows):

| group | I s0 net | I s1 net | I minus market (s0 / s1) | R s0 net | baseline (always long, same rows) |
|---|---|---|---|---|---|
| sigma quintile 1 (narrowest) | -0.10 % | -0.11 % | -0.17 / -0.18 % | -0.17 % | (market) |
| sigma quintile 5 (widest) | +0.81 % | +0.84 % | -0.11 / -0.12 % | +0.78 % | (market) |
| mu quintile 5 in book | +0.39 % | +0.57 % | -0.16 / -0.17 % | +0.72 % | (market) |
| calm coins | +0.11 % | +0.20 % | -0.16 / -0.16 % | +0.20 % | -0.14 % |
| wild coins | +0.05 % | +0.09 % | -0.17 / -0.16 % | +0.08 % | -0.14 % |
| market up (realised) | +1.70 % (hit 0.79) | +1.80 % | -0.10 / -0.06 % | +1.87 % | +1.26 % |
| market down (realised) | -2.12 % (hit 0.13) | -2.24 % | -0.25 / -0.29 % | -2.40 % | -1.74 % |
| 2023 / 2024 | +0.15 / +0.20 % | +0.13 / +0.24 % | -0.14 .. -0.17 % | +0.15 / +0.25 % | -0.07 / -0.12 % |
| 2025 / 2026 (to 06) | -0.07 / +0.03 % | +0.08 / -0.07 % | -0.14 .. -0.19 % | +0.05 / -0.11 % | -0.22 / -0.22 % |

Every group beats always-long on its own rows, and no group beats the market move of its timestamps: the book's
gross over the market is about 0.00 % per trade in every row group (net minus market = -0.16 %, the cost). The
widest-sigma fifth earns the most and loses the least against the market (-0.11 %): the high-beta coins. In falling
markets the book loses 0.25-0.29 % per trade MORE than the market (beta above 1).

## The samples it failed on

The confident losses (top 2 % of mu, largest realised loss) are of two kinds, the same rows for I and R:

| kind | rows (coin, end of window, realised 8 h return, market move) |
|---|---|
| market crash, every high-beta alt falls | 2023-06-10 00:00: GTC -24 %, SUSHI -24 %, C98 -21 %, BEL -21 %, LIT -20 %, CELO -19 %, market -16 %; 2025-02-02 20:00: DYDX -22 %, ATOM -20 %, IOTX -20 %, ADA -19 %, market -16 %; 2025-10-10: CELR -29 %, market -21 % |
| coin-specific collapse while the market is flat | LINA 2025-03-21 -40 % / -31 % (market 0 %), BLZ 2024-12-18 -38 % / -25 %, NKN 2025-04-03 -32 %, RUNE 2025-01-23 / 02-01 -24 % / -29 %, ZEC 2026-06-05 -33 % / -26 % |

The worst days of the book are the market's worst days (2025-10-10 -16 % with 22 trades, 2024-04-13 -8 % with 279
trades, 2025-02-02 -7 % with 232 trades): the book's day loss is within 1 % of the market's on 8 of the 10 worst days (seed 0).
The single-coin collapses (LINA, BLZ, ZEC) get a high mu; whether they follow a run-up (the model buying a pump)
is NOT checked here (the table holds no past return): hypothesis PUMP below.

## Bias

Seed 0 of arm I leans down: share of rows with mu > 0 is 0.30-0.42 per fold against a realised share up of
0.46-0.50 (mean mu -0.05 %, realised mean +0.01 %); seed 1 is near balanced (0.45-0.55). R's seeds swing both ways
(0.38-0.60). The long book is a VAL quantile, so the level bias does not change which rows are bought; it shows the
close mean is not calibrated at lr 1e-5 (one more sign of under-training). The loss sits on the long side in falling
markets (beta 1.35 to BTC), not in a level bias.

## Selective use

Four rules, fitted on VAL, applied to TEST (written in `read_detail.py` before its numbers; 4 rules tried, mean of
fold x seed):

| rule | rows kept | I net per trade | I minus market | R net per trade | R minus market |
|---|---|---|---|---|---|
| top 5 % of mu | 5.2 % | +0.24 % | -0.16 % | +0.31 % | -0.16 % |
| top 10 % of mu, p_dir > 0.5 | 10.3 % | +0.11 % | -0.16 % | +0.12 % | -0.16 % |
| top 10 % of mu / sigma | 10.4 % | +0.00 % | -0.17 % | +0.03 % | -0.17 % |
| top 10 % of mu, sigma under the VAL median | 2.9 % | -0.12 % | -0.17 % | -0.18 % | -0.17 % |

A narrower book raises the net per trade only through more market exposure; against the market every rule is at
minus one cost. Preferring a narrow sigma makes it worse (calm coins move less with the market). No rule keeps a
coin-selection edge.

## Why it failed

**Not trained to the end, so the channels were never given a fair test:** the best epoch is the last epoch in all 14
fold x seed runs at lr 1e-5 x 16 epochs, while R (lr 3e-4) reaches its best by epoch 2-8. The comparison with R mixes
three changes (channels, encoder, lr / epochs) and the under-trained one loses. Shared with R and every reading of
2026-11-19: the long book's gross over the market is about 0, so its after-cost result is the market's move times
beta (B-074, `docs/results/btc-hedge-long-001.md`).

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| I1 | The indicator channels help when trained like R (lr 3e-4, 8 epochs, GRU): close rank IC above R's | IC not above R's in at least 5 of 7 folds on both seeds | arm "R + 4 channels" on `art_ind`, the R job with only the artifact changed (2 seeds, about 1 h with 4 processes) |
| LR | lr 1e-5 with warm-up + cosine and 32-48 epochs reaches a better optimum than lr 3e-4 | best epoch still the last one, or IC / VAL long book not above R's | 2 folds x 1 seed look (stage-1 style) before any walk-forward |
| PUMP | Excluding rows after a large run-up (the LINA / BLZ / ZEC rows) removes the coin-specific collapses from the book | the book's net minus market does not rise on VAL folds | a VAL-fitted filter on the last 24 h return, read on the saved tables (no training) |
