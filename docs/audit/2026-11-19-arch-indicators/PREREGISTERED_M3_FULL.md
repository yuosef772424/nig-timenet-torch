# Pre-registration M3-full: method 3 on the owner's artifact (written 2026-11-19, to run 2026-11-20)

Written before any number of the run. Exploratory (rule C5). Follows the local look `docs/results/m3-local-001.md`
(feasible on 60 coins, does not pay at 8 h; the market put back does not help).

## Data and arm
`/content/art_mres` = `market_residual` on `/content/art1` (artifact `c572b1b36960`, 535 coins: the market is their
equal-weight close log move per bar; beta per coin from its last 180 bars), candles from the Drive cache
`crypto_model/candle_cache_4h_hlcqv`. Arm M3 = arm R of `rolling-ind-001` with only the artifact changed (GRU 64 x 2,
lr 3e-4, 8 epochs, patience 3, 400k rows, top 100 coins, 7 expanding folds, seeds 0 and 1). Reference A = arm R
(`rolling-ind-001/out_R_s{0,1}`, commit 458bcdb, on Drive). Rows of art1 and art_mres are the same rows.

## Reading (scripts in this folder; arms joined by coin and time)
1. `m3_local.py read --arm A=<out_R_s0> --arm A=<out_R_s1> --arm M=<out_M3_s0> --arm M=<out_M3_s1>`: close rank IC by
   timestamp against the residual label, M vs A; the VAL-90th-percentile top decile's residual gross per trade.
2. `m3_readd.py /content/art_mres <out_R_s0> <out_R_s1> <out_M3_s0> <out_M3_s1>`: V0 / V1 / V2 / always long.
3. `read_arms.py` on M3 alone (long and short apart, charts to Drive `analysis-ind-001/m3_full`).

## Accept rules (both seeds)
- Feasible: M's IC against the residual label above A's, and M's top-decile residual gross per trade above A's.
- Pays: M's top-decile residual gross per trade above 0.16 %.
- Market back better: V1 or V2 above V0 on the mean day AND on the mean without the best 5 % of days.
A fail gets the failure reading of rule C7 before any next arm.

## Amendment 1 (2026-11-20, after the three readings above, before the numbers below)
V1 (M's residual forecast + beta x the market forecast, VAL 90th percentile) passed "market back better" on both
seeds (`docs/results/m3-full-001.md`). Before any claim, V1 is read as a sceptic with
`docs/audit/2026-11-20-readings/m3_v1_read.py` (information, no new accept rule): day mean per fold and per year;
the same book minus the equal-weight market move and minus BTC (two round trips); beta to BTC; share of days with a
trade and trades per day; and a timing-only book on the same timestamps (every row of a timestamp that V1 trades,
equal weight) to separate the coin choice from the timing. V1 stays a hypothesis for live use until it repeats on
other seeds (2, 3) of the same commit.
