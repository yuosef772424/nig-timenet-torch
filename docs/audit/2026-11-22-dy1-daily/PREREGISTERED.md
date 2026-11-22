# DY1 pre-registration: a daily model that predicts two days ahead, judged as money after costs

Owner task 2026-11-22 (step DY1 = ROADMAP N3, the cost wall, carried out through a daily model; N9 is deferred
until DY1 is read). Goal in the owner's words: a model that makes money after fees, even a small amount ("at least
it buys me a cup of coffee"). Try the panel model AND the single-coin model. Exploratory (rule C5). Written before
any number of the daily data was read.

## Why daily (the hypothesis)
Every 4 h reading so far: rank IC above zero, gross edge per trade of the order of 0.04-0.10 % over the market,
below the 0.16 % round trip (B-074, `docs/handoff/2026-11-20-start-here.md`). A 2-day label has a move about
sqrt(12) = 3.5 x larger than an 8 h label; if the IC per row stays of the same order, the gross edge per trade grows
about 3.5 x while the cost per trade stays 0.16 %, and a daily rebalance of two 2-day sub-books trades a sixth as
often as the 4 h book. H1: at the daily frame the gross edge per unit of turnover exceeds the cost.

## Data (one artifact for every arm)
`build_daily.py` here: public Binance USDT-perpetual daily klines (data.binance.vision), every non-sealed symbol whose
first daily month is <= 2023-06 (delisted symbols included), bars from 2020-01-01, bars whose label ends at or after
the time seal (2026-10-01) dropped. Definition = `data/vision_slice.py::build_logret_dataset`: 32 daily bars of
[log(high / prev close), log(low / prev close), log(close / prev close)], labels log(future / last close) for close
(after 2 days), high (max of the next 2 days), low (min), stride 1 (one row per coin per day). The artifact is NOT
the owner's Drive artifact (other source, no TF pipeline): numbers here are about the method, not the recorded 4 h
baseline.

## Folds (as every 4 h walk-forward)
Entry `rolling`, expanding, first train span 1095 days, test 180 days, step 180 days, embargo 2 days, 7 folds, seeds
0 and 1, `top_coins` 100 (the 100 coins with the most TRAIN rows of each fold), validation = the last 15 % of each
train window. The TEST days of the 7 folds pooled per seed (about 1260 days, 2023-01 .. 2026-06 when the data start
2020-01-01).

## Arms (one commit, one artifact, the 4 h configurations unchanged except the data)
| arm | configuration | from |
|---|---|---|
| S | single-coin GRU, `selection_metric` val_raw_loss, 8 epochs, patience 3, lr 3e-4, batch 512 | `docs/jobs/rolling-dw1-r-001.json` |
| P | panel: `cross_section`, `cross_input` relative, `target_norm` logret_std, `cross_section_loss` rank_ic, selection val_rank_ic_reg, the rest as S | `docs/jobs/panel-rolling-std-001.json` (the best 4 h account, `n8-account-001.md`) |

`max_rows` is not set (the daily artifact is small). Rule C6: the `rolling` entry does not run the gates (as every
4 h walk-forward); recorded as not run.

## Model-free first (N3; before any training is read)
`model_free.py` here, on the same folds and coins (top 100 by TRAIN rows): per fold a ridge fitted on the TRAIN rows
(features: the 96 window values, the 1 / 3 / 7 / 14 / 30-day returns and the 30-day range; each demeaned across the
coins of the timestamp; target the 2-day close log return demeaned across coins), and the trivial rules momentum 7 d,
momentum 30 d, reversal 1 d, reversal 3 d (score = +/- the past return). Each score goes through the SAME account
(`eval/account.py::account_section`) as the arms. The table: per rule, rank IC per day (mean, t), gross edge of the
long book over the equal-weight basket per trade, turnover per period, CAGR at 0 and at 0.08 % per side.

## Measurement (the N8 account, unchanged code)
`workflow/battery_run.py::account_from_folders` over the 7 TEST folds of one seed: the long book = rows with mu >= the
fold's VAL q90 of mu, fixed slots of 1 / (coins of TEST), cash for unused slots, 0.08 % per side on the turnover,
2 sub-books (2-day label on daily rows), compounded; beside always long (same slots), BTCUSDT buy-and-hold, the
random-coin timing placebo (20 draws), the basket x breadth. Also the battery per fold (as written by the entry).

## Accept rule (T-profit for DY1; per arm)
An arm PAYS when, on BOTH seeds: (1) the account CAGR of its long book at 0.08 % per side is > 0, (2) it beats always
long on CAGR, (3) it beats the placebo p95 on CAGR (the money comes from the model's coin choice, not the market).
Reported, not required: BTC buy-and-hold (+ its drawdown), Sharpe, max drawdown, years, the fee grid, the day-block
bootstrap CI of the mean daily account return (level 1 - 0.05 / 2: two candidate arms; the model-free rules are
baselines, not candidates). An arm that does not pay gets the failure reading of rule C7 before anything else is
tried.

## Secondary (information, decided now so it is not fitted later)
- the market-neutral book: long the VAL-q90 picks, short the VAL-q10 picks, equal slots each side, the same fee;
  (short-side-001.md: the short side has lost at 4 h; reported, not a candidate).
- the 4 h runs over the same TEST window (`n8-account-001.md`) as the reference: does the daily frame beat
  panel-rolling-std-001's +23.0 / +18.9 %.

## Amendment 1 (owner 2026-11-22, written before any arm table was read)
Owner: test every normalisation with two seeds each (the default, std, and the method that removes the market
direction entirely), each with both model kinds (panel and single-coin); prefer coins of similar volatility and stay
away from coins that move little (BTC).
- Universe VM: `vol_universe.py` -> `coins_volmatched.txt`: 82 coins whose sd of daily close log moves over the
  rows ending before 2023-01-01 (TRAIN era of every fold; >= 180 such days) lies between the 25th and 95th
  percentile (0.063 .. 0.094 per day). Artifact `art_vm` = `build_daily.py --coins-file coins_volmatched.txt`.
- Market removed (M3, `data/market_residual.py --raw <raw_1d> --beta-bars 90`): inputs and the close label minus
  beta x the equal-weight move of the 82 coins (beta from each coin's past 90 days), artifact `art_vm_mres`.
- Six arms x seeds 0, 1, the same folds: `dy1-vm-s-def` (single, target_norm none), `dy1-vm-s-std` (single,
  target_norm logret_std), `dy1-vm-s-mres` (single on art_vm_mres), `dy1-vm-p-def` (panel, relative + rank_ic,
  target_norm none = panel-rolling-001), `dy1-vm-p-std` (panel + logret_std = panel-rolling-std-001), `dy1-vm-p-mres`
  (panel-rolling-001 config on art_vm_mres). The arms S and P on the 164-coin universe (above) stay as the reference.
- The accept rule is unchanged (account at 0.08 % per side, CAGR > 0, > always long, > placebo p95, both seeds);
  candidates are now 8 arms: the CI level becomes 1 - 0.05 / 8. BTC is not in VM: BTC buy-and-hold is read from the
  164-coin artifact over the same days.
- An M3 table's y_ret is the residual return (coin minus beta x market): its account is the P&L of a long position
  hedged with beta x the basket (hedge cost not charged). It is ALSO read on the coins' own returns (the picks joined
  to `art_vm`'s y_ret), and only that own-return account counts for the accept rule of the M3 arms.

## Amendment 2 (owner 2026-11-22, written before any arm table was read)
Owner: add arms with, beside the rank loss, the absolute loss of each coin on its own. The p arms already train the
per-row NIG loss of every coin PLUS the rank_ic term (`trainer/trainer.py`: the term is added with
`cross_section_weight` 1.0), so the new arms drop the rank term and keep the per-row loss alone: `dy1-vm-pa-def`,
`dy1-vm-pa-std`, `dy1-vm-pa-mres` (= p-def / p-std / p-mres with `cross_section_loss` none; selection metric
unchanged, val_rank_ic_reg), seeds 0, 1, started on the same commit (594b0c0) and runtime as the six others.
Candidates are now 11 arms: the CI level becomes 1 - 0.05 / 11.

## Amendment 3 (owner 2026-11-22: "for each model separately"; written before any arm table was read)
The loss grid for BOTH model kinds: per-row loss only vs per-row loss + rank term. The panel has both (pa / p); the
single-coin model gets the rank term in `dy1-vm-sr-def`, `dy1-vm-sr-std`, `dy1-vm-sr-mres` = s-def / s-std / s-mres
with `split.cross_section` true (the rank term ranks the rows of one timestamp: it needs whole-timestamp batches; the
inputs stay the coin's own, `cross_input` none), `cross_section_loss` rank_ic and selection val_rank_ic_reg (as every
rank arm). Seeds 0, 1, commit 594b0c0, the same runtime. Candidates: 14 arms, CI level 1 - 0.05 / 14.
