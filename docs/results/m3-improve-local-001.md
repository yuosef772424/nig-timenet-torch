# m3-improve-local-001: verifying M3 and improving its information, local CPU (owner 2026-11-20; exploratory, C5)

NOT the owner's artifact: the local 60-coin slice (`data_io/vision_slice.py`, public Binance futures 4h klines,
2022-01 .. 2025-12, stride 2, 252,562 rows, raw fingerprint `1d66bb68`), its residual artifact by
`data_io/market_residual.py --raw` (fingerprint `71c1f2bd`: byte-identical to `m3-local-001`'s). Goal (owner): improve
the INFORMATION of method 3 (ranking coins inside a timestamp by their market-residual move), not the profit.
Pre-registration `docs/audit/2026-11-20-m3-improve/PREREGISTERED.md` (+ amendment 1), written before the numbers;
script `docs/audit/2026-11-20-m3-improve/m3_improve.py` (`leak`, `variants`, `read`, `detail`). Runs: `jobs.rolling`
on this sandbox's CPU (4 cores, two seed processes at a time, about 10 minutes per arm), single-coin GRU 64 x 2, lr
3e-4, batch 512, 5 epochs, patience 2, 60k train rows (strided), all 60 coins, 4 expanding folds (TEST 2024-01-03 ..
2025-12), embargo 2 days, seeds 0 and 1. Code: B ran on 08b3cd2, H2 - H5 on a6c1b8e; between them only
`data_io/market_residual.py` changed (the two options, default output unchanged: fingerprint `71c1f2bd`), a module the
training path does not import, so all arms run the same model / trainer / evaluation code. Outputs stay in
`/tmp/m3_improve` (not on Drive). Every number below is a look on 60 coins and 2 test years.

## Verification of M3 (step 2)

- **Reproduced exactly.** Arm M of `m3-local-001` re-run on this commit (arm B here) gives, through the original
  `m3_local.py read`: residual rank IC by timestamp 0.07583 / 0.07736 (recorded 0.076 / 0.077), folds +0.078 +0.072
  +0.087 +0.066 / +0.084 +0.072 +0.086 +0.068; top decile residual gross +0.046 % / +0.033 % per trade, day mean
  +0.062 % [+0.032, +0.091] / +0.046 % [+0.007, +0.084] (recorded the same). The CPU runs are deterministic.
- **No leakage found in `market_residual.py`.** (1) Perturbing every coin's closes after a cut bar K (4 cuts, 10
  coins) leaves the market, beta and all channels of bars <= K unchanged (max change 0). (2) 3000 random rows
  recomputed from the raw candles: residual label = own label - beta(bars <= last bar) x market move of the two bars
  after the window, last-bar residual close and the beta channel, all to <= 6e-8. (3) `last_candles` timestamp is the
  OPEN time of the window's last bar (vision_slice docstring; settled for art1 by T5r), and the market grid is
  indexed by open time: aligned.
- **The coin in its own market does not matter here**: beta with the coin in the equal-weight market minus beta
  without it: median +0.006 over the 60 coins (+0.002 .. +0.011); H4's model-free variant `ex_coin` reads the same
  ridge IC (0.054 vs 0.055).
- **Discrepancies (facts):**
  1. `market_residual.py:_fill_coin` writes the residual label for every row on the grid but the channels only for
     rows whose 32 window bars all have valid channels: 765 rows (0.3 %) keep a residual label with ZERO inputs and
     enter training and test (`m3_local.py ridge` drops them, the model does not). A row off the grid keeps its OWN
     label (0 here, 836 on art1 by its derivation record).
  2. `market_residual.py` takes beta and the market from the candle source but never checks that the source's closes
     agree with the artifact's (`indicator_channels.py` counts `close_mismatch`; this tool does not). Same source
     locally; on art1 not verified.
  3. The report metric `targets.close.skill.rank_ic_by_timestamp` (`test_report.json`, `fold_row.json`) averages
     over every timestamp with >= 2 rows. On this stride-2 slice half of the test-era timestamps (2,186 of 4,372, all
     on the 04:00 / 12:00 / 20:00 phase) hold fewer than 5 coins (5 % of the rows); their 2-row Spearmans are +-1:
     H2 fold 0 reads 0.052 in its report and 0.080 on timestamps with >= 5 coins (B: 0.077 / 0.078). Read M3 only
     with a minimum group size (`m3_local.py read` and `m3_improve.py read` use 5).
  4. `m3_local.py ridge` hard-codes `FIRST_TEST = 2024-01-01`; the jobs' test windows start 2024-01-03, 2024-07-01,
     2024-12-28, 2025-06-26 (two days later), and the ridge fits all train rows (not the job's 60k strided rows and
     its VAL cut): the bound is near, not on, the model's rows. Its `read` join is correct (each arm's TEST row is
     joined by (coin, end_time) to M's residual return of the same fold and seed; no duplicate keys; per-fold VAL
     threshold).
  5. The run log line "N batches per epoch (batch_by rows)" (`workflow/run.py:195`) does not reflect
     `batch_order: time_blocks` (H2 printed 118 batches, the block count is different).
  6. B's best epoch is the last one (5 of 5) in 8 of 8 runs: the baseline stops while still improving its VAL loss.

## Verdict per hypothesis (pre-registered rules, both seeds; B = 0.0758 / 0.0774 IC, +0.046 / +0.033 % top decile)

| arm | change | IC seed 0 / 1 (paired dIC [95 % CI]) | top decile gross % | d10 excess % | rule | outcome |
|---|---|---|---|---|---|---|
| H1 | rank by the lognormal mean of the prediction (no training) | 0.0742 / 0.0763 (-0.0017 [-0.0024, -0.0010] / -0.0011) | 0.042 / 0.035 | 0.020 / 0.037 (B 0.028 / 0.040) | d10 and gross above B, IC within 0.005 | **rejected** (d10 lower on both seeds) |
| H2 | rank term `rank_ic` + time blocks of 28 | 0.0792 / 0.0777 (+0.0033 [+0.0001, +0.0066] / +0.0005 [-0.0027, +0.0038]) | 0.024 / 0.030 | 0.037 / 0.041 | IC above B, paired CI > 0, >= 3 of 4 folds, both seeds | **rejected** (seed 1: CI contains 0, 2 of 4 folds) |
| H3 | rank-gauss close label (ridge gate passed: 0.073 vs 0.051) | 0.0747 / 0.0788 (-0.0010 / +0.0014, both CIs contain 0) | 0.035 / 0.028 | 0.022 / 0.037 | as H2 | **rejected** |
| H4 | 60-bar beta (gate passed: bottom-5 % contamination 0.012 % -> 0.006 %) | own label 0.0730 / 0.0761; B's label 0.0742 / 0.0777 | own 0.036 / 0.029 | 0.029 / 0.039 | own IC and own top decile above B's | **rejected** |
| H5 | 10 epochs, patience 3 (amendment 1) | 0.0801 / 0.0789 (+0.0042 [+0.0019, +0.0066] / +0.0017 [-0.0006, +0.0040]) | 0.042 / 0.041 | 0.043 / 0.042 | as H2 | **suggestive** (above B on both seeds, in 3 / 4 and 4 / 4 folds; the seed-1 CI contains 0) |

H4 model-free (TEST windows, ridge on 9 residual features; `m3_improve.py variants`): corr(mean residual of the
timestamp, market move over the label span) base -0.045, 60-bar -0.016, downside beta -0.109, market without the coin
+0.286 (mechanical: each coin is measured against a market without itself, the base span includes it); mean
residual on the bottom-5 % market spans +0.012 % / +0.006 % / +0.023 % / -0.049 %; ridge IC against the
own label 0.055 / 0.052 / 0.055 / 0.054. The gate passed the 60-bar beta on a relative shrink of an already
negligible base (the gate was badly specified: it should have had an absolute floor).

Trivial baselines on the same rows (TEST pooled, both seeds the same rows): shuffled IC -0.002 / +0.004; short-term
reversal (minus the last residual close move) 0.039; low volatility (minus the window's residual sd) 0.043; beta
-0.033. B's IC net of the low-volatility score (partial rank IC, not pre-registered) 0.064 / 0.065: most of the
model's IC is not the volatility ranking. The trivial top deciles: reversal +0.039 %, low volatility -0.014 %,
shuffled +0.020 / +0.026 % per trade (the mean residual of all rows is +0.015 %).

## What the model learned

- Every arm reads the same information: IC 0.074 - 0.080 against the residual label on both seeds and in every fold
  (folds 0.065 - 0.096), against 0.039 / 0.043 for the best trivial scores and about 0.055 for a ridge on the same
  features. Target, loss term and beta changes (H1 - H4) move the IC by -0.002 .. +0.003, inside the seed spread (B
  seeds differ by 0.0016, H2 seeds by 0.0015). Net of the low-volatility ranking the IC is still 0.061 - 0.068 in
  every trained arm.
- The one change that moved it is training longer (H5): +0.0042 / +0.0017, with the top of the ranking better on
  both seeds (d10 excess +0.043 / +0.042 % vs +0.028 / +0.040 %; the VAL-fitted top 5 % of mu +0.088 / +0.047 % vs
  +0.064 / +0.040 % per trade) and the IC inside a predicted-sigma tercile up (seed 0: 0.049 - 0.053 vs 0.040 -
  0.042). Its best epoch is still the last (10 of 10) in 4 of 8 runs: B is under-trained at 5 epochs.
- H3 learned the target it was given: its mu is centred (share mu > 0 0.53 / 0.51 against 0.47 realised), so the
  down-bias of M3 is a property of the target, not of the inputs; the ranking did not change.
- The ridge gain of the rank-gauss label (0.051 -> 0.073) is a least-squares effect: the ridge on the raw residual is
  pulled by the right tail; the NIG loss is already robust, so the network gains nothing from the same change.

## Where it is right and where it fails

Error map of B (seed 0 / seed 1; `m3_improve.py detail`; IC by timestamp inside the group, top decile = VAL 90th
percentile of mu, residual gross per trade; the trivial low-volatility IC on the same rows next to it):

| group | rows | IC B | IC low-vol | top-decile trades | top-decile gross % | group mean % |
|---|---|---|---|---|---|---|
| market move over the span down / up | 60,074 / 64,816 | 0.071 / 0.080 (s0) | 0.035 / 0.050 | 5,937 / 6,448 | +0.052 / +0.040 (s0); +0.049 / +0.018 (s1) | +0.014 / +0.015 |
| market span calm / mid / wild | 41.6k each | 0.070 / 0.074 / 0.083 | 0.043 / 0.048 / 0.038 | 4,450 / 4,362 / 3,573 | +0.026 / +0.036 / +0.083 (s0) | +0.012 / +0.014 / +0.019 |
| coin volatility calm / mid / wild (window sd) | 41.6k each | 0.065 / 0.069 / 0.057 | 0.016 / 0.018 / 0.025 | 7,919 / 3,435 / 1,031 | +0.034 / +0.075 / +0.039 (s0) | -0.005 / +0.018 / +0.031 |
| predicted sigma narrow / mid / wide | 41.6k each | 0.041 / 0.042 / 0.044 | 0.009 / -0.003 / 0.020 | 9,935 / 1,909 / 541 | +0.034 / +0.065 / +0.197 (s0) | +0.016 / +0.002 / +0.026 |
| year 2024 / 2025 | 62.8k / 62.1k | 0.075 / 0.077 | 0.052 / 0.033 | 5,328 / 7,057 | +0.058 / +0.036 (s0); +0.065 / +0.014 (s1) | +0.017 / +0.013 |

- The ranking holds in every group (no group carries it); it is a bit stronger on wild-market spans.
- Inside one predicted-sigma tercile the IC drops to about 0.04: roughly half of the timestamp IC is the ordering
  BETWEEN volatility classes, half within a class.
- The top decile is 80 - 87 % narrow-sigma (calm) coins in every arm (H3's symmetric target included: 10,633 of 12,270);
  the few wide-sigma rows that reach it earn +0.12 .. +0.21 % per trade. The book's small gross is a composition effect:
  it buys calm coins whose residual moves are small.
- Decile curve inside the timestamp (excess over the timestamp mean, d1 / d2 / d5 / d9 / d10): B +0.029 / -0.027 /
  -0.017 / +0.014 / +0.028 % (s0), +0.059 / -0.032 / +0.002 / +0.013 / +0.040 % (s1): the top is monotone (d10 > d9 >
  d5) on both seeds but the bottom decile is POSITIVE (U shape), in every arm.

## The samples it failed on

The worst top-decile rows of B (the two seeds share about half of their 12 worst): the crash of 2025-10-10 20:00 (market
-21.6 % over the 8 h span): THETA -9.1 %, ENJ -10.3 %, ICP -8.4 %, VET -7.8 %, ATOM -7.7 %, COMP -5.5 % residual, all
calm coins (window sd 0.4 - 0.6 %) with beta 0.8 - 1.1 that fell about 1.1 - 1.5 x the market; ETH 2024-08-04 20:00
(market -14 %, beta 0.74, -7.0 %), BCH 2025-02-02 20:00 (market -18 %, beta 0.79, -6.8 %); and single-coin events in
calm markets (CRV 2024-06-12 -23 %, UNI 2024-04-10 -7.3 %, KAVA 2025-04-01 -8.9 %). In crashes the residual of calm,
low-beta coins is strongly negative: the 180-bar beta under-states their fall (E5 holds on these rows), although the
mean residual over all bottom-5 % market spans is +0.012 % (the under-statement is concentrated in a few crashes).

## Bias

B: share mu > 0 0.11 / 0.09 against 0.47 realised (per fold 0.02 - 0.16), mean mu -0.10 / -0.11 % against +0.015 %
realised (the M3 down-bias reproduces). H3 (symmetric target) removes it (0.53 / 0.51) without changing the ranking;
H1's mean correction does not change the share (sigma^2 / 2 is about 0.01 %). The book uses a VAL quantile, so the
bias does not change which rows are bought; it matters only to rules that read the sign of mu.

## Selective use

Three rules fitted on VAL, applied to TEST (residual gross per trade, information only, no cost; 3 tried per arm):
B top 5 % of mu +0.064 / +0.040 % (minus the timestamp mean +0.052 / +0.028 %), top 10 % with sigma under the VAL
median +0.039 / +0.031 %, top 10 % of mu / sigma +0.043 / +0.037 %. In the same order: H2 +0.034 / +0.028, +0.026 /
+0.028, +0.030 / +0.033 %; H3 +0.032 / +0.017, +0.030 / +0.028, +0.030 / +0.030 %; H4 +0.061 / +0.041, +0.041 /
+0.029, +0.039 / +0.032 %; H5 +0.088 / +0.047, +0.038 / +0.036, +0.045 / +0.037 %. A narrower top (5 %) helps B and
H5 on both seeds; the low-sigma filter and mu / sigma do not (they buy calmer coins with smaller moves). No rule of
H2 / H3 / H4 beats B's; H5's top 5 % beats B's on both seeds. The rule family is `read_detail.py`'s, not
pre-registered here; 3 rules x 6 score sets were read (multiple testing).

## Why it failed

**H1 - H4: they fix things that do not limit the ranking (skew, bias, beta); the limit is how much the model has
read:** target, loss and beta changes keep the IC at 0.074 - 0.079, the network already beats the ridge (0.055) and
the trivial scores (0.043), the robust NIG loss already gives it what a rank target gives a least-squares model, and
the only lever that moved the IC and the top (H5, suggestive) is more training of the same model. What limits the
TOP of the ranking is composition: the global VAL threshold fills the top decile with calm coins (80 - 87 %), whose
residual moves are too small to carry information in money terms, while half of the IC is the
between-volatility-class ordering.

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| N0 | M3 on the full artifact is under-trained too: a longer budget (16 epochs, patience 4) raises the residual IC and d10 above M3-full's | IC not above 0.093 / 0.094 on both seeds, or the best epoch stays below 8 (no headroom used) | one GPU arm on `art_mres`, seeds 0 / 1, the rest of `rolling-mres-001` unchanged |
| N1 | Ranking INSIDE a volatility class (top decile per predicted-sigma tercile, or score = mu / trailing residual sd) puts wide-sigma coins with real residual moves into the top and raises the top-decile residual gross per trade | the per-class top decile's residual gross is not above B's top decile on both seeds | read B's saved tables (no training); a vol-scaled label (residual / trailing sd) as a `CLOSE_LABELS` entry only if the reading passes |
| N2 | A beta that reacts in crashes (downside or shorter beta only when the market falls, or a crash flag channel) removes the 2025-10-10-type losses from the top decile | the worst 1 % of top-decile rows on VAL do not shrink | rebuild the artifact with the variant, re-read the top-decile tail model-free first |
| N3 | The full artifact (535 coins, 7 folds) shows the same ceiling: H2 / H3 do not raise the IC there either | H2 or H3 raises the IC above M3-full's 0.093 / 0.094 by more than the seed spread on both seeds | one GPU run of H2 (switches exist) on `art_mres`, seeds 0 / 1 |
