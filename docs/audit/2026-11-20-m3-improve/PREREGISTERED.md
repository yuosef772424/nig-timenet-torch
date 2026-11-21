# Pre-registration: improving the information of method 3 (M3), a LOCAL CPU study (owner 2026-11-20)

Written before any number of the arms below is read (the reproduction of the baseline arm B was running while this
was written; none of its numbers had been read). Exploratory (rule C5). NOT the owner's artifact: the local 60-coin
slice (`data_io/vision_slice.py`, 4h, 2022-01 .. 2025-12, stride 2, 252,562 rows, raw fingerprint `1d66bb68`;
residual artifact by `data_io/market_residual.py --raw`, fingerprint `71c1f2bd`, the same bytes as `m3-local-001`).

## The owner's goal

"Improve the INFORMATION M3 found, not necessarily make it profitable: several models will run live, each used for
the skill it is best at." M3's skill = ranking coins inside a timestamp by their market-residual move. Profit after
costs is NOT an accept criterion here.

## What the M3 reading says (the evidence the hypotheses follow from)

- E1: residual rank IC by timestamp 0.077 local / 0.093 full, every fold; the ridge reads less (0.051 - 0.062).
- E2: the top of the ranking is weak: residual return by decile of mu is not monotone (local, seed 0: d1 +0.046 %,
  d8 +0.063 %, d10 +0.036 %); full artifact: mu quintile 5 inside the book not above quintile 1.
- E3: mu is tied to the predicted sigma (Spearman across deciles about -1): the ranking is partly "low volatility =
  high mu"; the lowest decile (volatile coins) has a positive mean simple residual (right tail).
- E4: extreme down-bias (share mu > 0 0.001 - 0.27 against 0.44 - 0.48 realised); the residual is right-skewed and
  the NIG location (a Student-t-like, robust loss) follows the bulk (median), not the mean.
- E5 (not verified): confident losses on market-wide days (2023-06-10) suggest the 180-bar beta under-states falls.

## Baseline arm B (the step-2 reproduction of arm M of `m3-local-001`, this commit)

`jobs.rolling`, CPU, single-coin GRU 64 x 2, lr 3e-4, batch 512, 5 epochs, patience 2, 60k train rows (strided),
all 60 coins, expanding folds (first train window 730 days, test 180 days, step 180 days, 4 folds, TEST 2024-01 ..
2025-12), embargo 2 days, seeds 0 and 1. Every arm below: the same, with ONE change.

## Metrics (information, not profit); per seed, TEST pooled over the 4 folds, and per fold

The reference label is B's residual label (log residual, 180-bar beta); the reference return is B's simple residual
return `y_res` (B's table `y_ret`), joined to every arm by (coin, end_time) (`m3_improve.py read`).

1. **IC**: rank IC by timestamp (timestamps with >= 5 coins) of the arm's score against the reference label.
   Paired test (rule C4): per-day mean of (IC_arm - IC_B) on the same timestamps, day-block bootstrap 95 % CI.
2. **Top decile**: rows with score >= the fold's VAL 90th percentile of the score: mean `y_res` per trade (gross),
   day-mean with a day-block 95 % CI.
3. **Decile curve inside the timestamp**: score deciles within each timestamp; mean of (`y_res` - the timestamp's
   mean `y_res`) for deciles 1, 2, 5, 9, 10. "Top monotone" = d10 > d9 > d5.
4. **Bias**: share of rows with mu > 0 against the share with `y_res` > 0.

Trivial baselines (rule C1), on the same rows, next to every IC and top-decile number: a shuffled score (within
timestamp), short-term reversal (minus the last bar's residual close move), low volatility (minus the sd of the
window's residual close moves), and beta (the window's last beta, high = up). A metric a trivial score matches is not
evidence.

## Hypotheses (one change per arm), each with its accept rule (both seeds)

Outcome words: **accepted** = the rule holds; **suggestive** = the point estimates hold on both seeds but a CI part
fails; **rejected** otherwise. Every rule is against B on the same commit, folds, seeds and rows.

### H1 - mean-corrected score (evaluation only, no training; follows E3 / E4)
Change: rank by the predicted MEAN simple residual instead of the NIG location: score = (1 + mu) x exp(s^2 / 2) - 1,
s = sigma / (1 + mu) (the lognormal mean under the predicted sd; mu, sigma from B's own tables).
Why: the location follows the median of a right-skewed residual, so it ranks volatile coins too low (E3); the mean
adds back sigma^2 / 2. Target: the top of the ranking.
Accept: decile-10 excess above B's d10 AND top-decile gross per trade above B's, on both seeds; the IC not below B's
by more than 0.005 on either seed.

### H2 - cross-section rank term on the residual data (follows E1 / E2; no code change)
Change: `train.cross_section_loss: rank_ic` (weight 1.0) with `train.batch_order: time_blocks`,
`time_block_bars: 28` (about 520 rows per batch at the 60k row cap, close to B's 512, so the steps per epoch stay
comparable). Why: the skill IS a within-timestamp ranking; the NIG loss learns it only indirectly (panel-001: the rank
term raised the rank IC). Known cost: time blocks alone worsened the raw loss (`batch-by-wf-002`).
Accept: IC above B's on both seeds AND the paired day-block CI of the IC difference above 0 on both seeds AND the IC
above B's in >= 3 of 4 folds on each seed.

### H3 - cross-sectional rank-gauss label (follows E2 / E3 / E4; needs a `market_residual` option, off by default)
Change: the close label = the normal score of the residual's rank inside its timestamp (rank / (n + 1) -> inverse
normal CDF), times a constant (the residual label's sd) so it stays in return units; inputs unchanged.
Why: a symmetric, unit-free target with no skew and no volatility scale: the NIG location then has no reason to
follow the median shift of volatile coins (E3 / E4) and every rank, the top included, weighs the same (E2).
Model-free bound first: ridge (the feature set of `m3_local.py`, each fold's train rows) fitted on the rank-gauss
label against fitted on B's label, both scored by IC against B's label. Train the arm only if the ridge IC on the
rank-gauss label is not below the ridge on B's label by more than 0.005 (pooled over folds).
Accept (if trained): as H2 (IC vs the reference label, paired CI, >= 3 of 4 folds).

### H4 - a different beta (follows E5; data variant; model-free gate first)
Candidates, built by the study script with the package functions, one change each: (a) a 60-bar beta window; (b) a
downside beta (cov / var on the bars where the market move < 0); (c) the market without the coin itself (with 60
coins the coin is 1/60 of its own market).
Model-free reading per variant (same rows, all bars of the 4 test windows): (i) market contamination: the mean
residual label on the timestamps whose market move over the label span is in its bottom 5 % (B: the reference),
and the correlation over timestamps of the mean residual with that market move; (ii) ridge IC against the variant's
OWN label and against B's label.
Gate to training (pre-registered): |mean residual on the bottom-5 % market spans| at least 25 % smaller than B's AND
the ridge IC against its own label not below B's ridge IC against B's label by more than 0.005. At most ONE variant
is trained (the one with the smallest contamination among those that pass).
Accept (if trained): IC against its own label above B's IC against B's label on both seeds, AND top-decile gross on
its own residual return above B's on B's residual, on both seeds. (Different labels: a cleaner residual that is as
predictable is more useful information; this is weaker evidence than H2 / H3 and is reported as such.)

### Dropped
- Longer label horizon: tested model-free in `m3-local-001` (M3a): the IC does not rise (24 h 0.041 - 0.056 vs the
  8 h ridge 0.051 - 0.062); only the gross per trade grows with the span (a profit question, not this one).
- New input channels: no reading points at a missing input; the model already beats the ridge (E1).

## Budget and multiple testing
Each CPU run <= about 20 minutes (both seeds in parallel); rows are reduced before seeds. Variants read: H1, H2, H3
(+ its ridge), H4 (3 model-free variants, at most 1 trained) = up to 4 trained arms + 1 evaluation-only score + the
trivial scores; every number is a look on 60 coins and 2 test years, a GPU run on the full artifact decides.

## Amendment 1 (written after reading B, H1, H2, H3 and the H4 gate; before H4's and H5's numbers)
Reading that motivates it: B's best epoch is the LAST epoch (5 of 5) in all 8 fold x seed runs (and in all 8 of H3):
the baseline may stop before it has read what the data hold, so every arm is compared at an under-trained point.
### H5 - longer training (one change: the epoch budget)
Change: `train.epochs` 10, `early_stop.patience` 3 (B: 5 / 2); everything else as B (the full-artifact run used 8 /
3 and stopped at epoch 4-7). Accept: as H2 (IC above B on both seeds, the paired day-block CI above 0 on both seeds,
above B in >= 3 of 4 folds on each seed). Also reported: the best epoch per run (still the last = still learning).
