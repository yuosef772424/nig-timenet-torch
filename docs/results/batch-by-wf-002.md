# H1 / H2 of the batch_by study on 60 coins (owner 2026-11-18): H1 refuted, H2 supported — a look, not a result

**Exploratory (rule C5).** Public Binance klines through `data_io/vision_slice.py`: the 60 default coins, 4h,
2023-01 .. 2024-12, stride 2, 129,900 rows. NOT the owner's artifact. Pre-registration:
`docs/audit/2026-11-17-batch-by/PREREGISTERED_2.md` (commit 2ff054f) and its amendment made after the GPU smoke and
before any number (commit f7e53d3: size control C34, because a timestamp holds about 34 of the 60 coins). The three
files were removed by an automated message (fb90d01) and restored unchanged (f983e65) while the runs trained; their
content at f983e65 equals f7e53d3 + the handoff commit. Runs: Colab L4, commit f7e53d3, 7 processes in parallel, 4
expanding folds (test = the quarters of 2024), 3 seeds, 4 epochs, run-default model, battery off. Reading:
`h1_check.py`, `h2_bound.py`, `failure_reading.py <artifact> <study> C34` (commit c72d0b4), doctor card A.

| arm | batch | GPU minutes (one process; 7 shared the L4) |
|---|---|---|
| A0 | rows, 256 (default) | 20.8 (all 3 seeds) |
| A2 | timestamp (about 34 rows) | 46.1 (per seed) |
| C34 | rows, 34 (size control of A2) | 46.2 (per seed) |

## Verdicts against the pre-registered rules

**H1 (one-timestamp batches help the low head's cross-sectional ranking): refuted.** Low-head rank IC by timestamp,
A2 against each reference (pairs where A2 is higher of 12; mean per-fold paired difference; 95 % t-interval, 3 df):

| comparison | A2 higher | mean diff | 95 % CI | per fold | passes |
|---|---|---|---|---|---|
| A2 vs A0 | 7 / 12 | +0.0007 | [-0.0421, +0.0435] | -0.029, +0.036, -0.001, -0.004 | no |
| A2 vs C34 | 10 / 12 | +0.0350 | [+0.0064, +0.0637] | +0.017, +0.060, +0.034, +0.029 | yes |

Read, not judged: the test raw loss of A2 is WORSE than A0's in all 12 pairs (+0.113, CI [+0.064, +0.162]) and
worse than C34's in 10 of 12 (+0.066); the high-head IC of A2 is below A0 (-0.008, 4 / 12) and above C34 (+0.014,
8 / 12).

**H2 (the model under-reads the range information in its inputs): supported.** The better of the two model-free
rankings beats A0 on BOTH heads in ALL 4 folds (pre-registered: >= 3 of 4); the same for A2 and C34 (read):

| fold | head | 6-bar range | ridge (HAR features) | A0 | A2 | C34 |
|---|---|---|---|---|---|---|
| 0 | high | 0.319 | **0.325** | 0.293 | 0.277 | 0.268 |
| 0 | low | 0.423 | **0.437** | 0.384 | 0.355 | 0.337 |
| 1 | high | 0.356 | **0.361** | 0.339 | 0.327 | 0.303 |
| 1 | low | 0.419 | **0.424** | 0.365 | 0.401 | 0.341 |
| 2 | high | 0.326 | **0.333** | 0.310 | 0.315 | 0.301 |
| 2 | low | 0.379 | **0.387** | 0.322 | 0.321 | 0.287 |
| 3 | high | 0.330 | **0.347** | 0.324 | 0.316 | 0.308 |
| 3 | low | 0.387 | **0.390** | 0.339 | 0.335 | 0.306 |

The ridge's penalty sat at the top of its grid (10,000) in 7 of 8 fits, so a wider grid could raise the bound
further: the gap is a lower bound. Doctor card A (its own 3-epoch run on the default split, 60,000 rows, CPU): the
high label's probe IC at the trunk is 0.150 against 0.204 for the simple features (warn: the trunk holds less than
the inputs); the low label's is 0.220 against 0.214 (pass); the stage `branch_out` erases the `drift` tracer (r2
1.00 at the input -> 0.82 at the trunk, warn); and the outputs lean on the `linear_path` (shuffling its rows moves mu
by 0.83, against 0.19 for the encoder output).

## What the model learned

- Every arm ranks the high and low moves across coins (A0: high 0.317, low 0.352) and the close move barely
  (0.012). The trivial 6-bar mean range alone gives 0.333 / 0.402 on the same rows, the ridge on the HAR terms
  0.325-0.361 / 0.387-0.437 per fold: **no arm reaches the model-free bound** (H2).
- Batching: small row batches (C34) cost 0.034 of low IC and 0.022 of high IC against the default 256 (C34 0.318 /
  0.295 vs A0 0.352 / 0.317). One-timestamp batches of the same size (A2) win that back on the low head (0.353, +0.035
  over C34, 10 / 12) but only reach the default, and they lose on the objective (test loss worse than A0 in 12 / 12).
  The study-1 gain of A2 over A0 (+0.029 on 20 coins, 7 / 8) does not repeat on 60 coins (+0.001, 7 / 12): it was a
  20-coin effect or noise.

## Where it is right and where it fails

Low-head rank IC (seeds pooled), by volatility tercile of the last input bar and by fold:

| group | rows | A0 | A2 | C34 |
|---|---|---|---|---|
| calm third | 60,723 | 0.366 | 0.351 | 0.331 |
| middle third | 60,717 | 0.216 | 0.226 | 0.184 |
| volatile third | 60,720 | 0.198 | 0.171 | 0.173 |
| fold 0 (2024 Q1) | 45,540 | 0.384 | 0.355 | 0.337 |
| fold 1 (Q2) | 45,540 | 0.365 | 0.401 | 0.341 |
| fold 2 (Q3) | 45,540 | 0.322 | 0.321 | 0.287 |
| fold 3 (Q4) | 45,540 | 0.339 | 0.335 | 0.306 |

As on 20 coins, every arm ranks best in calm conditions and loses about half of it in the volatile third. A2 beats A0
only in fold 1 and in the middle third; it loses in fold 0 and in the volatile third. Per coin the low-head error moves
by at most 0.0006 between A0 and A2: batching does not change which coins are hard.

## The samples it failed on

The largest errors of A0 on the high head (seed 0) are again upward jumps predicted at 1-4 %, mostly out of a quiet
last bar: UNI 2024-02-23 (+57 %, last-bar range 0.016), XTZ 2024-11-18 (+44 %, 0.026), CELO 2024-03-11 (+42 %,
0.031), HBAR 2024-04-23 (+40 %, 0.014; and +36 % the next row), plus the November-December 2024 rally (SAND, TRX,
XLM, already volatile). The same two kinds as on 20 coins: a coin-specific squeeze from a quiet base (B-077) and the
market-wide alt-coin rally.

## Bias

The close head leans short on 60 coins more than on 20: it predicts up on 42.6-45.2 % of the test rows against 50.3 %
realised, mean prediction -0.00041 .. -0.00004 against a mean label of +0.00069. A2 leans the most (42.6 %). A
re-centring fitted on the inner validation was not tried (not part of this question).

## Selective use

Not measured: T-profit and the battery were off by pre-registration. The error map says where a later rule would
look: the calm third (low IC 0.33-0.37 against 0.17-0.20 in the volatile third). Rules tried on TEST here: none.

## Why it failed

H1 failed because **the gain of timestamp batches is a gain over SMALL batches, not over the default**: grouping by
timestamp recovers what a 34-row batch loses (A2 vs C34 +0.035), but the default 256-row batch never lost it (A2 vs
A0 +0.001), and the 7x more, noisier steps of A2 cost it on the objective (test loss worse in 12 / 12). Behind all
arms stands the cause H2 confirms: the volatility signal of the inputs is not fully read (the ridge on the HAR terms
beats every arm in every fold), and no batching choice touches that.

## Next hypotheses

1. **H3 (H2's action): give the model the range terms it under-reads.** One switch, e.g. the HAR range terms of
   `simple_features` as an extra input of the readout (or the existing `input_skip` variant `late_raw`), through the
   gates (rule C6) first. Refuted if, pre-registered on the per-fold rank IC of high and low on these 60 coins, the arm
   does not reach the ridge bound in at least 3 of 4 folds on both heads. Cheapest test: the E1 gates (minutes), then
   this runner with A0 vs the switch, 3 seeds (~25 min on the L4).
2. **H4: whole-timestamp batches of the DEFAULT size.** The grouping effect at equal size (+0.035) may add to the
   default if a batch holds several whole timestamps (~256 rows, the groups shuffled: what `trainer/groups.py`
   `group_batches(gid, 256)` already does for cross-section runs, without the group channel in the input). Refuted if
   the paired low-head IC against A0 has a 95 % CI that includes 0 (3 seeds, 4 folds). Cheapest test: a `batch_by`
   value that passes `batch_size` to `group_batches` (one line in `Trainer._train_epoch`), then the same runner.
3. The doctor's `branch_out` erases the drift tracer and the outputs lean on `linear_path`: read card A's stage
   table on a fold model of A0 before choosing H3's switch (where the range signal is lost decides where to feed it).
