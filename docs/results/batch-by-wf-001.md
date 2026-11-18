# batch_by walk-forward on a 20-coin real sample (owner 2026-11-17) — a look, not a result

**Exploratory (rule C5).** CPU, public Binance klines through `data_io/vision_slice.py` (20 liquid unsealed coins, 4h,
2023-01 .. 2024-12, stride 2: 43,520 rows). NOT the owner's artifact. Question, arms, metrics and accept rule were fixed
before any number: `docs/audit/2026-11-17-batch-by/PREREGISTERED.md` (commit 5166f17). Code at commit adb16c1
(runner `run_arms.py`), reading `analyse.py` + `failure_reading.py` in the same folder. Walk-forward: expanding, 4
folds (test = the four quarters of 2024), 2 seeds, 4 epochs, run-default model, battery off.

| arm | batch | steps / epoch (fold 0) | CPU minutes (8 runs, 1 thread) |
|---|---|---|---|
| A0 | rows, 256 (default) | 72 | 6.5 |
| A1 | day (about 60 rows) | 305 | 9.0 |
| A2 | timestamp (20 rows) | 914 | 16.4 |
| C60 | rows, 60 (size control of A1) | 305 | 8.9 |
| C20 | rows, 20 (size control of A2) | 914 | 16.5 |

## Verdict against the pre-registered rule

No arm is better than another on the primary metric. Test raw loss, mean (sd over the 8 fold x seed runs): A0
-6.123 (0.281), A1 -6.056 (0.321), A2 -6.100 (0.344), C60 -6.100 (0.321), C20 -6.099 (0.292). Pre-registered
comparisons (X lower in how many of 8 pairs; mean paired difference X - Y; 95 % t-interval over the 4 folds):

| X vs Y | X lower | mean diff | 95 % CI | rule a | rule b | better |
|---|---|---|---|---|---|---|
| A1 vs A0 | 2 / 8 | +0.067 | [-0.014, +0.148] | no | yes | no |
| A2 vs A0 | 4 / 8 | +0.023 | [-0.089, +0.136] | no | yes | no |
| A1 vs C60 | 2 / 8 | +0.044 | [-0.052, +0.140] | no | yes | no |
| A2 vs C20 | 5 / 8 | -0.000 | [-0.095, +0.095] | no | yes | no |

`rows` (A0) stays the default; nothing changes in the code.

## What the model learned

- Every arm ranks the high and low moves across coins well above shuffled (rank IC by timestamp, all folds and seeds:
  high 0.372-0.381, low 0.420-0.449) and the close move barely (0.013-0.028).
- **None of them beats the trivial volatility baseline of rule C1** on the same test rows: the mean range of the last 6
  input bars ranks the high move at 0.394 and the low move at 0.475 (the last bar alone: 0.351 / 0.423). The models
  sit 0.013-0.022 under it on the high head and 0.026-0.055 under it on the low head.
- What the batching changed: the low-head rank IC of A2 (timestamp batches) is above A0 in 7 of 8 fold x seed pairs
  (+0.029, every fold positive: +0.026, +0.045, +0.034, +0.011) and above its size control C20 in 7 of 8 (+0.023);
  the A0 seed spread of that IC is 0.009-0.045 per fold. A2 closes half of the gap to the 6-bar baseline (-0.026
  against -0.055). A1 (day) shows the same sign, weaker (+0.017 vs A0, 6 / 8; +0.011 vs C60, 4 / 8). This was not
  the pre-registered metric and 3 heads x 4 comparisons were looked at: it is a hypothesis (below), not a result.
- Interval coverage at 80 %: A0 is under (close 0.763), the grouped and small-batch arms are nearer or above (close
  0.766-0.785, low 0.832-0.850).

## Where it is right and where it fails

Low-head rank IC (seeds pooled) by volatility tercile of the last input bar and by fold:

| group | rows | A0 | A2 | C20 |
|---|---|---|---|---|
| calm third | 13,494 | 0.354 | 0.409 | 0.369 |
| middle third | 13,494 | 0.231 | 0.269 | 0.250 |
| volatile third | 13,492 | 0.203 | 0.228 | 0.198 |
| fold 0 (2024 Q1) | 10,120 | 0.461 | 0.487 | 0.487 |
| fold 1 (Q2) | 10,120 | 0.426 | 0.471 | 0.445 |
| fold 2 (Q3) | 10,120 | 0.382 | 0.416 | 0.364 |
| fold 3 (Q4) | 10,120 | 0.411 | 0.422 | 0.406 |

The models rank best in calm conditions and lose most of their ranking in the volatile third; A2's gain over A0 is
largest in the calm third (+0.055) and smallest in the volatile third (+0.025). Fold 3 (Q4 2024, the post-election
rally) has the worst test loss for every arm (-5.63 .. -5.75 against -6.31 .. -6.42 in folds 1-2). Per coin, the
mean absolute error of the low head moves by at most 0.0002 between A0 and A2 (largest coin errors: DOGE 0.016,
AAVE 0.015, ATOM 0.014): the batching does not change WHICH coins are hard.

## The samples it failed on

The largest errors of A0 on the high head (seed 0) are upward jumps the model put at 1-4 %:

| coin | time (UTC) | predicted high | realised high | last-bar range |
|---|---|---|---|---|
| UNI | 2024-02-23 08:00 | 0.012 | 0.568 | 0.016 |
| TRX | 2024-12-03 16:00 | 0.036 | 0.388 | 0.199 |
| XLM | 2024-11-22 16:00 | 0.033 | 0.361 | 0.127 |
| BCH | 2024-03-01 16:00 | 0.020 | 0.283 | 0.015 |
| ETC | 2024-01-10 16:00 | 0.022 | 0.277 | 0.040 |
| DOGE | 2024-11-05 16:00 | 0.027 | 0.271 | 0.070 |

Two kinds: a coin-specific jump out of a quiet base (UNI on the day of its fee-switch proposal, BCH, ETC: last-bar range 0.015-0.040)
and the November-December 2024 alt-coin rally (XLM, TRX, DOGE, already volatile). The first kind is the squeeze of
B-077; no input in the window announces it.

## Bias

The close head predicts up on 47.5-50.2 % of the test rows against 50.9 % realised; its mean prediction is about 0
(-0.00026 .. +0.00003) against a mean label of +0.00091 (2024 was an up year). All arms lean slightly short; A2 is
the least biased (50.2 %). A re-centring fitted on the inner validation was not tried (not part of this question).

## Selective use

Not measured: T-profit and the battery were switched off in the pre-registration (20 coins x 90 days per fold give
too few trades for the battery's CIs). The error map above names the selection that a later study can pre-register:
the calm third, where every arm ranks best (low IC 0.35-0.41 against 0.20-0.23 in the volatile third). Rules tried
on TEST here: none.

## Why it failed

ONE cause: **the volatility signal is there but the model does not read it** (PHILOSOPHY section 3, "signal there but
not read"). A model-free ranking by the mean range of the last 6 bars beats every arm on both the high and the low
head, on the same rows. Against that gap (0.013-0.055 in rank IC), how a batch is put together changes little on the
objective: the paired test-loss differences between arms (-0.000 .. +0.067) have mixed signs and sit inside the seed
and fold spread (A0 seed spread 0.015-0.047 per fold).

## Next hypotheses

1. **H1: one-timestamp batches help the cross-sectional ranking of the low (and high) head.** Evidence: A2 vs A0
   +0.029 low IC in 7 / 8 pairs, A2 vs C20 +0.023 in 7 / 8; the size control does not show it, so it is the grouping,
   not the batch size. Refuted if, pre-registered on the low-head rank IC, a walk-forward with 60 coins and 3 seeds
   gives a paired 95 % CI that includes 0. Cheapest test: the same runner on the 60-coin slice
   (`vision_slice` default coins), A0 / A2 / C20 only, about 1.5 h CPU.
2. **H2: the model under-reads the range information already in its inputs.** Evidence: the 6-bar mean range beats
   every arm. Refuted if a ridge on `doctor/common.py::simple_features` (the HAR terms) does NOT beat the model on
   the same folds, or if the doctor's information-flow probe (`doctor/info_flow.py`) shows the range signal
   reaching the readout intact. Cheapest test: the doctor on this artifact (minutes on CPU), then, if the probe
   shows the loss, one arm with the range terms as a skip input.
3. (cost) A2 costs 2.5x the CPU time of A0 on this machine (more, smaller steps); on a GPU the gap is larger. Any
   adoption needs the GPU walk-forward on the owner's artifact (rule E5 smoke first).
