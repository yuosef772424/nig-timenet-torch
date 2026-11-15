# quick_arms: encoder x lr on the Drive artifact (2026-11-14)

Command: `python docs/audit/2026-11-14-model-first/quick_arms.py /content/art1 OUT --rows 20000 --epochs 6 --seed S --arms ... --device cuda`
(artifact `preprocessing_output_1`, 535 coins; 20,000 strided rows per split; batch 128; default model = preset +
`MODEL_FIRST_FIXES`, encoder and lr changed; Colab T4, code at 8dcb9a1). Best value = selection loss (lower is better);
IC = rank IC by timestamp on the TEST rows.

| arm | seed | best value | IC high | IC low | IC close |
|---|---|---|---|---|---|
| gru 3e-4 | 0 | -6.474 | 0.397 | 0.437 | 0.032 |
| gru 3e-4 | 1 | -6.464 | 0.394 | 0.432 | 0.033 |
| transformer 3e-4 | 0 | -6.462 | 0.393 | 0.429 | 0.028 |
| transformer 3e-4 | 1 | -6.455 | 0.393 | 0.431 | 0.023 |
| gru 1e-4 | 0 | -6.376 | 0.394 | 0.431 | 0.036 |
| gru 1e-4 | 1 | -6.380 | 0.391 | 0.430 | 0.022 |
| transformer 1e-4 | 0 | -6.357 | 0.391 | 0.430 | 0.033 |
| transformer 1e-4 | 1 | -6.351 | 0.394 | 0.430 | 0.022 |

Reading:
- CPU run (seed 0, same rows) reproduces the GPU numbers to ~3 decimals (e.g. gru 3e-4: -6.473 vs -6.474).
- The best epoch is the last (6) in every arm: no overfitting yet at lr <= 3e-4; lr 1e-4 has not converged in 6 epochs.
- GRU vs transformer: IC differences <= 0.01 on high / low, the same size as the seed-to-seed spread; no claim either
  way (rule C4). The close head has no direction signal (IC 0.02-0.04), consistent with N-08.
- Not run: longer runs (where the first overfit epoch appears), more rows, the doctor on the trained weights.

## Battery (T-profit, TEST rows, cost 0.08 % per side, `--battery --n-variants 4`, seed 0, same arms)

CI = day-block bootstrap at 98.8 %. Net average per trade; "profit" needs the model to beat every baseline row AND a CI
above 0.

| arm | close direction (all rows) | LONG at pred_low (limit) | SHORT at pred_high (limit) | battery verdict |
|---|---|---|---|---|
| gru 3e-4 | -0.203 % [-0.281, -0.130] | -0.039 % [-0.138, +0.063] | -0.120 % [-0.227, -0.007] | not tradeable |
| transformer 3e-4 | -0.150 % [-0.212, -0.090] | -0.043 % [-0.142, +0.058] | -0.117 % [-0.226, -0.006] | not tradeable |
| gru 1e-4 | -0.171 % [-0.243, -0.102] | -0.032 % [-0.143, +0.080] | -0.131 % [-0.249, -0.019] | not tradeable |
| transformer 1e-4 | -0.175 % [-0.246, -0.107] | -0.041 % [-0.142, +0.063] | -0.129 % [-0.244, -0.018] | not tradeable |

- No arm has a positive row; the long-limit CIs straddle 0 but the model is not better than the "constant offset"
  baseline (gru 1e-4: model -0.032 % vs shuffled -0.062 %, constant offset -0.051 %, with overlapping CIs).
- Sigma as position size does not help (gru 3e-4: 1/sigma -0.197 % vs equal weight -0.203 % vs shuffled -0.205 %).
- Direction accuracy 0.51 on TEST; confidence is flat against accuracy; |mu| is 5 % of the true move (shrunk).
- The weekly and per-coin sections are empty or tiny (the TEST rows are 20,000 strided rows, 1 week of one coin
  group), so they say nothing: the large-sample run is the one to read.

## Models of `claude/sequential-gradient-model-uhub56` on the current Drive artifact (CPU, 2026-11-15)

`colab_ops runs itransformer,conv1d_net,gru --rows 20000 --epochs 6 --device cpu` (that branch's code bb8225b, lr
1e-4 -> 1e-6 cosine, batch 256, seed 0, 20,000 strided rows per split, `variants=("full",)`; TEST rows, battery
n_variants 1). One seed, CPU, small: a look, not a result (rule C4).

| model | val loss best | IC high / low / close | market-neutral (best of 3 q) | battery verdict |
|---|---|---|---|---|
| conv1d_net | -6.18 | 0.382 / 0.419 / 0.014 | -0.35 .. -0.45 % per period | `tradeable=True`, but every profit row negative (read the rows, not the flag) |
| gru (branch default model) | -0.70 | 0.071 / 0.088 / 0.000 | -0.33 .. -0.38 % | same |
| itransformer | -0.45 | 0.007 / 0.121 / -0.003 | -0.29 .. -0.31 % | `tradeable=False` |

- **Not comparable with the earlier report of the owner** (itransformer high IC 0.284, market-neutral +0.65 %): that
  report had 318,613 train rows over 2020-02 .. 2026-09 (about one row per coin-day) and per-trade moves of ~3 %, so it
  was a different artifact than the one now at `preprocessing_output_1` (2,506,192 4h windows, log-return labels,
  `reg_target_scale` 1.0, rebuilt 2026-11-14 01:00). The earlier artifact is not on Drive any more.
- gru and itransformer (the project's `nig_timenet_v2` with that encoder) barely learned in 6 epochs on this artifact,
  conv1d_net did. Argued, not run: that branch predates `MODEL_FIRST_FIXES` (label scale N-01, BatchNorm warm-up N-04),
  and the default model needs them when labels are raw log-returns (sd 0.02-0.03). Test: the same encoders on a
  branch that has both (merge of the two branches, or port `itransformer` / `timesnet` here).
- The `battery` flag `tradeable=True` for conv1d_net and gru is not a profit: "best strategy" is chosen on TEST among
  many rows; all model rows after cost are negative with CIs containing 0 or below.

## `steps_per_execution` (D-024) and the attention NIG head (experiment), 2026-11-15

Local CPU, the 8-coin real slice (`vision_slice`, 82,650 rows; NOT the Drive artifact), 20,000 strided rows per split,
batch 128, lr 3e-4, default model (preset + `MODEL_FIRST_FIXES`).

**steps_per_execution = 32 vs 1** (3 epochs, seed 0): the weights are identical. SHA-256 of the final state_dict:
gru `9fc6fa309f6abf36` in both runs, transformer `fbd4275b97329fb1` in both; best value -6.408851 / -6.397815 equal to
6 decimals. `tests/trainer/test_steps_per_execution.py`: bit-identical (`torch.equal`) for N in 2, 5, 32, 1000, with
and without clipping + EMA. Speed: 32 / 34 s (gru) and 32 / 32 s (transformer) per run on a 4-core CPU = no change,
as expected: the option removes one host sync per step, which costs nothing on a CPU. The GPU timing is NOT measured
yet (Colab runtime was down).

**Attention head vs Dense head** (`nig_head`, `model/attn_heads.py`: port of the AI note's `ScalarOutputAttention`
for the NIG heads only; floors, unc clip, confidence and class heads unchanged; the note's other changes - floors 0,
no unc clip, attention confidence / class heads - are NOT tested here). GRU encoder, 6 epochs, 2 seeds, val selection
loss (lower is better) and TEST rank IC (high / low / close); parameters 102,310 dense vs 104,126 attention:

| head | seed | best val loss | IC high | IC low | IC close |
|---|---|---|---|---|---|
| dense (old) | 0 | -6.491 | 0.354 | 0.424 | 0.027 |
| dense (old) | 1 | -6.482 | 0.360 | 0.428 | 0.021 |
| scalar_joint | 0 | -6.363 | 0.364 | 0.434 | 0.047 |
| scalar_joint | 1 | -6.349 | 0.355 | 0.411 | -0.010 |
| scalar_factorized | 0 | -6.342 | 0.116 | 0.420 | 0.026 |
| scalar_factorized | 1 | -6.352 | 0.130 | 0.420 | 0.001 |

Reading (a look: 2 seeds, 6 epochs, 8 coins, CPU): the attention heads give a worse validation loss by ~0.13 on both
seeds and no better IC; `scalar_joint` ties dense on IC within the seed spread; `scalar_factorized` loses the `high`
head (IC 0.12-0.13 vs 0.36). No overfit appears in any arm (best epoch = last), so the note's premise (the Dense head
memorises) is not visible at this size and the attention head cannot show its claimed benefit here. Not run: the Drive
artifact, a long run where Dense starts to overfit, the note's floors / confidence / class-head changes, the
attention-map checks across regimes (the note's own proposed test).
