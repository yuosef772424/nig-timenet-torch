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
