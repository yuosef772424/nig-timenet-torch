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
