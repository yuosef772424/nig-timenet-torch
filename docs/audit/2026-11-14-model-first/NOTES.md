# Model-first audit — running notes (2026-11-14)

Task: `docs/audit/FABLE_DESIGN_AUDIT_PROMPT.md`. Branch `ccr-adb40d84-31jfyt`, PR #55. One row per problem: what,
where, evidence (command + result), action, and which recorded results it puts in doubt. Everything was run on a
4-core CPU, no GPU, no Drive. "E1" = the synthetic environment with a known answer (`data_io/synthetic.py`); "real
slice" = 60 unsealed coins, 4h bars 2021-01 .. 2026-09, 706,104 windows built from public klines
(`data_io/vision_slice.py`) with the real artifact's definition. It is NOT the owner's Drive artifact (535 coins).

Confidence words: **proved** = measured by a run named in the row; **argued** = follows from reading code, not run;
**not looked at** = out of reach here.

## 1. Findings

| id | what | where | evidence | action |
|---|---|---|---|---|
| N-01 | The port skipped the TF normaliser. TF fits `logret_std` at split time (inputs AND targets standardised with train statistics, clip 5); the artifact is exported `price_norm_state: unfitted` and this project trained on the raw values (inputs sd ~0.02, labels sd ~0.03). Every unit-ful constant then acts on the wrong scale: NIG `beta_min` 0.01, the softplus offsets of the ordered means, InstanceNorm `eps` 1e-4. | TF `data/normalizers.py:296-358` (LogretStd); here `workflow/flat_run.py:flat_inputs`, `model/nig_layers.py`, `model/layers.py:InstanceNorm` | proved: card C units on E1 and on the real slice, pre-audit model: label sd in head units 0.02-0.03 (floor sd 0.10), median window variance / eps 1.9 (E1); real slice median window variance 1.3e-4 .. 3.0e-4 vs eps 1e-4 | **fixed**: `model/scalers.py` (InputScaler, fitted NIG label scale), fitted on train rows in `workflow.run.build`, state in the model's buffers; default since `MODEL_FIRST_FIXES` |
| N-02 | Label side of N-01 is the part that matters. | E1 arms, 3 seeds (`python -m nig_timenet_torch.workflow.e1_arms`) | proved: blocking gate fails per seed 7.7 (pre-audit) / 5.7 (FX1 bundle + hand-set `nig_scale` 0.01) / 3.3 (bundle + fitted scale) / 2.7 (+ input scaler) at 3 epochs; 7.3 -> 2.7 at 10 epochs (fitted scale, no input scaler); T-sharp high / low 0.62 / 0.54 -> 0.92 / 0.98, close corr with its true mean 0.11 -> 0.88 (10 epochs) | fixed with N-01 |
| N-03 | Input side of N-01 has no measurable effect on E1 while the level path with BatchNorm exists (that BatchNorm already standardises the raw level). | same arms: 5.7 -> 5.3 fails, mu calibration unchanged; one-batch overfit unchanged | proved (E1 only) | kept in the default because it makes the model independent of the level path for its input scale and makes InstanceNorm's statistics meaningful; **argued**, weakly measured |
| N-04 | The pre-audit model cannot fit 64 rows. Worst head corr(mu, label) after 200-400 steps, no regularisation: 0.27 (E1), 0.57 (real slice). Cause isolated by switching one thing at a time: the BatchNorm moving statistics (eval reads statistics that have not left their init, B-062) and the ordered-means coupling (B-063); with `bn_warmup` + `order_detach_close` 0.98 / 0.98 / 1.00. | card B "overfit one batch"; `doctor/trainer_checks.py:overfit_one_batch` | proved | fixed (both are in the default) |
| N-05 | The pre-audit model's level path and encoder were dead at evaluation in short runs: shuffling the `level` stream changed mu by 5e-5, the encoder output by 0.03, while `linear_path` changed it by 0.99. With the fixes: level 0.20, encoder 0.47, linear path 0.70. | card A "path use" | proved (E1 quick) | fixed by N-04's switches; the instrument stays |
| N-06 | At init the high and low regression heads pull the trunk in opposite directions (mean gradient cosine -0.75) and own it (high_reg share 0.51, class heads 0.01-0.02): both push the shared close mean through the ordered offsets. With the fixes: worst cosine -0.08, largest share 0.30, every head >= 0.035. Units that never activate at init: 0.45-0.52 -> 0.0. | card B gate A5 / A6 / A7, start phase, E1 and real slice | proved | fixed (closes the A5 / A7 part of B-060 / B-061 for the start of training; A7 at the end of the real 3-epoch run is 0.25 = warn) |
| N-07 | Real slice, pre-audit vs new default, same rows and seed (60,000 strided rows per split, 3 epochs): card fails A / B / C / D 1 / 11 / 4 / 8 -> 0 / 5 / 0 / 5; val rank IC high 0.315 -> 0.338 (simple-model ceiling 0.339), low 0.370 -> 0.388 (ceiling 0.411); MAE skill high -0.021 -> +0.060; NLL gain over climatology 0.24 / 0.71 / 0.13 -> 0.47 / 1.00 / 0.26 nats; slope of the label on mu (high) 1.77 -> 1.21; val raw loss -5.18 -> -5.95. | `/home/user/work/doc_real_{old,new}/readiness.md` (not committed: commands in NEXT.md) | proved, ONE seed, short run, strided rows | the default change stands; a 3-seed long run on the Drive artifact is the confirmation (NEXT) |
| N-08 | The inputs hold no readable signal for the close label: ridge and gradient boosting on HAR-style features score rank IC 0.013 / 0.015 (val / test) on the real slice; E1 small 0.04 / -0.03 (planted IC 0.05). High / low ceilings 0.34 / 0.41. | card C "simple-model ceiling" | proved on the real slice; agrees with T5r on the Drive artifact | documented. No model or trainer change fixes this: direction needs inputs that carry it (volume, funding, cross-coin context: B-055, B-059) |
| N-09 | The close CLASS head has val AUC 0.53 with a day-block CI above 0.5 on the real slice, for both models (0.5335 [0.511, 0.556] and 0.5286 [0.505, 0.551]), yet the trade row equals "always down" (-0.041 % vs -0.042 % net per trade): the head learned the period's drift, not timing. | card D direction row and profit rows | proved, one split | documented; an AUC must be read next to its always-up / always-down profit row |
| N-10 | Rows are far from independent. Real slice: window overlap with the next row 0.97, label overlap 0.50, coins move together; 60,000 training rows are ~7,200 independent labels, and the default model has 14 parameters per independent label. | card C "independence" (`data_checks.overlap`) | proved for the estimate's arithmetic; the estimate itself is a rough one (pairwise correlation of the close label only) | documented; this is the measured form of the memorisation worry (B-038) |
| N-11 | The test period of the real slice is another volatility regime: worst sd ratio against train 0.58. Train-mode BatchNorm on such rows normalises with THEIR statistics, eval with train's: measured on val rows the eval / train spread ratio was 0.81 from BatchNorm alone. | card C shift; mode-gap localisation run | proved | the mode-gap check now uses TRAIN rows (ratio 1.00 for BatchNorm there) and names the layer type; the regime change itself is documented |
| N-12 | Dropout dilution is real but moderate: evaluated mu has 0.87 of the trained function's spread (dropout 0.93, feature dropout 0.94, input noise 1.00, BatchNorm 1.00). The low head's slope of the label on mu stays 1.6-1.7 (high 1.2). | `doctor/trainer_checks.py:mode_gap` on the real slice | proved (one model) | documented; open question O-1 |
| N-13 | The selection can pick a worse epoch: `BestTracker` smooths over 3 epochs and keeps the weights of the epoch where the SMOOTHED value is best; E1 quick picked epoch 3, best MAE at epoch 2 (relative MAE regret 3.3 %). A non-finite validation value is ignored, so a run whose validation loss is NaN never stops early. | `trainer/tracker.py:71-90`; card B "picked epoch" | proved (regret); argued (NaN) | documented, not changed (NEXT) |
| N-14 | Weight decay does nothing: lr x wd = 1e-7 per step against a median update of 2e-3 (ratio 5e-5). Configured lr 1e-3 is 15-38x below the best of the lr range sweep (1.5e-2 real, 3.8e-2 E1) and far from divergence. | card B optimiser rows | proved | documented (confirms B-041); not changed: a higher lr needs a long-run check |
| N-15 | The health report's layer probe (`model/probe.py`) cuts folds over ROWS; artifact rows are coin-major, so its folds split by coin, not by time, and have no embargo. | `model/probe.py:91` | argued | not changed; the doctor's probe uses time-ordered folds with an embargo (`doctor/common.py:time_folds`) |
| N-16 | The model could not answer "where is the residual stream": module hooks see a block's output, not the sum. | `model/builder.py`, `model/encoders.py` | — | **fixed**: `model._tap` stream taps in forward order, editable for ablations (`model/taps.py:stream_taps`) |
| N-17 | Threaded LAPACK on a busy machine: one 576 x 576 solve takes 0.6 s on 4 threads and 4 ms on one. | measured | proved | the doctor's ridge solves on one thread (`doctor/common.py:_ridge_path`) |

## 2. Behaviour changes made on purpose

1. `workflow.settings.default_model_settings()` = anti-memorization preset + `MODEL_FIRST_FIXES`
   (`bn_warmup`, `order_detach_close`, `nig_scale="train_std"`, `input_scaler="train_std"`). New buffers in the
   state_dict (`input_scaler.mean/std`, `*.fitted_scale.value`), a new run identity. Reason: N-01, N-02, N-04, N-06.
   Old model: `ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG))`; doctor flag
   `--model '{"bn_warmup": false, "order_detach_close": false, "nig_scale": null, "input_scaler": "none"}'`.
2. `workflow.run.build` calls `model.scalers.fit_scalers` on the train rows (a no-op for a model without scalers).
3. A model built with the new options and not fitted raises on its first forward (never a silent identity).
4. Outputs are unchanged in meaning: mu / beta stay in the artifact's label units, so the codec, the prediction
   table and every report read them as before.

## 3. Recorded results put in doubt

- Every number of `docs/results/baseline-001.md` and `deep-side-001.md` describes the PRE-audit model. Its shrinkage
  (B-058), interval width (B-052) and head balance (B-060) were partly artefacts of N-01 / N-04 / N-06. The verdict
  "no profit" is not put in doubt by anything here (N-08).
- `docs/results/e1-fixes-001.md`: its recommended bundle is confirmed; its `nig_scale` 0.01 arm is superseded by the
  fitted scale (N-02). Its "E2 gate B alarm" for nig_scale was not re-measured with the fitted scale (NEXT).
- Gate reports of the baseline configuration are for a configuration that is no longer the default.

## 4. Not verified

- No GPU: step time, cuDNN paths, mixed precision on CUDA. CPU bfloat16 autocast runs (finite loss and gradients,
  mu differs by 0.3 % median, 4 % max): the heads should stay in float32 before any sharpness number is trusted.
- Not the Drive artifact (535 coins, 2.5 M windows) and no long run: every real-data number is one seed, 3 epochs,
  60,000 strided rows per split.
- `input_scaler` on real data in isolation (only the whole bundle was run there).
- Multi-timeframe models with the input scaler (the fit handles a dict of inputs; no test ran it).

## 5. Doctor on trained weights, Drive artifact (2026-11-14, second session)

Colab T4, artifact `preprocessing_output_1` (535 coins, 2,506,192 windows), `quick_arms` weights (seed 0, lr 3e-4, 6
epochs, 20,000 strided rows per split), `doctor ARTIFACT --run-dir RUN/ckpt --trained --max-rows 20000`.

| id | what | evidence | action |
|---|---|---|---|
| N-18 | The doctor's alignment check (card C) FAILED on the Drive artifact for low / close (max err 0.61 / 0.34): a false alarm. 99.99 % of 600,000 sampled pairs match exactly (median / q99.9 error 0); the few misses (close 1e-5, low 8e-5 of pairs) are rows whose label is exactly +-1.0, the TF export's clip (B-054). The artifact records no `reg_target_clip`, so the check did not skip saturated labels. | `align_dist.py`-style error distribution on the artifact; after the fix card C = 16 pass / 1 warn / 0 fail | **fixed**: `doctor/data_checks.py:label_clip` (the artifact's key, else the mode's registry default); test added |
| N-19 | Readiness on the Drive artifact, GRU vs transformer (same rows, seed 0, lr 3e-4): both "not ready", same stages. A 0 / 0 fail, B 3 fail (A6 head conflict at the end, close mode gap, E1 oracle low slope 1.59 GRU / 1.47 transformer), D 5 (GRU) / 7 (transformer) fail: the transformer's close head is no better than a constant (MAE skill -5e-5, slope 0.49). Profit rows negative for both: close -0.140 % / -0.135 % per trade net, limit-long -0.067 % / -0.071 %, short -0.179 % / -0.187 % (all CIs contain nothing above 0 or lose to a trivial rule). Slope of the label on mu: high 1.91 / 1.84, low 1.60 / 1.45 (mu shrunk, O-1 again, on the real artifact). | `/content/doc2_{gru,transformer}/readiness.md` on Colab (not committed) | documented; one seed, 6 epochs, 20,000 rows: a decision needs the large-sample run |

## 6. Transformer blocks, SwiGLU, encoders and the output head (2026-11-15, third session)

Numbers, commands and tables: `TRANSFORMER_HEAD_RESULTS.md`. Real slice (60 coins, NOT the Drive artifact), CPU, 2
seeds per arm unless a row says otherwise.

| id | what | evidence | action |
|---|---|---|---|
| N-20 | The outputs go around the sequence encoder. Information reaches the transformer blocks and leaves them unchanged (probe IC of `y_high` 0.268 at `enc_in`, 0.250 at `enc_out`), but with ALL blocks switched off the label rank IC moves by <= 0.02 (of 0.33-0.39) and with the encoder output shuffled by <= 0.05, against -0.08 .. -0.15 for `linear_path` and -0.03 .. -0.10 for `stats`. Attention entropy 0.91-0.95 of uniform: it averages. Same picture with the GRU encoder. | card A rows `attn / ffn / block use`, `model/all blocks`, path-use detail `d_ic_<t>`; trained transformer and GRU, validation rows | **instrument added** (`doctor/block_use.py`, card A); documented. Proved on the real slice at the default size; consistent with N-08 (three log-move channels hold a volatility forecast a linear map already reads) |
| N-21 | SwiGLU (72 % of a block's weights) is not needed: `ffn: none` gives best val -5.919 against -5.906 with it (equal or better on both seeds), IC equal, 45 % fewer weights, 30 % less time; moving the saved weights into attention (4 blocks, full kv heads) gives -5.914. | section 2 of the results file | **switch added** (`ffn`, `ffn_dim`; default unchanged = SwiGLU). Recommended for transformer runs; NOT made the default: the default encoder is the GRU (the switch does nothing there) and the owner's size (d_model 384, 6 blocks) was not run |
| N-22 | A transformer with per-bar tokens and a causal conv stem (`conv_transformer`) is level with the GRU (-5.921 / -5.925 without SwiGLU / GRU -5.929); the patch transformer with SwiGLU is the weakest arm (-5.906) on both seeds. No arm changes the IC. | section 3 | **encoder added** (registered, default unchanged). The encoder is not the limit (N-20) |
| N-23 | Output head: Dense, attention (`scalar_joint`) and ONE Linear layer (`linear`, new control) tie at 20,000 rows (-5.906 / -5.902 / -5.900, same best epochs and run length). At 60,000 rows the sign depends on the trunk, consistently over both seeds: attention head +0.013 on the SwiGLU transformer, -0.018 on the transformer without SwiGLU, -0.033 on the GRU (best arms: Dense on GRU -5.956, Dense on the no-SwiGLU transformer -5.949). | section 3a | documented; `linear` head added. The attention head helps only the weakest trunk (it acts as a regulariser there); this is the source of the contradicting earlier results |
| N-24 | The attention head starts as a near-constant function (output sd across rows 0.025 vs 0.16 for Dense at init; range bounded) and must grow its queries / gain first, so at lr 1e-5 it is far behind at equal epochs (-5.541 vs -5.766 at epoch 37; Drive artifact epoch 1: -5.513 vs -6.089). "Trains for more epochs" follows from the parametrisation; it is not by itself evidence of less memorisation. | init measurement in section 3a; lr 1e-5 curves | documented. The earlier "Dense better" Colab table (10-epoch limit, GRU) compared the heads before the slower one had finished |
| N-25 | `quick_arms` had no early-stop control and no model override: every earlier arm table ended at its epoch limit. | this session | **fixed**: `--patience`, `--model`, `--tag`, `--card-a` |

Behaviour changes of this session: none (new options `ffn`, `ffn_dim`, encoder `conv_transformer`, head `linear`; every
default builds the same model, state_dict keys and weights as before: tests/model/test_assembly_pins.py unchanged).

## 7. Direct input path, multi-coin model, targets (2026-11-15, fourth session; branch `claude/audit-models-output-layer-90e1uu`)

Tables: `TRANSFORMER_HEAD_RESULTS.md` sections 5-7. Real slice unless a row says Drive.

| id | what | evidence | action |
|---|---|---|---|
| N-26 | A direct line from the input into the transformer blocks changes nothing: `input_skip` concat_each_block / dense_skip / late_raw give -5.910 / -5.906 / -5.900 against -5.906 without (2 seeds). | section 5 | documented; no new code (the three paths existed). Agrees with N-20: the blocks are not starved |
| N-27 | 60 % of the close label's variance (81 % of low, 38 % of high) is the move all coins share at that timestamp. A relative TARGET does not raise the simple-model ceiling; features RELATIVE TO THE TIMESTAMP do: `low` rank IC +0.04 on val and test (ridge), close from -0.01 to +0.01. | section 6a (`rel_target` check: ridge on simple features, 150,000 rows of whole timestamps) | documented: the target preparation is sound for the NIG heads; the gain is on the input / trunk side. A rank head next to the NIG heads is a later option |
| N-28 | Multi-coin model built inside the default architecture (`cross_coin_layers`, kinds `attention` / `mean`, identity at init; timestamp-id channel; `SplitSettings.cross_section`; whole-timestamp batches). At 200,000 rows the `mean` kind raises the TEST rank IC on both seeds: low 0.403 -> 0.425, close 0.013 -> 0.052; attention 0.419 / 0.033; validation loss equal; at 60,000 rows no effect. | section 6b; tests/model/test_cross_coin.py, tests/workflow/test_run.py | **built** (D-025, defaults unchanged). `model/panel.py` (TF port) left parked |
| N-29 | Whole timestamps at a fixed row budget are a poorer sample than rows strided over all dates: 60,000 rows as 1,000 timestamps reach -5.82 (best epoch 1 with shuffled batches), strided rows -5.956. | section 6b arm A vs section 3a | documented: cross-section runs need more rows; compare arms only on the same row set |
| N-30 | The doctor and the health report do not run on a cross-section run (they build their own inputs without the timestamp channel; the model then treats every row as alone). | read, not run | open: pass `cross_section` through `DoctorContext.xy` and `health_step` before reading a cross-coin model with them |
| N-31 | Drive artifact, L4: Dense beats the attention head on the no-SwiGLU transformer at lr 3e-4 to the stop (-6.541 vs -6.505); at lr 1e-5 the attention head is still 0.09 behind after 58 epochs, neither stopped. The multi-coin block's slice gain is NOT repeated on the Drive artifact (seed 1: low IC +0.010 with `mean`, close unchanged, val loss worse). | section 7 | documented; seed 0 of 7b and the last epochs of the lr 1e-5 pair were still running |
| N-32 | The owner's `1-e5` runs (close IC 0.07-0.08, market-neutral +0.65 %) used another dataset: ~one row per coin-day, 318k train rows, labels 6-7x larger than the present artifact's; that dataset is no longer on Drive and its label definition is not recorded. The +0.65 % row is the TEST-chosen ratio over 71 periods, one seed; VAL-chosen +0.21 %. | section 8 (subagent read of the Drive folder) | documented; open: the dataset definition (owner) |

