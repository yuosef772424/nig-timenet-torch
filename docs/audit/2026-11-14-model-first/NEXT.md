# Model-first audit — what remains (2026-11-14)

Read `NOTES.md` first. The loop this round leaves behind:

`run the checks -> the report names the stage -> fix that one thing -> run the checks again -> compare`

## 1. Commands

```bash
# everything, synthetic data with a known answer (~90 s on a CPU)
python -m nig_timenet_torch.doctor e1 --out doctor_out/e1

# real candles without Drive: build once (public 4h klines, sealed coins refused), then run (~4 min on a CPU)
python -m nig_timenet_torch.data_io.vision_slice /data/real4h --raw /data/raw4h
python -m nig_timenet_torch.doctor /data/real4h --out doctor_out/real

# the owner's artifact, or a FINISHED run's best weights (GPU smoke job or Colab)
python -m nig_timenet_torch.doctor /content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1 \
    --out doctor_out/drive --full
python -m nig_timenet_torch.doctor ARTIFACT --run-dir RUN_DIR --trained --out doctor_out/run

# after a fix: same command, another --out, then compare the "Verdict per card" tables and readiness.json
python -m nig_timenet_torch.doctor /data/real4h --out doctor_out/real_after --model '{"dropout": 0.1}'

# the model before this audit, for comparison
python -m nig_timenet_torch.doctor e1 --out doctor_out/e1_old \
    --model '{"bn_warmup": false, "order_detach_close": false, "nig_scale": null, "input_scaler": "none"}'
```

Cards: `cards/A_information_flow.md`, `B_trainer.md`, `C_data.md`, `D_metrics.md`, `E_readiness.md`.

## 2. What the readiness report says today (new default, real slice, 3 epochs, one seed)

Not ready for trading. In reading order:

1. Data: sound (alignment, purge, codec, units all pass). The test period is another volatility regime (warn).
2. Trainer: learns. Fails left: the E1 oracle slope of the low head (1.65: mu shrunk) and the short E1 run's
   close signal (warn).
3. Model: no stage loses the labels' information; the trunk holds what the simple features hold (low: 0.17 vs
   0.22 on 3,000 rows, a warning inside the probe's noise).
4. Inputs: no readable signal for the close label (ceiling IC 0.013). **This is the stage to work on**: the high /
   low heads already sit at the simple-model ceiling (0.338 vs 0.339, 0.388 vs 0.411), so a better model of the SAME
   inputs cannot become profitable through direction.
5. Goal: no use of the outputs is profitable after 0.08 % per side.

## 3. Ordered by expected effect on the model

| # | what | why | what settles it |
|---|---|---|---|
| 1 | New inputs for direction (volume, taker flow, funding, cross-coin context) | N-08: the ceiling for close is ~0 on 3 log-move channels; the kline files already carry volume and taker-buy volume (`vision_slice` reads the first six columns only) | add channels to `build_logret_dataset`, rerun card C: the close ceiling must rise above 0.02 before any model work on direction |
| 2 | Confirm the new default on the Drive artifact, 3 seeds, full length (GPU) | N-07 is one seed, 3 epochs, 60 coins | `baseline` job with the default config x 3 seeds, then `doctor --run-dir ... --trained --include-test`; accept if high / low IC and NLL beat the pre-audit seeds on every seed |
| 3 | Low-head shrinkage (O-1) | slope of the label on mu 1.6-1.7 on the low head, 1.2 on high, on E1 and on real data; dropout dilution explains ~0.87 of spread, not 1.6 | (a) fit a NIG head on PERFECT features with skewed labels: if the slope stays > 1 the likelihood's location is the cause (mode, not mean); (b) arms with `readout_dropout` 0.1 and `feature_dropout` 0; (c) a registered affine recalibration of mu fitted on validation (`eval/calibrator.py`) and its effect on the limit-order rows |
| 4 | Model size against independent labels | N-10: 14 parameters per independent label on 60,000 rows; the full train split of the slice (565k rows) gives ~68k independent labels for ~100k parameters | card C on the full rows; arms `d_model` 32 / 16 on the real slice: val loss and IC must not fall |
| 5 | Learning rate | N-14: 1e-3 is 15-38x below the sweep's best | arms lr 3e-3 and 1e-2 with warm-up on the real slice, 3 seeds; read card B (A4, mode gap) and val IC |
| 6 | Selection | N-13: smoothed selection keeps the wrong epoch's weights; NaN validation never stops | keep the weights of the epoch with the best RAW value inside the smoothing window; count a non-finite value as a wasted epoch; a rank-IC selection metric for the high / low heads |
| 7 | E2 gate B alarm with the fitted scale | FX1 saw a confidence-gap alarm with `nig_scale` 0.01 on permuted labels | `gates` job on the new default, 3 seeds |
| 8 | `model/probe.py` folds | N-15: row folds on coin-major rows | reuse `doctor.common.time_folds` in the health report's probe |

## 4. Needs a GPU smoke job

- `python -m nig_timenet_torch.doctor ARTIFACT --full` end to end on CUDA (the GRU backward in eval mode of gate A
  needs cuDNN off: `trainer/trace.py` handles it; the doctor's own passes are forward-only or train-mode).
- Mixed precision: wrap the NIG heads (and `ordered_means`) in `autocast(enabled=False)` before reading T-sharp or
  coverage; `InputScaler` and `FittedScale` buffers are float32 and must stay so; the loss already casts to float32.
- `torch.compile`: `model._tap` reads a Python attribute per call (a guard, not a graph break); recompile after
  `stream_taps` is entered, or run the doctor on the uncompiled module.
- Step time with the input scaler (one subtraction and division per batch).

## 5. Open questions

- O-1: is the slope > 1 of the low head a property of the likelihood under skew or of the training? (table row 3)
- O-2: N-09's AUC 0.53 on the close class head: drift of the period or timing? Settle with the same check on test
  and on a second period, next to always-up / always-down.
- O-3: does `input_scaler` matter on real data on its own? One arm without it on the real slice.
