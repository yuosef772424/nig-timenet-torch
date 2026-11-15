# Card B — Trainer

Question: does the training procedure make the model learn, and does it learn the right thing (not a train-mode artefact, not a leak)?

## Run

```
python -m nig_timenet_torch.doctor e1 --out doctor_out --cards B
python -m nig_timenet_torch.doctor /path/to/artifact --out doctor_out --cards B --max-rows 60000 --epochs 3
python -m nig_timenet_torch.doctor /path/to/artifact --out doctor_out --cards B --run-dir RUN_DIR --trained
python -m nig_timenet_torch.doctor e1 --out doctor_out --cards B --train '{"lr": 0.003}'
```

- Every check builds its OWN fresh model (`fresh_trainer`) or restores what it changed; the trained model is untouched.
- CPU only is enough. On small E1 card B takes about 55 s (header `'B': 55.2` / `56.3`), the longest card: it trains two small sub-runs (E1 and E2, 3 epochs quick, 6 with `--full`) and a resume pair. `--full` is meant for a GPU smoke job or a longer CPU run.

## What it measures

| check name (as printed) | what is computed | baseline |
|---|---|---|
| `overfit one batch (no regularisation)` / `(as configured)` | fresh model, 64 train rows, 200 steps (400 full), dropout / noise off; worst head corr(mu, label). The second row (regularisation on) is info only | >= 0.9 (constant predictor: 0) |
| `gradients finite [A1]`, `dead parameters [A2]`, `layer gradient scale / batch spike [A3]`, `median update / weight of one step [A4]`, `head share of the trunk gradient [A5]`, `head gradient conflict (mean cosine) [A6]`, `units that never activate [A7]` | gate A of trainer/grad_report.py, measured at `(start)` = a fresh model and `(end)` = the trained model, on 4 batches (quick) | A2: grad norm > 1e-12; A3: layer RMS within 1e-3..1e3 x median, spike < 10 (fail 100); A4: 1e-4..1e-2 (fail outside 1e-5..1e-1); A5: each head >= 0.02, largest <= 0.6; A6: cosine >= -0.1; A7: <= 0.2 (0.5 fail) |
| `weight decay per step (lr x wd) / median update per step` | is weight decay a no-op | < 0.001 = no effect (info) |
| `gradient clipping / EMA` | the configured values | info |
| `lr range: configured lr vs the sweep` | exponential lr sweep 1e-6..1 on a fresh model (60 steps quick, 120 full): best lr and the lr where the loss diverges | best and diverge lr printed |
| `val loss vs constant predictor (same model's mean output)` | raw loss on validation vs the loss of the model's mean output on every row | the constant's loss |
| `picked epoch vs the best-MAE epoch (relative MAE regret)` | relative MAE at the picked epoch minus the best epoch | 0; warn > 0.02, fail > 0.10 |
| `train vs eval mode, <head>: sd ratio eval / train (corr)` | on strided TRAIN rows: sd of mu in eval mode / sd of the train-mode signal (8 train-mode passes; the signal's variance is the covariance of the means of two halves of the passes, so their noise does not count), and their corr | (0.8, 1.25), corr >= 0.9 |
| `train-mode layer that changes the function most (its sd ratio)` | the same ratio with ONE layer group in train mode at a time: dropout, feature dropout, input noise, batch norm (`MODE_COMPONENTS`) | 1.00 = that layer changes nothing |
| `batch norm moving variance vs a train batch: <layer>` | worst ratio of the moving variance to a train batch's variance | <= 2.0 |
| `resume vs uninterrupted: max weight difference` | 2 epochs in one go vs 1 + resumed 1 | <= 1e-6 (0 bitwise on CPU) |
| `E1 oracle, <head>: corr(mu, true mean) / slope` | on synthetic E1 with a known conditional mean: corr and slope of the oracle on mu | corr >= 0.9 (high, low), 0.5 (close); slope (0.75, 1.33); the close head only warns (planted IC 0.05, `E1_WARN_ONLY`) |
| `shuffled labels (E2): worst abs IC on test` and `train loss below the constant's` | same configuration trained on E1 with permuted labels | IC < 0.06 and val loss not below the constant's by 5%; the train row is info (memorising) |

## How to read it

Verdict (readiness.md diagnosis line): "does not learn" is decided by FAIL of `overfit one batch (no regularisation)`, `gradients finite`, `dead parameters`, `val loss vs constant predictor`, `lr range` (constant `NOT_LEARNING`); "learns the wrong thing" by FAIL of `shuffled labels`, `train vs eval mode`, `E1 oracle`, `batch norm moving`, `resume` (`WRONG_THING`); otherwise "learns", and other failing rows (A3-A7, selection) are listed as health checks that do not decide.

| Observation | Conclusion | What to change |
|---|---|---|
| overfit corr < 0.5 FAIL, < 0.9 WARN (default model 0.267; fixed 0.9654) | cannot fit 64 rows: loss, head parametrisation or optimiser step is broken | A4 and lr range rows; model override `nig_scale: "train_std"`, `input_scaler: "train_std"`; train `lr` |
| A1 fail | NaN / inf gradient | find the layer with `grad_report(...)["layers"]` (trainer/grad_report.py) |
| A2 warn / fail (12 tensors, both reports) | tensors never get a gradient | a detached path or a loss term that is off (e.g. `order_detach_close`, loss weights) |
| A3 fail (`1 out of band`) | one layer's gradient is orders of magnitude off | init or normalisation of that layer |
| A4 warn 0.01369 (just above 0.01) | step too large / small for the weights | train `lr`, init scale |
| A5 fail (default: `high_class` 0.021, `close_class` 0.008 at end) | one head owns the trunk or a head is starved | loss weights / task weighting (`uncertainty_weighting`, `class_loss_weight`, `lambda_reg_end`) |
| A6 warn / fail | heads pull the trunk in opposite directions (default `high_reg/low_reg` -0.75) | separate or detach the conflicting head |
| A7 fail (default 0.5156) | half of the units never activate | input scale / init of that layer: `input_scaler: "train_std"` |
| lr FAIL: configured lr > diverge / 3; WARN: lr < best / 100 | lr too close to divergence / far too low (E1: best 2.4e-02, diverges 3.1e-01) | train `lr` |
| val loss FAIL (not below its constant) | nothing generalising was learned | check cards C, A before touching the trainer |
| selection regret WARN > 0.02 / FAIL > 0.10 | selection metric picks a worse epoch | train `selection_metric`: a target metric or a shorter smoothing |
| mode gap FAIL (pre-audit model on E1: `high` sd ratio 0.16, corr 0.14) | the evaluated function is not the trained one | read the `train-mode layer` row: `batch norm` far from 1 with a failing `batch norm moving variance` row -> model override `bn_warmup: true`; `dropout` / `feature dropout` below 0.8 -> lower that rate (`dropout`, `readout_dropout`, `feature_dropout`) or recalibrate mu on validation. Real slice, new default: 0.87 overall (dropout 0.93, feature dropout 0.94, noise 1.00, batch norm 1.00) = pass. The rows are TRAIN rows on purpose: on a split in another volatility regime BatchNorm in train mode uses that split's statistics and the ratio measures the regime change, not the model (NOTES N-11) |
| BN variance ratio FAIL (default `level_bn` 377.2, `stats_bn` 445.1; fixed 1.001 / 1.013) | moving statistics did not reach the data's | model override `bn_warmup: true` |
| resume FAIL | a resumed run is not the run it continues | state missing from the checkpoint (trainer/checkpoint.py) or the per-epoch reseed |
| E1 oracle FAIL (default `high` corr 0.01 / slope 0.04; fixed `low` 0.95 / 1.77) | known mean not recovered; slope > 1 = mu shrunk, low corr = signal not learned | mode gap, head scale, then longer training (see Limits) |
| E2 FAIL | skill on permuted labels | a leak between inputs and labels, or an evaluation that rewards something trivial: card C alignment first |

## How we know the instrument works

- E1 has a known oracle (the true conditional mean of each label): a healthy trainer must recover it; the check fails on the pre-audit model (corr 0.01 / 0.34 / 0.11) and mostly passes on the fixed one.
- E2 is the same data with permuted labels (`permute_labels`): a correct trainer finds no skill (fixed model: worst abs IC 0.02278, pass).
- tests/doctor/test_trainer_checks.py: planted faults are named by the matching check, and `verdict` maps the check names to the three verdicts.
- tests/doctor/test_common.py: `fresh_trainer` and `predict_outputs` leave the trained model and BatchNorm statistics unchanged.

## Limits

- Quick mode trains only a few epochs (3) on strided rows: a FAIL on E1 oracle may only mean "not converged yet" (fixed model after 3 epochs: `close` corr 0.43 against 0.5). Re-run with `--full --epochs 10`, or on a finished run with `--run-dir RUN_DIR --trained` (the E1/E2 sub-runs are still short).
- E1 / E2 are skipped (info row) unless the configuration reads 3 input channels and the architecture is `nig_timenet_v2` or `tiny_mlp`.
- One seed: gate A and the lr sweep are single measurements; do not read small differences.
- The doctor does not prove a long run converges; GPU and the full real artifact are needed for real training behaviour.
