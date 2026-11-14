# Card A — Information flow inside the model

Question: at which stage of the model is information about the label lost (or never gained), and does the head use what the trunk holds?

## Run

```
python -m nig_timenet_torch.doctor e1 --out doctor_out --cards A          # synthetic E1
python -m nig_timenet_torch.doctor /path/to/artifact --out doctor_out --cards A [--run-dir RUN_DIR --trained]
```

- Inputs: the VALIDATION rows (3000 strided rows in quick mode, 12000 with `--full`) of the model the doctor trained (a short run) or of a finished run (`--run-dir ... --trained`).
- CPU is enough. On small E1 card A takes about 5.7 s (readiness.md header: `'A': 5.6` / `5.7`); the whole report takes ~80-90 s on 4 CPU cores.
- Outputs: the `A` rows of `readiness.md` / `readiness.json` and the stage table `info_flow.csv`.

## What it measures

Stream taps: `NigTimeNetV2.branch` (model/builder.py) calls `self._tap(name, tensor)` after each stage, in order: `input, scaled, inst_norm, stats, level, tokens, enc_in, enc_block_i, enc_out, pooled, readout, linear_path, branch_out`, then `trunk`; `head_<t>` = hidden layer of each NIG head. `stream_taps` (model/taps.py) records them; with an `edit` it can replace a stream. A model without `_tap` is walked by module hooks in registration order.

Signals read at every stage with a ridge probe (features plus their squares, time-ordered 4 folds with an embargo of window + horizon bars, out of fold): the labels `y_<t>` (score = rank IC), the E1 oracle means `oracle_<t>` (R2), and three tracers that are known functions of the input window (R2): `vol_level` (log mean bar range), `last_move` (last close move), `drift` (mean close move). A tracer is ~1 at the input, so any later drop is a loss inside the model.

Parent graph (`PARENTS` in info_flow.py): a stage is compared with the best of its parents, not with the stage before it, because `stats`, `level` and `linear_path` are side paths, e.g. `tokens <- (inst_norm, level)`, `pooled <- (enc_out, stats)`, `branch_out <- (readout, linear_path)`, `trunk <- branch_out`.

`info_flow.csv` columns: `stage`, `dim` (pooled width), `eff_rank` (participation ratio of the covariance eigenvalues: how many directions carry variance), `const_share` (share of features that are constant over rows), then per signal `<signal>` = probe score and `d_<signal>` = score minus the best parent's.

| check name (as printed) | what is computed | baseline |
|---|---|---|
| `<t>: probe IC at the trunk vs simple baseline` | probe IC of the label from the trunk | same probe on HAR-style simple features of the raw window |
| `<t>: IC of the model's own mu vs the trunk probe` | Spearman of the head's mu with the label | the trunk probe IC |
| `<signal>: largest loss on the way to the trunk (r2)` | the most negative `d_<signal>` ON THE ROUTE the signal takes to the trunk: walked backwards from the trunk, always through the parent stream that holds most of it, so a side path that never held the signal is not blamed (tracers and oracle only) | `input a -> trunk b` |
| `constant share / effective rank` | printed only for a collapsed stage | const share > 0.5 or eff rank < 1.5 |
| `path use: 1 - abs(corr(mu, mu with this path's rows shuffled))` | rows of ONE path (`stats, level, inst_norm, enc_out, readout, linear_path`) are shuffled, how much mu moves | 0 = outputs ignore the path |
| `<t>: training gain of the trunk probe IC` | trunk probe IC, trained minus an untrained twin (seed + 1000) | untrained IC |
| `self-check: tracer R2 across a destroyed stage (branch_out)` | the planted loss must be seen | must fall to ~0 |

## How to read it

Noise: differences below `_TOL = 0.03` are not read.

| Observation | Conclusion | What to change |
|---|---|---|
| trunk IC below baseline by more than max(0.06, 20% of baseline): FAIL; by more than 0.03: WARN. Default model: `high` trunk 0.1698 vs 0.2758 | the trunk holds less than simple features; the stage table shows which stage drops | read `info_flow.csv`, find the most negative `d_` column; fix that stage (rows below) |
| trunk and baseline both <= 0.03: info | the inputs do not carry this label | not a model fault: card C `data/information` |
| head mu IC below the trunk probe by the same margins (default: `head_high` 0.01682 vs 0.1698: FAIL) | head throws away what the trunk holds | card B: `overfit one batch`, `train vs eval mode`, A5 loss balance; model override `bn_warmup: true`, `nig_scale: "train_std"` |
| WARN `largest loss`: drop below -0.09 on the route and trunk score below max(input, baseline) minus 0.09 | the named stage erases the signal and nothing downstream restores it | give that stage a bypass (the `level` path and `linear_path` are such bypasses around `inst_norm`), or remove the normalisation that erases it; rerun card A and compare `info_flow.csv` |
| WARN `constant share / effective rank` (default: `stats` 0.00 / 1.2 of 6) | dead units or a saturated normalisation | input scale: model override `input_scaler: "train_std"`; `stats_mode` / `stats_bn` |
| WARN `path use` < 0.01 (default: `level` 5.50e-05) | outputs ignore this path: dead weight or blocked gradient | check the path gets a gradient (card B A2); remove the path or fix its scale |
| WARN `training gain` below -0.03 | an untrained twin reads as much: training does not shape the representation | card B (lr range, A4 update ratio, loss weights) |
| FAIL `self-check` | the instrument did not see a planted loss | do not trust the table; file a bug (tests/doctor/test_info_flow.py) |
| `self-check` info "tracer too weak" | tracer R2 before `branch_out` < 0.4, cannot self-check on this model | none |

To add a tap for a new layer: call `self._tap("name", tensor)` where the tensor is created in `branch` (one line; name it in the `branch` docstring; add the stage to `PARENTS` in doctor/info_flow.py, else it is compared with the stage before it). Taps change no output, state dict or loss.

## How we know the instrument works

- Built-in self-check: `branch_out` is replaced by its rows shuffled (`destroy_rows`, a fixed permutation) and the tracer `vol_level` must fall from its value before the stage to below half of it (`d < -0.5 * before` and trunk `< 0.5 * before`), else FAIL.
- tests/doctor/test_info_flow.py: planted destroyed stages are named by the table; tracers are ~1 at the input; the parent graph.
- tests/doctor/test_common.py: the purged time-folds and the ridge probe (out of fold, no leak between train and test rows).

## Limits

- A ridge probe with squares gives a LOWER bound of the information: a stage may hold a nonlinear code the probe cannot read. Judge differences between stages measured the same way, never a single score.
- Scores on a few thousand validation rows carry noise of about 0.03 (`_TOL`); smaller moves are not conclusions.
- Quick mode examines a model trained for a few epochs, not a finished run (use `--run-dir --trained`).
- Wide stages (`T x D` above 512 values) are probed as [last, mean, sd] pooled over time, so a loss that lives only in the middle of the window can be missed.
- Path use shows dependence of the outputs, not usefulness. Needs the full real artifact for any statement about real labels; E1 oracle rows exist only on synthetic data.
