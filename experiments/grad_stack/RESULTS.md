# grad_stack results — synthetic E1 (32 coins x 2000 rows), CPU, 2026-11-10

Test split, project's own test (`target_report` + `gates.stage_rows`). H1 = 12,768 rows, H2 = 12,800 rows,
val 2,656, test 2,688. Gradient target = `output` (label-free).

## tiny_mlp, 8 epochs, 3 seeds (mean ± sd)
| metric | A (H1) | data_h2 | grad | grad_data | full |
|---|---|---|---|---|---|
| high IC | -0.073±0.102 | -0.011±0.086 | 0.112±0.056 | 0.189±0.094 | 0.246±0.019 |
| low IC | 0.077±0.080 | 0.075±0.034 | 0.204±0.085 | 0.170±0.014 | 0.307±0.025 |
| close IC | 0.020±0.020 | 0.023±0.028 | 0.001±0.033 | -0.003±0.035 | 0.069±0.007 |
| high AUC | 0.537 | 0.537 | 0.568 | 0.568 | 0.571 |
| low AUC | 0.492 | 0.448 | 0.591 | 0.600 | 0.585 |
| close AUC | 0.497 | 0.501 | 0.502 | 0.503 | 0.514 |
| gate checks passed (of 20, per seed) | 7/8/9 | 6/6/9 | 9/6/3 | 5/7/10 | 15/16/15 |

## nig_timenet_v2 (project default), 3 epochs, seed 0
| metric | A | data_h2 | grad | grad_data | full |
|---|---|---|---|---|---|
| high IC | 0.201 | 0.207 | 0.296 | 0.294 | 0.304 |
| low IC | 0.268 | 0.267 | 0.338 | 0.345 | 0.354 |
| close IC | 0.058 | 0.049 | -0.064 | -0.018 | 0.064 |
| high / low / close AUC | .707/.756/.528 | .690/.747/.526 | .702/.757/.528 | .705/.757/.529 | .758/.804/.530 |
| gate checks passed (of 20) | 9 | 9 | 7 | 6 | 17 |

## Reading
- Gradients beat the data-only model on the SAME rows (high/low IC up ~0.1) — but they carry model A's knowledge of H1,
  so the fair reference is `full` (all train rows, data only): `full` is better or equal everywhere, and much
  better on the gates (calibration / uncertainty). No gain over plain training on all the data.
- Close head (the direction): gradients make it worse (v2: IC 0.058 -> -0.064).
- `--grad-target loss` leaks the label (test AUC = 1.00): invalid by construction, upper bound only.
- Not run: the real artifact (not in this container), more seeds of v2, GPU.

## grad_full (B sees only A's FULL per-layer gradients, per-layer encoders) — synthetic E1, 2026-11-10
tiny_mlp, 8 epochs, 3 seeds (mean ± sd): high IC 0.095±0.079, low IC 0.025±0.028, close IC -0.003±0.007,
high/low/close AUC 0.53/0.49/0.50; gate checks passed 6/9/8. nig_timenet_v2, 3 epochs, seed 0: high/low/close IC
0.089/0.286/0.088, AUC 0.687/0.731/0.481, gate checks 6/20; 100 s vs 17 s for A (~6x cost: 9 gradient passes / step).
Reading: on E1 the full gradients are worse than the summarized ones (`grad`) and than `full`; the best epoch was the
last one (3 of 3), so it may be under-trained; one 12.8k-row E1 is small for ~6k inputs per row. Not run on the real
artifact yet.
