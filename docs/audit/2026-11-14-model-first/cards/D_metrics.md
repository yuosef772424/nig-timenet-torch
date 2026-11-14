# Card D — Metrics and profit

Question: do the reported numbers mean what we say, next to trivial baselines, and does the model make money after cost?

## Run

```
python -m nig_timenet_torch.doctor e1 --out doctor_out --cards C,D      # D reads the ceiling that C computes
python -m nig_timenet_torch.doctor /path/to/artifact --out doctor_out --run-dir RUN_DIR --trained --include-test
```

- Scored on VALIDATION rows; `--include-test` also scores test. About 0.3-0.4 s on small E1 (header `'D': 0.3` / `0.4`) once the model exists. CPU is enough. Without card C the rank IC row has no simple-model baseline and is info.

## What it measures

| check name (as printed) | what is computed | baseline |
|---|---|---|
| `rank IC by timestamp: model vs the simple model on the same inputs` | rank IC of mu with the label, per timestamp then averaged | best ridge / boosting IC by timestamp from card C |
| `MAE skill vs the constant predictor (1 - MAE / MAE constant)` | point forecast error | train median (0) |
| `NLL gain over climatology (train mean / sd Gaussian), nats per row` | Student-t NLL of the NIG predictive vs a Gaussian with train mean and sd | 0 |
| `coverage of mu +- 1.28 predictive sd / median sd vs climatology sd` | share of labels inside the 80% interval; sharpness | 0.80 (0.88 for skewed high / low, B-056); sd ratio < 1 = sharper |
| `slope of the label on mu / sd(mu) over sd(label)` | calibration of the mean | 1.0 |
| `decode: encode(decode(mu)) error / price error vs label error (median ratio)` | round trip of the codec and the price error against the label error | 0 and ~1 (`< 1e-6`, 0.8..1.25) |
| `direction AUC of the close class head (day-block CI)` | AUC with a day-block bootstrap interval | 0.5 |
| `<family>: net return per trade % (CI)` | battery profit rows (`profit_sections`), cost per side from the run settings, day-block CI | best baseline row of the family |
| `GOAL: any use of the outputs profitable after cost (0.08% per side)` | any profit row with CI above 0 AND above its baselines | no |
| `selection metric vs the goal` | the metric that picks the epoch | info |

## How to read it

| Observation | Conclusion | What to change |
|---|---|---|
| rank IC below the simple model by max(0.04, 20%): FAIL; by 0.02: WARN (default `high` 0.02047 vs 0.3249) | the model adds nothing over a volatility baseline | cards A and B say where it loses |
| both ICs < 0.02: info | neither reads this label | card C `data/information` |
| MAE skill < 0 FAIL, < 0.005 WARN (default `high` -0.4298; fixed `close` 6.77e-04) | point forecast no better than a constant | usually the same cause as the card A/B findings; `close` has almost no signal on E1 |
| NLL gain < 0 FAIL, < 0.01 WARN (default `high` -0.4083; fixed `high` 0.2446) | uncertainty outputs carry no information, or are miscalibrated | `nig_scale: "train_std"`, head beta floor, card C units |
| coverage outside 0.72..0.92 WARN, < 0.6 or > 0.97 FAIL (default `high` 0.920 / 2.12) | intervals too narrow (< 0.8) or too wide | head scale / beta floor |
| slope < 0.75 or > 1.33 WARN, < 0.5 or > 2 FAIL (default `high` 0.06; fixed `low` 1.73) | slope > 1: mu shrunk to its average; < 1: over-dispersed | card B mode gap, head scale |
| decode FAIL | a decoded price does not match the label-space error | wrong anchor or codec for the head: `data_io/target_codec.py`, card C codec row |
| direction AUC CI lower bound <= 0.5 FAIL | no direction skill (fixed model: 0.5299 [0.5051, 0.5541], pass) | none from card D; the high / low heads are price levels, never a direction (owner rule) |
| profit row FAIL (fixed: `profit_close` -0.1463 [-0.3462, +0.0631] vs best baseline `shuffled direction` -0.1191) | CI contains 0 or a trivial rule does as well | the stage named by cards C, B, A |
| GOAL `no` FAIL | NOT ready for trading on this split | the diagnosis line says which stage to work on first |
| selection metric `val_raw_loss` (info) | the epoch is chosen by a likelihood loss; nothing in the fit measures profit or rank skill | card B `picked epoch` shows the regret against MAE; train `selection_metric` |

## How we know the instrument works

- Trivial baselines: constant predictor (MAE), train-fitted Gaussian (NLL), ridge / boosting (IC), `always up` / `shuffled direction` / market-at-last-close rows inside each profit family. A model must beat them; a model that adds nothing scores about 0 skill and a negative profit row after cost.
- The decode round trip must be at machine precision (fixed: 1.2e-16) and the price / label error ratio ~1 (fixed: 1.003).
- tests/doctor/test_metric_checks.py: trivial predictors (a constant) must show no skill, and `profit_rows` must refuse a model that does not beat its baseline.
- tests/doctor/test_readiness.py: the GOAL row decides the last diagnosis line.

## Limits

- Profit is the battery's own definition (cost per side, day-block CI); a strategy counts only when its CI is above 0 AND it beats every baseline row of its section. A different execution (limit fills, sizing) is another definition.
- Scored on a few thousand validation rows in quick mode (small E1: 1328): intervals are wide, rows near zero are noise.
- The profit and selection rows judge the model that was trained for a few epochs unless `--run-dir --trained` is used. Needs the full real artifact and a finished run before any trading statement.
- The row label `profit_close: profit_close:` repeats its family name (cosmetic, from `profit_rows`).
