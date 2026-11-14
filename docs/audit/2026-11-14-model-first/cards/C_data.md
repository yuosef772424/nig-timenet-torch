# Card C — Data

Question: are the labels, windows, splits and scales sound, and do the inputs hold any signal a simple model can read?

## Run

```
python -m nig_timenet_torch.doctor e1 --out doctor_out --cards C
python -m nig_timenet_torch.doctor /path/to/artifact --out doctor_out --cards C --max-rows 60000
python -m nig_timenet_torch.data_io.vision_slice OUT_DIR --raw RAW_DIR      # real data without Drive: public Binance 4h klines, sealed coins refused
```

- Reads the artifact (mmap) and the trained context; card C itself takes about 1.4 s on small E1 (header `'C': 1.4`). The doctor still trains a short model first (needed for the unit and batch-independence checks).
- CPU is enough. Card C runs first because card D reads its ceiling.

## What it measures

| check name (as printed) | what is computed | baseline |
|---|---|---|
| `shared bars of overlapping windows identical (max abs diff)` | pairs of rows of one coin, k bars apart (horizon <= k <= window): the bars both windows hold must be equal | <= 2e-5 (`ALIGN_TOL`) on N pairs |
| `<t> label == the next bars read from a later window (max err)` | the label of row i rebuilt from the bars AFTER i read in a later window (close = sum of next h close moves; high / low = max / min of running close + move); clipped labels skipped | <= 2e-5 |
| `an evaluated row is independent of its batch mates (max output diff)` | eval-mode outputs of 64 rows alone vs inside a bigger batch | <= 1e-5 |
| `codec decodes the stored labels to the future prices` | `run_common.check_decode` | ok |
| `close class label == sign of the close regression label (mismatch)` | disagreement of the two labels of the close head | < 1e-4 |
| `<t>: label sd in the head's units` | train label sd / head scale | ~1; warn < 0.3 or > 5, fail < 0.1 or > 20 |
| `median per-window input variance / InstanceNorm eps` | window variance after the input scaler, over eps | warn < 30, fail < 3 |
| `<a> -> <b>: bars between the last label and the first window` | split purge | >= 0 |
| `rows train / val / test (coins; coins unseen in train)` | row counts, coin counts, date span (JSON) | info |
| `val vs train` / `test vs train: worst sd ratio over inputs and labels` | sd ratio of every input channel and label against train | (0.67, 1.5) |
| `<t>: simple-model ceiling, rank IC by timestamp val / test` | ridge and gradient boosting (scikit-learn if installed) on HAR-style features (`simple_features`), fitted on up to 60000 train rows | WARN if the best val IC by timestamp < 0.02 |
| `parameters per effectively independent training label` | parameters / (train rows x (1 - label overlap) x effective coins / coins per timestamp) | < 1 |
| `window overlap with the next row / label overlap / exact duplicates` | shares shared with the next row; share of duplicate windows | info |
| `non-finite inputs or labels (train)` | counts | 0 |
| `flat bars / zero close moves / constant windows / clipped labels` | shares | info |
| `distinct coins per shuffled batch (mean of 20)` | batch composition | info |

## How to read it

| Observation | Conclusion | What to change |
|---|---|---|
| `shared bars` FAIL | a window is not a pure slice of the coin's bars: per-window state, misaligned rows or future bars in a window | data/ window builder (`data_io`, artifact writer); stop and fix before anything else |
| `label == the next bars` FAIL | horizon, anchor or row alignment of the label is off | label construction in data/ (anchor, `forecast_horizon`) |
| `independent of its batch mates` FAIL | eval outputs depend on other rows (batch-mode normalisation or a cross-row path) | the layer that uses batch statistics in eval; `bn_warmup`, `level_norm: "batch"` |
| `codec decodes` FAIL | stored labels do not decode to the future prices | `data_io/target_codec.py`, `reg_target_scale`, anchors |
| `close class label == sign` FAIL | two labels of one head built from different bars | label writer |
| label sd in head units FAIL/WARN (default model on E1: 0.02033, 0.02186, 0.03142) | NIG head constants (beta floor, softplus offsets) assume sd ~1 | model override `nig_scale: "train_std"` (or a codec scale); fixed model: 0.8126 / 0.8735 / 1.256 |
| `InstanceNorm eps` FAIL/WARN (default 1.903; fixed 6646) | eps is as big as the window variance, normalisation is damped | model override `input_scaler: "train_std"` |
| `bars between` < 0 FAIL | a later split starts before an earlier label ends: leakage | split plan `data_io/split_plan.py`: purge / embargo |
| `sd ratio` WARN outside (0.67, 1.5) | the split sits in another volatility regime; errors mix skill with regime | choose another split, or report per regime |
| ceiling WARN (< 0.02) | the inputs hold no readable signal for this label | no model or trainer change fixes it: add inputs, or the target is unpredictable (E1 `close`: 0.043 / -0.029 is info, since E1 `close` has almost no signal) |
| `parameters per effectively independent label` WARN (8.003 on small E1) | more parameters than independent labels, memorisation risk | smaller model, `dropout`, `weight_decay`, more history |
| `non-finite` FAIL | NaN / inf in the artifact | data writer |

The diagnosis treats FAIL in `data/alignment`, `data/split`, `data/coverage` as "DATA is not sound" (first line, fix first) and FAIL in `data/units` as "UNITS" (second).

## How we know the instrument works

- Labels rebuilt from later windows: the alignment check recomputes each label from bars the label should be made of; it passes at 7.45e-09 / 1.02e-08 on E1 (limit 2e-5); a label built from the wrong bars cannot pass it.
- E1 has known conditional means, so the ceiling can be compared with the oracle: E1 `high` ceiling 0.325 / 0.292, `low` 0.325 / 0.318 (val / test).
- tests/doctor/test_data_checks.py: planted faults (labels built from the wrong bars, overlapping splits, scale errors) must be named by the matching row.
- tests/doctor/test_common.py: `simple_features` and the purged folds the ceiling uses.

## Limits

- Alignment needs raw log-move inputs (channels high, low, close; one timeframe; `price_norm_state` "unfitted"); other artifacts print "not applicable" (info).
- The ceiling is a simple model (ridge, gradient boosting on HAR-style features), not a proof that nothing better exists; a WARN does not rule out signal in other features. The test split is read only for composition, shift and the ceiling's own score.
- Effective independent labels is an estimate (same-timestamp correlation of close labels), not exact.
- Large samples are strided (60000 train rows, 100000 for counts); rare defects can be missed. A real statement about real data needs the full real artifact, not the strided slice.
