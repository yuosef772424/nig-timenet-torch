# Data-artifact contract (BUILD_PLAN.md P01)

What this project consumes from `crypto-signal-prediction`'s data pipeline, recorded from the real code, not
designed. Source: `yuosef772424/crypto-signal-prediction` at `aeafc564f25c78f252936affbd374f68aa49b8d9`, read only.

## How the fixture was produced

Inside a read-only checkout at that commit (`./upstream`, Python 3.11.17 venv with `requirements-ci.txt`
installed, working directory outside the checkout so nothing was written there), the P01 command was run:
`_nbload.load_pipeline()`, `tests.run_fixture.build_dataset(ns, two_tf=False, coins=6, days=70)` (defaults,
unmodified), `data.save_dataset_dir(ds, "tiny_dataset.dataset")`, then
`ns["split_data"](ds, config=ns["CONFIG"])` and a keys/shapes/dtypes dump of `train`/`val`/`test`.
One deviation: the plan's `shapes()` helper only recurses one level and raises on `test` (which is
`{asset: split}`, two levels deep), so a recursive version with the same output format was used
(`docs/friction_log.csv`). Outputs, committed as-is:

- `tests/fixtures/tiny_dataset.dataset/` — `npy_dir` format (`data/storage.py:61-85`): one `<key>.npy` per numpy
  array key, `meta.pkl` (pickle, highest protocol) for every other key, `manifest.json` written last as the
  completion marker (`{"format": "npy_dir", "version": 1, "id": uuid, "created": iso time, "key_order": [...],
  "arrays": {key: {"shape", "dtype"}}}`; `id`/`created` change on every run, everything else is deterministic).
  The loaded dict = arrays + meta, in `key_order`.
- `tests/fixtures/tiny_split.json` — keys, shapes, dtypes of `train`, `val`, `test` (no data).

Fixture size: 1227 windows, 6 assets (`C0USDT`..`C5USDT`), split 962 / 120 / 121 (test is per asset: 19, 20, 20, 21, 22, 19; the 24 remaining windows fall in the purge gaps between splits).

## Contract (the literal P06 and P25-P27 implement against)

`N` = number of windows (rows); all arrays share axis 0. `test` maps each asset name (as in `asset_bounds`) to a
split dict with the same keys as `train`. `"dtype": "dict"` entries hold their children under `"items"`.

```python
CONTRACT = {'dataset': {'base_params': {'dtype': 'float32',
                             'shape_meaning': '(N, 2): per window [center, scale] of the close window (median/IQR '
                                              "for scaler 'robust'); feature normalization only, never a target "
                                              'reference (data/windows.py:190)'},
             'last_candles': {'dtype': 'float64',
                              'shape_meaning': '(N, 7): columns = core.schema.LAST_COLUMNS (last_high, last_low, '
                                               'last_close, timestamp[ns], future_close, future_low_min, '
                                               'future_high_max); float64 because of the ns timestamp'},
             'mode': {'dtype': 'str', 'shape_meaning': "(): label mode name, fixture: 'direction_robust'"},
             'scaler_type': {'dtype': 'str', 'shape_meaning': "(): window scaler, fixture: 'robust'"},
             'feature_order': {'dtype': 'list[str]',
                               'shape_meaning': '(F=18,): feature names, order of the last axis of X_{tf}'},
             'timeframes': {'dtype': 'list[str]', 'shape_meaning': "(n_tf,): timeframes present; fixture: ['1h']"},
             'targets': {'dtype': 'list[str]',
                         'shape_meaning': "(n_heads,): '{t}_class', '{t}_reg' for t in price_targets"},
             'price_targets': {'dtype': 'list[str]', 'shape_meaning': "(n_t,): fixture: ['high', 'low', 'close']"},
             'asset_bounds': {'dtype': 'list[dict]',
                              'shape_meaning': "(n_assets,): {'name': str, 'start': int, 'end': int} row ranges per "
                                               'asset, contiguous, in order'},
             'base_tf': {'dtype': 'str',
                         'shape_meaning': "(): base timeframe (targets, last_candles, grid); fixture '1h'"},
             'window_sizes': {'dtype': 'dict[str,int]',
                              'shape_meaning': "(): {tf: window length}; fixture {'1D': 32, '1h': 16}"},
             'stride': {'dtype': 'int', 'shape_meaning': '(): window stride in base-tf bars; fixture 8'},
             'forecast_horizon': {'dtype': 'int', 'shape_meaning': '(): horizon in base-tf bars; fixture 1'},
             'reg_target_mode': {'dtype': 'str', 'shape_meaning': "(): fixture 'return'"},
             'reg_target_scale': {'dtype': 'float', 'shape_meaning': '(): y_{t}_reg = return x this; fixture 100.0'},
             'split_dates': {'dtype': 'dict[str,str]',
                             'shape_meaning': "(): {'train_end': date, 'val_end': date}; set by "
                                              'tests/run_fixture.py'},
             'skipped_assets': {'dtype': 'dict', 'shape_meaning': '(): assets dropped by the pipeline; fixture {}'},
             'X_1h': {'dtype': 'float32',
                      'shape_meaning': "(N, T=16, F=18): normalized input windows of timeframe '1h', features in "
                                       'feature_order; one X_{tf} key per entry of timeframes'},
             'y_high_class': {'dtype': 'float32', 'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
             'y_high_reg': {'dtype': 'float32',
                            'shape_meaning': '(N,): regression target = return x reg_target_scale (reg_target_mode '
                                             "'return')"},
             'y_low_class': {'dtype': 'float32', 'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
             'y_low_reg': {'dtype': 'float32',
                           'shape_meaning': '(N,): regression target = return x reg_target_scale (reg_target_mode '
                                            "'return')"},
             'y_close_class': {'dtype': 'float32',
                               'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
             'y_close_reg': {'dtype': 'float32',
                             'shape_meaning': '(N,): regression target = return x reg_target_scale (reg_target_mode '
                                              "'return')"}},
 'train': {'base_params': {'dtype': 'float32',
                           'shape_meaning': '(N, 2): per window [center, scale] of the close window (median/IQR for '
                                            "scaler 'robust'); feature normalization only, never a target reference "
                                            '(data/windows.py:190)'},
           'last_candles': {'dtype': 'float64',
                            'shape_meaning': '(N, 7): columns = core.schema.LAST_COLUMNS (last_high, last_low, '
                                             'last_close, timestamp[ns], future_close, future_low_min, '
                                             'future_high_max); float64 because of the ns timestamp'},
           'y': {'dtype': 'dict',
                 'shape_meaning': '(): {y_{t}_class, y_{t}_reg} -> (N,) float32 arrays, keys = build_target_configs '
                                  'true_key',
                 'items': {'y_high_class': {'dtype': 'float32',
                                            'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
                           'y_high_reg': {'dtype': 'float32',
                                          'shape_meaning': '(N,): regression target = return x reg_target_scale '
                                                           "(reg_target_mode 'return')"},
                           'y_low_class': {'dtype': 'float32',
                                           'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
                           'y_low_reg': {'dtype': 'float32',
                                         'shape_meaning': '(N,): regression target = return x reg_target_scale '
                                                          "(reg_target_mode 'return')"},
                           'y_close_class': {'dtype': 'float32',
                                             'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
                           'y_close_reg': {'dtype': 'float32',
                                           'shape_meaning': '(N,): regression target = return x reg_target_scale '
                                                            "(reg_target_mode 'return')"}}},
           'X_1h': {'dtype': 'float32',
                    'shape_meaning': "(N, T=16, F=18): normalized input windows of timeframe '1h', features in "
                                     'feature_order; one X_{tf} key per entry of timeframes'}},
 'val': {'base_params': {'dtype': 'float32',
                         'shape_meaning': '(N, 2): per window [center, scale] of the close window (median/IQR for '
                                          "scaler 'robust'); feature normalization only, never a target reference "
                                          '(data/windows.py:190)'},
         'last_candles': {'dtype': 'float64',
                          'shape_meaning': '(N, 7): columns = core.schema.LAST_COLUMNS (last_high, last_low, '
                                           'last_close, timestamp[ns], future_close, future_low_min, '
                                           'future_high_max); float64 because of the ns timestamp'},
         'y': {'dtype': 'dict',
               'shape_meaning': '(): {y_{t}_class, y_{t}_reg} -> (N,) float32 arrays, keys = build_target_configs '
                                'true_key',
               'items': {'y_high_class': {'dtype': 'float32',
                                          'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
                         'y_high_reg': {'dtype': 'float32',
                                        'shape_meaning': '(N,): regression target = return x reg_target_scale '
                                                         "(reg_target_mode 'return')"},
                         'y_low_class': {'dtype': 'float32',
                                         'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
                         'y_low_reg': {'dtype': 'float32',
                                       'shape_meaning': '(N,): regression target = return x reg_target_scale '
                                                        "(reg_target_mode 'return')"},
                         'y_close_class': {'dtype': 'float32',
                                           'shape_meaning': '(N,): binary direction label, values {0.0, 1.0}'},
                         'y_close_reg': {'dtype': 'float32',
                                         'shape_meaning': '(N,): regression target = return x reg_target_scale '
                                                          "(reg_target_mode 'return')"}}},
         'X_1h': {'dtype': 'float32',
                  'shape_meaning': "(N, T=16, F=18): normalized input windows of timeframe '1h', features in "
                                   'feature_order; one X_{tf} key per entry of timeframes'}},
 'test': {'<asset>': {'dtype': 'dict',
                      'shape_meaning': '(): one split dict per asset name (asset_bounds names), same keys as train',
                      'items': {'base_params': {'dtype': 'float32',
                                                'shape_meaning': '(N, 2): per window [center, scale] of the close '
                                                                 "window (median/IQR for scaler 'robust'); feature "
                                                                 'normalization only, never a target reference '
                                                                 '(data/windows.py:190)'},
                                'last_candles': {'dtype': 'float64',
                                                 'shape_meaning': '(N, 7): columns = core.schema.LAST_COLUMNS '
                                                                  '(last_high, last_low, last_close, timestamp[ns], '
                                                                  'future_close, future_low_min, future_high_max); '
                                                                  'float64 because of the ns timestamp'},
                                'y': {'dtype': 'dict',
                                      'shape_meaning': '(): {y_{t}_class, y_{t}_reg} -> (N,) float32 arrays, keys = '
                                                       'build_target_configs true_key',
                                      'items': {'y_high_class': {'dtype': 'float32',
                                                                 'shape_meaning': '(N,): binary direction label, '
                                                                                  'values {0.0, 1.0}'},
                                                'y_high_reg': {'dtype': 'float32',
                                                               'shape_meaning': '(N,): regression target = return x '
                                                                                'reg_target_scale (reg_target_mode '
                                                                                "'return')"},
                                                'y_low_class': {'dtype': 'float32',
                                                                'shape_meaning': '(N,): binary direction label, '
                                                                                 'values {0.0, 1.0}'},
                                                'y_low_reg': {'dtype': 'float32',
                                                              'shape_meaning': '(N,): regression target = return x '
                                                                               'reg_target_scale (reg_target_mode '
                                                                               "'return')"},
                                                'y_close_class': {'dtype': 'float32',
                                                                  'shape_meaning': '(N,): binary direction label, '
                                                                                   'values {0.0, 1.0}'},
                                                'y_close_reg': {'dtype': 'float32',
                                                                'shape_meaning': '(N,): regression target = return x '
                                                                                 'reg_target_scale (reg_target_mode '
                                                                                 "'return')"}}},
                                'X_1h': {'dtype': 'float32',
                                         'shape_meaning': '(N, T=16, F=18): normalized input windows of timeframe '
                                                          "'1h', features in feature_order; one X_{tf} key per entry "
                                                          'of timeframes'}}}}}
```

## Notes (reality vs. what BUILD_PLAN.md P01 lists)

- `workflow/run.py:109-165` also reads `price_norm_mode`, `price_pct_clip`, `holdout_start`, `higher_tf_mode`
  from the dataset, always via `dataset.get(key, default)` (defaults: `"window_scale"`, `CONFIG["price_pct_clip"]`
  (100.0), absent = no sealed holdout, absent = legacy higher-tf alignment). The fixture does not carry them, so
  they are OPTIONAL keys, not in the literal above (whose key set must equal the fixture's exactly).
- `y_{t}_*` arrays are 1-D `(N,)`, not `(N, 1)`; the model/oracle side uses `(B, 1)`, so a reshape happens at feed time.
- `y_{t}_class` is stored as `{0.0, 1.0}` in this fixture.
- `split_data` alone does not stamp `reg_target_scale` (or `target_mode`) into the splits; `workflow/run.py`'s
  `make_splits` does that afterwards (`_m["reg_target_scale"] = info.reg_target_scale`). The literal records what
  `split_data` returns.
- `ns["split_data"](ds, config=ns["CONFIG"])` printed train <= 2025-02-25 / val <= 2025-03-04 although the dataset
  carries `split_dates = {train_end: 2025-02-08, val_end: 2025-02-22}`: the split used `CONFIG`'s own dates, not the
  dataset's. Recorded as produced by the plan's command; the split logic is ported later (P25), not here.

## Export manifest (ROADMAP T2, rule D1)

The real dataset reaches this project only as an artifact written by `crypto-signal-prediction`
`tools/export_torch_artifact.py` (TF PR [#65](https://github.com/yuosef772424/crypto-signal-prediction/pull/65)). It is the
`npy_dir` layout above, filtered, with an `export` block added to `manifest.json` (the npy_dir keys are unchanged, so
`data_io/npy_dir.py` still reads it). `data_io/artifact.py::load_artifact` refuses a directory without the block, or
with any of these inconsistent:

| `export.` field | Meaning | Checked by the loader |
|---|---|---|
| `schema` | `"torch_artifact/1"` | equal |
| `source_commit` | git commit of the TF repo that ran the export (the tool refuses a dirty tree) | 40 lowercase hex |
| `coins` | coins in the artifact, in row order | non-empty, equals the dataset's `asset_bounds` names, none on the sealed list |
| `sealed_excluded` | the WHOLE sealed list of the TF lock (`tools/holdout_guard.py`, `docs/research/plan/lock/holdout.json`) | exactly `EXPECTED_SEALED_COUNT` (183) names, pinned in `data_io/artifact.py`, not read from the manifest; no coin of `coins` matches it (case and trailing `USDT` ignored) |
| `lock_sha256` | sha256 of the TF `holdout.json` the export ran against | equals `EXPECTED_LOCK_SHA256`, pinned in `data_io/artifact.py` (= the TF `holdout.sha256`); required |
| `time_seal` | `2026-10-01T00:00:00+00:00` | equals the constant in `artifact.py` |
| `max_bar_time` | latest LABEL bar open time (last window bar + `forecast_horizon` x base-tf bar) | `< time_seal`; the loader recomputes it from `last_candles` and refuses data that reaches later than the manifest says |
| `data_sha256` | sha256 of the JSON of `files` (sorted keys); this is the `data_fingerprint` of `RunIdentity` | 64 lowercase hex |
| `base_tf`, `timeframes` | OPTIONAL (D-015): the bar interval and the timeframes the export holds; declared together or not at all | when present, equal to the dataset's `base_tf` / `timeframes` (else refused); when absent the artifact loads and `ArtifactManifest.declares_timeframes` is False. A job passes `LoaderSettings.expect_base_tf` (baseline key `base_tf`) to refuse another interval |
| `files` | sha256 of every file in the directory except `manifest.json` | format; re-hashed only by `verify_hashes` / `--verify-hashes` |
| `sealed_found`, `max_window_time`, `rows`, `rows_source`, `dropped_*`, `source_*` (incl. `source_format`: `"pkl.gz"` or `"npy_dir"`, TF PR 66) | provenance for the reader | not checked |

`build_run_identity(fit_settings, model_settings, trainer.config, trainer.optimizer, artifact.manifest.data_fingerprint,
code_sha=...)` (`trainer/identity.py`) is the one way a real run gets its `RunIdentity`. Its `settings_hash` covers the
fit settings (without `epochs`), the model settings, the WHOLE trainer config (targets with loss weights, loss with
schedules, constraints) and the optimizer (class, defaults, every param group); all five are required arguments. A
changed loss weight or learning rate therefore makes `Trainer.fit` refuse to resume (`ResumeMismatchError`).
`Trainer.fit` raises `SeedMismatchError` when `identity.seed != settings.seed`.

A real run builds its identity in ONE place, after the `Trainer` exists (`colab/identity.py::identity_for_run`): the
Trainer adds the Kendall log-variances to the optimizer as an extra param group, so an identity taken before it describes
another optimizer. `src/` is scanned for any other caller of `build_run_identity`.
A `torch` upgrade can change `optimizer.defaults` (new keys, new default values), which changes `settings_hash`: a checkpoint
written under the old torch is then refused with `ResumeMismatchError` naming `settings_hash`. That is intended (the hash
cannot tell a harmless default from a real change). Colab upgrades its torch from time to time, so a long run that must
survive a runtime restart should be resumed on the same torch version (`env.json` of the bundle records it).

The two lock pins (`EXPECTED_SEALED_COUNT`, `EXPECTED_LOCK_SHA256`) are values of the TF lock at crypto-signal-prediction
`cea9cd5` (`holdout.json`: 183 sealed coins, sha256 `70322ba6...`). They bind the manifest to that lock; they do not
prove the listed names are the lock's names (see `docs/suggestions.csv`, T2 review).

### Real data: what the first Colab run showed, and how to repeat it

Owner's run on Colab, 2026-11-03 (TF `c0634ba`, torch `b34579c`), from `preprocessing_output_latest.pkl.gz` (29 Oct 2026),
converted by hand with `data.load_data_from_drive(fmt="pkl.gz")` + `data.save_dataset_dir`, then exported and checked with
`python -m nig_timenet_torch.data_io.artifact <dir> --verify-hashes` (passed):

| What | Value |
|---|---|
| Artifact | `MyDrive/crypto_model/torch_artifacts/preprocessing_output_1` |
| Windows kept | 2,520,913 of 3,460,789 |
| Coins | 535 |
| `max_bar_time` | 2026-09-26T20:00Z |
| Sealed coins present in the TF data | 183 of 183 (all removed by the export) |
| `data_sha256` | `c572b1b36960...` |

Two consequences. (1) All 183 sealed coins were inside the TF main data, so earlier TF results (profile-004 included) are
not a clean holdout test (`docs/bottlenecks.csv` B-022). (2) STILL UNVERIFIED: that `last_candles` column 3 is the bar OPEN
time; `max_bar_time` and the time-seal check rest on that assumption (B-020).

To repeat it, open `notebooks/colab_nig_timenet_torch.ipynb` and run section 2: it exports with
`tools/export_torch_artifact.py --dataset ...pkl.gz` (TF PR 66; the tool reads the old format the way `data/storage.py`
does, no manual conversion), verifies with the loader above, and skips the export when `<artifact>.source.json` says the
artifact is sound and was made from the same source file, export tool and lock (`colab/prepare.py`). The loader needs only
numpy and pandas (no torch install, so Colab's CUDA torch is left alone).
