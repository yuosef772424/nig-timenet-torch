# Settings parity audit 2026-11-23: every tunable setting of the TF project against this one, and what the notebooks show

Owner request (2026-11-23): "update the pipeline so that ALL the settings that can be tuned are shown, preferably the same
settings as the old project or better". This note is the audit behind the change: `workflow/settings_listing.py` (the
generated list of every setting), the listing cell of both notebooks and `tests/workflow/test_settings_listing.py`.
It is a record, not a result (no model was trained): rule C7 does not apply.

## Method

* Old side, read in `/home/user/crypto-signal-prediction`: `workflow/settings.py` (`RunSettings`, 7 sections), the data
  pipeline's `PIPELINE_DEFAULT_CONFIG` in `data/defaults.py` (101 keys, reached through `ProjectSettings.config_overrides`),
  `docs/research/main_lab_code_layout.md` (old variable -> new field table).
* Torch side, read as code: the dataclasses `RunSettings` (+ `LoaderSettings`, `SplitSettings`, `TrainSettings`,
  `EvalSettings`, `BatterySettings`, `EarlyStopSettings`, `StageSettings`, `DayWeightSettings`, `ModelSettings` and the
  model builder's `BuilderOptions`), `DataSettings` (the native data pipeline), `RollingSettings`, the Colab layer
  (`colab/job.py`, `jobs/*.py` configs) and the notebooks' own variables. A torch path is only named below when a field of
  that name exists (`tests/workflow/test_settings_listing.py` pins the listing to the dataclasses; the paths in the tables
  were checked against the listing when this note was written).
* Classes: **a** same setting in torch; **b** renamed or merged (the torch path is given); **c** not applicable (why);
  **d** MISSING in torch though meaningful there; **e** torch has it, the notebook did not show it.

## Summary

| | a | b | c | d | e | total |
|---|---|---|---|---|---|---|
| old `RunSettings` leaves (dict settings expanded: `lambda_reg` x4, `lambda_calib` x4, `anti_memorization_trainer` x5) | 5 | 35 | 9 | 17 | 1 | 67 |
| old data `CONFIG` keys (table B) | 23 | 36 | 30 | 12 | 0 | 101 |

Torch side (the generated listing, `python -m nig_timenet_torch.workflow.settings_listing`): **253 settings**: `RunSettings`
152 (73 own fields, 55 model options `model.overrides.*` of the builder, 17 `train.staged.*`, 7 `train.day_weights.*`),
`DataSettings` 91, `RollingSettings` 10; 35 of them have a set of valid values read from a registry.

What was NOT shown before this change (class e, counted from the notebook at commit 161f7c0): of the 152 `RunSettings`
settings 63 were not in `main.ipynb` (37 of the 55 model options, 17 `train.staged.*`, 7 `train.day_weights.*`, and
`run_dir` / `data.name`, which the notebook derives), none of the 91 `DataSettings` was listed anywhere in a notebook
(`data_pipeline.ipynb` had `CHANGES = {}` and a pointer to `data.cli show`), and 4 of the 10 `RollingSettings` fields
(`mode`, `retrain_every_k`, `fold_ids`, `val_share`) were not named in the `ROLLING` example. Valid values were typed by
hand in some of the comments and shown nowhere for the rest.

Biggest gaps against the old project (class d, details in "Proposals"): the shape of the NIG regulariser / calibration
schedules (start, warm-up epochs, kind: hard-coded), the confidence calibration (a component exists, no setting, no
step), the market-neutral report's legs and universe (the battery's top shares are function defaults), the panel's
`k_eval` / `min_coins` / `max_days`, the run-time target mode (`target.target_mode`: agent 2's step), and the old data
sources funding rate, open interest and intraday 15m.

## Table A: the old `RunSettings` against this project (67 leaves)

| # | old setting | old default | meaning | class | torch (or why not) |
|---|---|---|---|---|---|
| 1 | `project.config_overrides` | `{'project_name': 'crypto_model'}` | overrides of the data pipeline's CONFIG (table B) | b | `DataSettings` (the whole tree, set in `data_pipeline.ipynb` CHANGES); `project_name` itself: c, the run folder is `run_dir` |
| 2 | `data.filename_base` | `'preprocessing_output'` | dataset base name | b | `data.name` |
| 3 | `data.format` | `'auto'` | auto | pkl.gz | npy_dir | a | `data.format` |
| 4 | `data.mmap` | `True` | memory-map the arrays | a | `data.mmap` |
| 5 | `data.local_dir` | `None` | copy a Drive folder to local disk first (None = yes, False = read in place, or a path) | b | `data.local_dir` ('auto' = copy under /content/drive, '' = never, or a path) |
| 6 | `data.path` | `None` | explicit dataset path | b | `data.directory` (a `data.name` with a directory part wins) |
| 7 | `data.model_tfs` | `None` | timeframes fed to the model, first = base | b | `data.input_timeframes` |
| 8 | `data.split_dates` | `None` | {train_end, val_end} | b | `split.train_end` + `split.val_end` |
| 9 | `target.target_mode` | `None` | target mode on the same data (return, scaled, volnorm, ... +relative) | d | build time only (`DataSettings` `targets.reg_target_mode`, per-coin modes); the run-time switch (`data.targets.retarget` on the loaded splits) is not a RunSettings field: agent 2's step adds it |
| 10 | `target.entry_close_reg` | `'abs_return'` | close head of entry_range | b | `targets.entry_close_reg` (build time) |
| 11 | `target.group_freq` | `None` | width of the cross-section group of the +relative modes | d | a keyword of `data.targets.retarget` only (the artifact grid is aligned, `windows.align_to_grid`); with the target-mode step |
| 12 | `target.suspended_targets` | `()` | price targets the model does not train | b | `model.overrides.price_targets` (+ `head_types`) and `targets.targets` at build time; direction of high / low is suspended in the evaluation code, not by a setting |
| 13 | `model.anti_memorization` | `True` | the small anti-memorization model preset | b | no boolean: the preset IS the default `model.overrides` (`ANTI_MEMORIZATION_CONFIG` + D-023 / D-026 switches); `{}` in CHANGES gives the plain architecture |
| 14 | `model.class_only` | `False` | classification heads only | b | `model.overrides.head_types` |
| 15 | `model.config_overrides` | `{}` | architecture overrides merged over the preset | b | `model.overrides` (and `model.architecture`) |
| 16 | `train.run_main_training` | `True` | False = build but do not train | b | notebook `LOAD_WEIGHTS` ('best' / 'last' read a finished run, no training) |
| 17 | `train.run_dir` | `'/content/drive/.../crypto_model_v1'` | training folder | b | `run_dir` (the notebook derives it from `RUN_NAME` + `WORK_ON`) |
| 18 | `train.epochs` | `60` | epochs | a | `train.epochs` (default 5) |
| 19 | `train.batch_size` | `64` | batch size | a | `train.batch_size` (default 256) |
| 20 | `train.train_mode` | `'auto'` | TF trainer mode | c | Keras trainer switch; the torch loop has `train.batch_by` / `train.batch_order` / `train.staged` |
| 21 | `train.seed` | `None (= 42)` | run seed | a | `train.seed` (default 0) |
| 22 | `train.early_stopping_patience` | `None` | early-stopping patience | b | `train.early_stop.patience` |
| 23 | `train.fit_verbose` | `1` | Keras progress bar | c | Keras option; torch prints one line per epoch (`workflow/flat_run.py::epoch_line`) |
| 24 | `train.use_uncertainty_weighting` | `True` | Kendall weighting of the class heads | b | `train.uncertainty_weighting` |
| 25 | `train.lambda_reg.end` | `0.05` | NIG evidence regulariser, end value | b | `train.lambda_reg_end` |
| 26 | `train.lambda_reg.start` | `0.0` | ... start value | d | hard-coded in `workflow/flat_run.py::LAMBDA_SCHEDULES` |
| 27 | `train.lambda_reg.warmup_epochs` | `5` | ... warm-up epochs | d | hard-coded in `LAMBDA_SCHEDULES` |
| 28 | `train.lambda_reg.schedule` | `'linear'` | ... schedule kind | d | hard-coded in `LAMBDA_SCHEDULES` |
| 29 | `train.lambda_calib.end` | `0.1` | calibration term, end value | b | `train.lambda_calib_end` |
| 30 | `train.lambda_calib.start` | `0.0` | ... start value | d | hard-coded in `LAMBDA_SCHEDULES` |
| 31 | `train.lambda_calib.warmup_epochs` | `5` | ... warm-up epochs | d | hard-coded in `LAMBDA_SCHEDULES` |
| 32 | `train.lambda_calib.schedule` | `'cosine'` | ... schedule kind | d | hard-coded in `LAMBDA_SCHEDULES` |
| 33 | `train.early_stopping_monitor` | `'val_raw_loss'` | metric that picks the best epoch | b | `train.selection_metric` |
| 34 | `train.early_stopping_mode` | `'min'` | min | max | c | derived from the metric's `higher_is_better` (rule C3) |
| 35 | `train.anti_memorization_trainer.optimizer.lr_initial` | `3e-4` | learning rate of the preset | b | `train.lr` (default 1e-3: the preset's value is not the default here) |
| 36 | `train.anti_memorization_trainer.optimizer.weight_decay` | `0.05` | weight decay of the preset | b | `train.weight_decay` (default 1e-4) |
| 37 | `train.anti_memorization_trainer.optimizer.ema_warmup` | `True` | EMA of the weights with TF warm-up | b | `train.ema_decay` (None = off; the warm-up is built in) |
| 38 | `train.anti_memorization_trainer.optimizer.ema_window_epochs` | `1.0` | EMA window in epochs | d | torch takes a decay, not a window: no window setting |
| 39 | `train.anti_memorization_trainer.callbacks.early_stopping.weights_snapshot` | `'ema_weights'` | the best weights are the EMA weights | b | `train.ema_decay`: with it the EMA copy is what validation, selection and the best weights hold (`trainer/trainer.py`) |
| 40 | `train.anti_memorization_label_smoothing` | `0.1` | label smoothing of the class heads | b | `train.label_smoothing` |
| 41 | `evaluation.calibrate_confidence` | `True` | post-hoc confidence calibration (fit on val) | d | `eval/calibrator.py` exists but no step or setting calls it |
| 42 | `evaluation.calibration_method` | `'isotonic'` | isotonic | platt | d | same: no setting |
| 43 | `evaluation.eval_all_rows` | `True` | every row of test enters the metrics | c | torch always evaluates every test row (`test_step`); `split.max_rows` is the smoke cap |
| 44 | `evaluation.out_dir` | `'analysis_outputs'` | report folder | b | `run_dir` + the notebook's `out/` folder |
| 45 | `evaluation.mn_quantiles` | `(0.05, 0.1, 0.2, 0.3, 0.5)` | legs of the market-neutral report | d | the battery's top shares (0.05 / 0.10 / 0.20) are function defaults (`eval/battery.py::trade_grid`), not settings |
| 46 | `evaluation.mn_universe` | `None` | universe of that report | d | no setting |
| 47 | `evaluation.trivial_baseline` | `False` | trivial-rule AUC column | c | the battery always reports its trivial baselines (rule C1) |
| 48 | `panel.enabled` | `False` | run the cross-asset panel model | b | `split.cross_section` (+ `model.overrides.cross_input`); no separate model |
| 49 | `panel.variants` | `('A_ic', 'B_ic')` | panel arms | e | job entry `panel` (`PanelConfig.arms`, `workflow/panel_study.py`); the notebook shows `ROLLING_PANEL` only |
| 50 | `panel.seeds` | `(0,)` | seeds | b | `train.seed` / `RollingSettings.seeds` / `PanelConfig.seeds` |
| 51 | `panel.epochs` | `None` | epochs | b | `train.epochs` |
| 52 | `panel.batch_samples` | `1024` | (coin, day) rows per batch of whole days | b | `train.batch_size` with `split.cross_section` (cuts at timestamp borders) |
| 53 | `panel.max_days` | `16` | days per batch cap | d | `train.batch_by` 'day' takes a whole day; no cap on days per batch |
| 54 | `panel.baseline` | `'auto'` | baseline to compare with | c | the battery compares with its own baselines (rule C1) |
| 55 | `panel.first_touch` | `None` | bracket_first_5.pkl | c | TF bracket tool; not used here |
| 56 | `panel.subset` | `None` | quick test {last_days, coins} | b | `split.max_rows`, `split.top_coins`, `split.exclude_coins` |
| 57 | `panel.run_root` | `None` | run folder | b | `run_dir` |
| 58 | `panel.overrides` | `{}` | train / model overrides of the panel | b | `model.overrides` and any train field (the notebook's `ROLLING_PANEL`) |
| 59 | `panel.patience` | `None` | early-stopping patience | b | `train.early_stop.patience` |
| 60 | `panel.monitor` | `'val_loss'` | epoch-selection metric | b | `train.selection_metric` ('val_rank_ic_reg' for a panel) |
| 61 | `panel.group` | `'auto'` | cross-section group width | c | the artifact grid is aligned (`windows.align_to_grid`), so a group is the exact timestamp |
| 62 | `panel.allow_mixed_timestamps` | `False` | skip the mixed-timestamps refusal | c | no leak-measuring mode here |
| 63 | `panel.min_coins` | `None` | fewest coins per group | d | no setting (`split.top_coins` caps from above only) |
| 64 | `panel.max_coins` | `None` | most coins per group | b | `split.top_coins` (partial: the whole run keeps the coins with the most train rows, not a cap per group) |
| 65 | `panel.k_eval` | `None` | test with only k coins visible | d | no such evaluation |
| 66 | `panel.k_draws` | `3` | random draws of that | d | no such evaluation |
| 67 | `panel.preset` | `None` | ready-made panel setup ('1h_s8') | b | `ROLLING_PANEL` in the notebook; `workflow/panel_study.py` arms |

## Table B: the old data pipeline CONFIG (101 keys) against `DataSettings`

Basis: "a" rows have a `DataSettings` field of that meaning that the data code reads; "b" / "c" / "d" were classified by
reading the torch code (not by running both pipelines against each other: see "Not verified").

| old CONFIG key | class | torch (or why not) |
|---|---|---|
| `tf_order` | a | `windows.tf_order` |
| `base_tf` | b | the first of `windows.tf_order` |
| `window_sizes` | a | `windows.window_sizes` |
| `stride` | a | `windows.stride` |
| `align_windows_to_grid` | b | `windows.align_to_grid` |
| `window_grid_anchor` | b | `windows.grid_anchor` |
| `higher_tf_offset` | a | `windows.higher_tf_offset` |
| `higher_tf_mode` | a | `windows.higher_tf_mode` |
| `x_storage_dtype` | b | `runtime.x_storage_dtype` (None becomes '') |
| `disk_backed` | b | `runtime.spill_dir` (non-empty = disk-backed build) |
| `scratch_dir` | b | `runtime.spill_dir` |
| `scratch_cleanup_after_save` | c | no scratch folder survives a build here |
| `save_format` | c | the output is always the contract artifact folder (`output.*`) |
| `save_single_copy` | b | `output.latest` (False = no `_latest` copy) |
| `pkl_compresslevel` | c | no pickle writer |
| `cross_asset_dtype` | c | TF cross_asset package; no counterpart |
| `targets` | a | `targets.targets` |
| `forecast_horizon` | b | `targets.horizon` |
| `target_mode` | c | the TF label mode ('direction'); here the class labels are fixed and the regression mode is `targets.reg_target_mode` |
| `enabled_heads` | b | `model.overrides.head_types` |
| `target_loss_weights` | d | no per-target (high / low / close) loss weight; only `train.class_loss_weight` |
| `abstention` | c | TF abstention head; the confidence head is part of the model (`model.overrides.unc_max`) |
| `scaler_type` | a | `scaling.scaler_type` |
| `scaling_window` | c | the scaling window is the window (`windows.window_sizes`) |
| `eps` | a | `scaling.eps` |
| `split_mode` | c | only the global time split exists (`split.*`) |
| `split_dates` | b | `output.split_dates` (the artifact's own) and `split.train_end` / `split.val_end` |
| `holdout_start` | a | `output.holdout_start` |
| `open_holdout` | c | the sealed holdout is a rule (D1), not a setting |
| `train_pct` | b | `split.train_share` |
| `val_pct` | b | `split.val_share` |
| `keep_asset_test_separate` | c | the split is by time over all coins; `split.exclude_coins` drops coins from every part |
| `embargo_candles` | b | `split.embargo_hours` (hours, not candles) |
| `min_split_samples` | c | no minimum; a split without rows raises |
| `download_interval` | b | `source.interval` |
| `market_context` | b | `features.market.*` |
| `momentum_rank` | a | `features.momentum_rank.*` |
| `market_breadth` | b | `features.breadth.*` |
| `momentum_orth_natr` | b | `features.orth_natr.*` |
| `rolling_retrain` | b | `RollingSettings` (the ROLLING dict of the notebook) |
| `funding_rate` | d | no funding-rate feature in `DataSettings` |
| `open_interest` | d | no open-interest feature |
| `intraday_efficiency` | d | no intraday-15m feature |
| `intraday_vwap` | d | no intraday-15m feature |
| `intraday_volume_concentration` | d | no intraday-15m feature |
| `phase2_data` | d | no futures-metrics / intraday source |
| `split_cutoff_method` | c | the split purges by label end (D2), not a cutoff method |
| `exclude_from_features` | a | `features.exclude_from_features` |
| `feature_order` | c | derived: `data.build.feature_order(settings)` |
| `indicator_settings` | a | `features.indicator_settings` |
| `static_indicators` | a | `features.static_indicators` |
| `static_indicator_params` | a | `features.static_indicator_params` |
| `use_custom_features` | b | `features.custom.enabled` |
| `custom_settings` | b | `features.custom.*` |
| `clip_abs` | a | `scaling.clip_abs` |
| `price_norm_mode` | a | `scaling.price_norm_mode` |
| `price_pct_clip` | a | `scaling.price_pct_clip` |
| `sample_filters` | a | `windows.sample_filters` |
| `stale_bars` | d | TF research toggle for dead-coin tails; not ported (no setting) |
| `reg_target_mode` | a | `targets.reg_target_mode` |
| `reg_target_clip` | a | `targets.reg_target_clip` |
| `reg_target_scale` | a | `targets.reg_target_scale` |
| `cs_norm` | c | replaced by `model.overrides.cross_input` ('relative') and the +relative target modes |
| `asset_registry_file_id` | c | gdown id; unused in the TF notebook too |
| `preprocessed_data_file_id` | c | gdown id; unused |
| `drive_raw_dir` | b | `source.raw_dir` |
| `drive_raw_pattern` | b | `source.pattern` |
| `asset_registry_path` | b | `source.registry_path` |
| `drive_mount_point` | c | Colab detail (`colab/` code) |
| `excluded_coins` | b | `source.exclude` |
| `live_max_requests_per_minute` | c | no live module here |
| `model_tf` | b | `data.input_timeframes` / `windows.tf_order` |
| `d_model` | b | `model.overrides.d_model` |
| `num_layers` | b | `model.overrides.num_layers` |
| `num_heads` | b | `model.overrides.num_heads` |
| `num_kv_heads` | b | `model.overrides.num_kv_heads` |
| `kv_latent` | c | TF ModernTST option; not in the torch builder |
| `expansion` | c | TF ModernTST option (the torch FFN is `model.overrides.ffn_dim`) |
| `model_dropout` | b | `model.overrides.dropout` |
| `model_name` | c | TF model name; `model.architecture` here |
| `use_decomposition` | b | `model.overrides` (only True is implemented) |
| `decomp_kernels` | b | `model.overrides.kernel_sizes` |
| `moe_every` | c | no mixture of experts in the torch model |
| `n_shared` | c | no mixture of experts |
| `n_routed` | c | no mixture of experts |
| `top_k` | c | no mixture of experts |
| `learning_rate` | b | `train.lr` |
| `gradient_clipnorm` | b | `train.grad_clip_norm` |
| `epochs` | a | `train.epochs` |
| `batch_size` | a | `train.batch_size` |
| `use_class_weights` | d | no class-weight setting |
| `class_weight_smoothing` | d | no class-weight setting |
| `early_stopping_patience` | b | `train.early_stop.patience` |
| `reduce_lr_patience` | d | no reduce-on-plateau; `train.lr_schedule` is constant / cosine / cosine_restarts |
| `reduce_lr_factor` | d | no reduce-on-plateau |
| `eval_batch_size` | b | `evaluation.chunk` |
| `eval_range_frac` | c | TF report display option |
| `eval_n_display` | c | TF report display option |
| `project_name` | c | the run folder is `run_dir` |
| `workspace_dir` | c | Colab detail |
| `use_drive` | c | Colab detail (`WORK_ON`) |

## Torch-only settings (kept)

No old counterpart (35 `RunSettings` fields not named in table A / B): `code_sha`, `data.allow_unchecked`,
`data.verify_hashes`, `data.expect_base_tf` (rule D1 checks), the whole `train.early_stop` section but `patience`
(`min_delta`, `smoothing`, `window`, `ema_beta`, `restore_best_weights`), `train.device`, `train.trace_representative`,
`train.steps_per_execution`, `train.init`, `train.weight_decay_groups`, `train.beta_nll`, `train.beta_nll_normalize`,
`train.loss_balance`, `train.kendall_init_logvar`, `train.cross_section_loss`, `train.cross_section_weight`,
`train.direction_balance`, `train.time_block_bars`, `train.day_weights` (+ 7 fields), `train.aux_*` (3), `train.lr_min_ratio`,
and the evaluation battery (`evaluation.health_max_rows`, `evaluation.battery.*` x7: cost per side, variants, day metrics,
verdict trim, drop flat, account books). Named above only as the nearest counterpart of an old setting, but with no old
equivalent in what they do: `split.cross_section`, `split.top_coins`, `split.coin_channel`, `split.exclude_coins`,
`split.max_rows`, `split.embargo_hours`, `train.batch_by`, `train.batch_order`, `train.staged` (+ 17 fields),
`train.lr_schedule`, `train.grad_clip_norm`. Also torch-only: the 37 model options beyond the 18 keys `CHANGES` holds
(encoder kinds, input skip, cross-coin blocks, NIG head kinds, aux heads, HAR range, target normalisation, ...), the
`RollingSettings` section, and the `DataSettings` sections `source` (fetch, synthetic source, categories),
`output` (artifact folder, stamp, `latest`) and `runtime`. All stay.

## Colab and job layer

| layer | settings | class |
|---|---|---|
| `main.ipynb` settings cell (43 variables before this change, 44 now) | code source and token (17: `TORCH_REPO`, `TORCH_REF`, `REPO_SOURCE`, `REPO_DIR`, `TORCH_DIR`, `UPDATE_DRIVE_COPY`, `DISCARD_LOCAL_CHANGES`, `GITHUB_TOKEN_SECRET`, `TOKEN_PATH`, `SAVE_TOKEN_TO_DRIVE`, `USE_ENV_CACHE`, `ENV_CACHE_DIR`, `VENV_DIR`, `EXTRA_MODULES`, `EXPECT_DEVICE`, `WORKDIR`, `DRIVE_RUNS`) | c (the old notebook had the token / git cell inline; all shown with Arabic comments) |
| | run (7: `RUN_NAME`, `WORK_ON`, `LOCAL_RUNS`, `RESTORE_FROM`, `OVERWRITE_ON_RESTORE`, `PUSH_TO_DRIVE`, `ARTIFACT`) | b (old `train.run_dir`, `data.filename_base`) |
| | experiment overlays (6: `FX1_BUNDLE`, `FX1_NIG_SCALE`, `KEEP_LAST_WEIGHTS`, `LOAD_WEIGHTS`, `ON_STOP_WEIGHTS`, `TARGET_NORM`) | torch-only; they write into `model.overrides` / `train.early_stop` AFTER `CHANGES` (the overlay wins) |
| | rolling study (3: `ROLLING`, `ROLLING_MODEL`, `ROLLING_PANEL`) | torch-only; `ROLLING` is a dict of `RollingSettings` (10 fields, listed now) |
| | battery of a finished folder (3), reports and plots (6: `LEGACY_REPORT`, `PLOT_*`), `CHANGES`, `SHOW_SETTINGS` (new) | torch-only |
| `data_pipeline.ipynb` (12 variables, 13 now) | `RAW_ROOT`, `OUT_DIR`, `NAME`, `PRESETS` (4 named patches of `data/presets.py`), `FETCH`, `CHANGES` (the `DataSettings` tree, dotted keys), paths and device | `CHANGES` is `ProjectSettings.config_overrides` (b); the 91 `DataSettings` were not listed |
| `colab/job.py` job.json (10 required + 3 optional keys) | `job_id`, `repo_ref`, `entry`, `args`, `config`, `max_minutes`, `device`, `needs_drive`, `expected_outputs`, `note`; `rerun`, `on_existing`, `base_job_id` | torch-only (queue schema, rule E1 / E2) |
| `jobs/baseline.py` `BaselineConfig` (12) | a strict VIEW of `RunSettings`: `device` -> `train.device`, `seed`, `epochs`, `batch_size`, `run_dir`, `artifact` -> `data.name`, `embargo_hours` -> `split.embargo_hours`, `health_max_rows` -> `evaluation.health_max_rows`, `split_dates` -> `split.train_end` / `split.val_end`, `max_rows` -> `split.max_rows`, `base_tf` -> `data.expect_base_tf`; `gate_report` is job-only (rule C6) | b |
| `jobs/rolling.py` (14), `panel.py` (18), `staged.py` (13), `gates.py` (15) | views plus `train` / `split` / `model_overrides` dicts built strictly into `TrainSettings` / `SplitSettings` / `ModelSettings` (so every field of those is reachable); job-only study knobs: `rolling` (`RollingSettings`), `day_metrics`, `seeds`, `arms`, `extra_arms`, `top_coins`, `patience`, `selection_metric`, `battery`, `staged`, `baselines`, and the gates' `e1` / `e2_*` / `e3_*` | b; the study knobs are job-level by design (their entry reads them), not RunSettings fields |

A job cannot reach `evaluation` (battery cost, variants, verdict trim ...) except the panel job's `battery` flag and
`BaselineConfig.health_max_rows`; the notebook can. That is a difference between the two ways to run, not a missing
setting.

## Findings

1. **Fixed here (B5, B6).** `TrainSettings` validated only `batch_by`, `batch_order`, `lr_schedule` at construction;
   `init`, `weight_decay_groups`, `loss_balance` and `selection_metric` were refused only when the trainer was built (after
   the artifact was loaded) and `cross_section_loss` was read as "off" by `RunSettings` for any unknown name. The notebook
   promised "a wrong value = an immediate error". They now raise at construction (`workflow/settings.py`); a valid
   setting behaves as before, no default changed. Found by the new test, which asks every listed set of valid values whether
   the settings themselves refuse a made-up value (`LAZY` in the test names the two settings that only a later step checks:
   `model.architecture`, `source.categories`).
2. **Stale, fixed here.** The card text of `trainer/tracker.py` said the TF `weights_snapshot="ema_weights"` was "not
   ported (needs optimizer EMA)"; the Trainer keeps the EMA copy as the best weights when `train.ema_decay` is set.
3. **Trap, not changed.** `CHANGES["model.overrides"]` REPLACES the default dict: an incomplete dict silently drops the
   anti-memorization switches back to the builder's defaults. The listing's `<- NOW` marks the model options that moved.
   `core.config.updated` cannot set one model option by a dotted key (`model.overrides.encoder`): a proposal below.
4. **Defaults differ from the old preset (facts, unchanged):** epochs 5 (old 60),
   batch size 256 (64), lr 1e-3 (the anti-memorization preset's 3e-4), weight decay 1e-4 (0.05), seed 0 (None = 42).
5. **Hard-coded but tunable in the old project:** `LAMBDA_SCHEDULES` warm-up is 5 epochs and the default run has 5 epochs,
   so the NIG regulariser and the calibration term ramp over the whole default run.

## Proposals for the owner (class d), and why none was added

Each new field of `TrainSettings` / `BatterySettings` costs: the field and its check, a row in `workflow/settings_roles.py`
(a test fails without it), a row in the notebook's `CHANGES` (a test fails without it), the pin of its default in
`tests/pins/defaults.json` with a decision line in the same PR (rule B1, `<new>`), and, when it reaches the run identity,
`IDENTITY_*` in `workflow/run_common.py`. Two other steps edit `workflow/settings.py` and `main.ipynb` in the same round,
so no field was added by this step; the guard test makes the next one cheap (the listing shows it by itself).

| # | proposal | value | cost |
|---|---|---|---|
| P1 | `train.lambda_reg_start`, `_warmup_epochs`, `_schedule` and the same three for `lambda_calib` (6 fields; read in `workflow/flat_run.py::trainer_config`, defaults = `LAMBDA_SCHEDULES`, so the trainer config stays byte-equal) | medium: the warm-up equals the whole default run (finding 5) | 6 fields, 6 role rows, 6 pins + decision |
| P2 | `core.config.updated`: a dotted key into a dict-valued field merges (`"model.overrides.encoder": "gru"`) instead of raising; removes the replace-all trap (finding 3) | high for usability | changes one function and its tests; no default |
| P3 | battery grids as `BatterySettings` fields (`top shares` 0.05 / 0.10 / 0.20 of `trade_grid`, the take-profit grids, the uncertainty quantiles); this is the nearest to the old `mn_quantiles` | medium | `eval/battery.py` signatures + pins |
| P4 | wire `eval/calibrator.py` into a report step and add `evaluation.calibrate_confidence` / `calibration_method` (old `calibrate_confidence`) | medium (it is a step, not a field) | a workflow step + test |
| P5 | per-target loss weights (old `target_loss_weights`) | low until a gate asks for it | trainer config + pins |
| P6 | reduce-on-plateau and class weights (old `reduce_lr_*`, `use_class_weights`) | low (`train.lr_schedule` and `class_loss_weight` cover the experiments run so far) | trainer work |
| P7 | funding rate, open interest and intraday 15m features (old `funding_rate`, `open_interest`, `intraday_*`, `phase2_data`) | data work, not a setting | data pipeline |
| P8 | run-time target mode and group width (`target.target_mode`, `group_freq`) | open | agent 2's step of this round |

## Not verified

* The (c) rows of table B were classified by reading the torch code and the old defaults file, not by building the same
  dataset with both pipelines; the build parity of the fields both have is `tests/data/test_parity_tf.py`.
* That `windows.align_to_grid` makes the old `panel.group` unnecessary was read from the code, not measured.
* The old `train_mode` and `fit_verbose` were classified from their names and the old settings docstrings.
* The notebooks were run headless on the fixture (`tests/notebooks`), not on a real Colab VM.
