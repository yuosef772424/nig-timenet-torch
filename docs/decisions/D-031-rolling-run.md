# D-031: rolling walk-forward on the run steps (ROADMAP F2, rolling half of I4): new options at the old behaviour

Date: 2026-11-15
Owner decision: ROADMAP F2 / decision O-2 (walk-forward in) and the session prompt that asked for the build ("a fold
has no validation set (an empty val), so decide how early stopping or epoch selection works inside a fold ... keep it
honest: no peeking at the test window").

## What changes behaviour
- Nothing by default: single-split runs, the seam functions of `workflow/rolling.py` (`rolling_folds`,
  `run_rolling_study`) and `RollingSettings` without `val_share` behave as before.
- DECISION (epoch selection inside a fold): the fold's validation is the LAST `val_share` (0.15) of its train window by
  time. Train keeps the rows with `label_end <= cut_v`, validation the rows with `feature_start > cut_v + embargo` whose
  labels end by the fold's cut, `cut_v = train_end - val_share * (train_end - train_start)`; `check_split_purity`
  runs on train -> val -> test. The existing epoch selection (`val_raw_loss`, early stopping, best weights) then works
  per fold unchanged. The test window is read by nothing before the report. Alternatives rejected: a fixed epoch count
  and the last weights (no early stop, overfit after epoch 1 on the real data, B-038); the epoch count chosen on fold 0
  (a one-fold choice applied to later folds); selection on the test window (peeking). Cost: the model never trains on
  the newest 15 % of each window (the price of an honest selection; `val_share` is a setting).
- Every fold fits the scalers and the NIG label scale on its own train rows (`steps.build` on the fold's Run); a fold
  that reuses the last model (`retrain_every_k`) is evaluated with that model's scalers.

## New defaults (pinned as they are)
- `nig_timenet_torch.jobs.rolling.CONFIG`: `<new>` -> `<class 'nig_timenet_torch.jobs.rolling.RollingConfig'>`
- `nig_timenet_torch.jobs.rolling.RollingConfig(artifact)`: `<new>` -> `'/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1'`
- `nig_timenet_torch.jobs.rolling.RollingConfig(base_tf)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.rolling.RollingConfig(batch_size)`: `<new>` -> `512`
- `nig_timenet_torch.jobs.rolling.RollingConfig(lr)`: `<new>` -> `0.001`
- `nig_timenet_torch.jobs.rolling.RollingConfig(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.rolling.RollingConfig(model_overrides)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.rolling.RollingConfig(split)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.rolling.RollingConfig(train)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.rolling.run_settings(code_sha)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.flat_run.predict_stage(fold)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.rolling_run.run_rolling(ckpt_root)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.rolling_run.run_rolling(on_epoch)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.rolling_run.run_rolling(on_fold)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run.Run(fold)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.settings.RollingSettings(val_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.workflow.study_settings.job_run_settings(code_sha)`: `<new>` -> `''`
