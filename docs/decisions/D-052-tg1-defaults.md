# D-052 — Full verification, target modes with decode, settings listing (owner 2026-11-23)

Date: 2026-11-23
Owner decision: owner message 2026-11-23 (translated): "add to the torch project the same tests as `run_full_verification` of the old project; add `retarget_splits` (return, return_close, scaled, magnitude, entry_range, +relative) with decoding for each; update the pipeline so every tunable setting is shown, preferably the same settings as the old project or better"; follow-up: the verification output also shows true move, error of the move and the price error per target in the latest-samples format.
Why: the TF project had a one-call verdict table (data integrity, baselines, confidence and trading, candle shape) and a switchable target mode; torch had neither as a run step, its decode refused three of the modes, and 63 of 152 run settings and all 91 data settings were not visible in the notebook.

## What changes behaviour
- `evaluation.verification.*` (new section, enabled=True, but it only runs when the step or the notebook cell is called; no existing run changes). The R:R cost is the battery's round trip 2 x cost_per_side = 0.16 % (TF used 0.08 %); the nig stop uses the NIG predictive sd (TF used the Student-t scale); the bootstrap unit is the UTC day.
- `split.target_mode` (None = the pipeline targets, today's behaviour), `split.entry_close_reg` ('abs_return', the TF default) and `split.target_clip` (None = the mode's own clip): the run-time target switch; with the defaults the run identity and every result are unchanged. `data.targets` gains `retarget_splits`, `decode_affine`, `entry_range_to_prices`, `TargetInfo.centers` and the optional decode arguments `sign` / `range_prices` / `center` (all default None: today's calls are unchanged).
- `check_decode` keeps only rows with `|y/scale| < 0.999 * clip` (clip = the stamp, else 1.0): identical for a dataset without a stamped clip. A TF artifact declaring a non-'return' `reg_target_mode` is now refused only if its labels fail the decode round trip.
- `TrainSettings` rejects an unknown `init`, `weight_decay_groups`, `loss_balance`, `selection_metric` or `cross_section_loss` at construction instead of at trainer build (valid runs unchanged).
- `run_common.decode_anchors(truth)` / `verification_step(verbose)` and the other `(new default)` rows below are parameters of new functions.

- `nig_timenet_torch.data.targets.TargetInfo(centers)`: `<new>` -> `{}`
- `nig_timenet_torch.data.targets.decode_affine(center)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.decode_affine(close_reg)`: `<new>` -> `'abs_return'`
- `nig_timenet_torch.data.targets.decode_affine(range_prices)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.decode_affine(scale)`: `<new>` -> `1.0`
- `nig_timenet_torch.data.targets.decode_affine(sign)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.decode_reg(center)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.decode_reg(close_reg)`: `<new>` -> `'abs_return'`
- `nig_timenet_torch.data.targets.decode_reg(range_prices)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.decode_reg(sign)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.entry_range_to_prices(close)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.entry_range_to_prices(close_reg)`: `<new>` -> `'abs_return'`
- `nig_timenet_torch.data.targets.entry_range_to_prices(high)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.entry_range_to_prices(low)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.entry_range_to_prices(p_close_up)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.retarget_splits(center)`: `<new>` -> `'median'`
- `nig_timenet_torch.data.targets.retarget_splits(clip)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.retarget_splits(close_reg)`: `<new>` -> `'abs_return'`
- `nig_timenet_torch.data.targets.retarget_splits(group_freq)`: `<new>` -> `''`
- `nig_timenet_torch.data.targets.retarget_splits(min_group)`: `<new>` -> `5`
- `nig_timenet_torch.data.targets.retarget_splits(mode)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.retarget_splits(reg_target_scale)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.retarget_splits(targets)`: `<new>` -> `None`
- `nig_timenet_torch.data.targets.row_anchors(truth)`: `<new>` -> `False`
- `nig_timenet_torch.eval.verification.DEFAULT_TARGET_MODES`: `<new>` -> `(None, 'return')`
- `nig_timenet_torch.eval.verification.run_full_verification(latest_coins)`: `<new>` -> `3`
- `nig_timenet_torch.eval.verification.run_full_verification(latest_n)`: `<new>` -> `1`
- `nig_timenet_torch.eval.verification.run_full_verification(min_n)`: `<new>` -> `200`
- `nig_timenet_torch.eval.verification.run_full_verification(n_boot)`: `<new>` -> `500`
- `nig_timenet_torch.eval.verification.run_full_verification(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.eval.verification.run_full_verification(rr)`: `<new>` -> `2.0`
- `nig_timenet_torch.eval.verification.run_full_verification(run_candle)`: `<new>` -> `True`
- `nig_timenet_torch.eval.verification.run_full_verification(stop_mode)`: `<new>` -> `'nig'`
- `nig_timenet_torch.eval.verification.run_full_verification(target_acc)`: `<new>` -> `0.65`
- `nig_timenet_torch.eval.verification.run_full_verification(target_mode)`: `<new>` -> `None`
- `nig_timenet_torch.eval.verification.run_full_verification(verbose)`: `<new>` -> `False`
- `nig_timenet_torch.eval.verification_candle.candle_baseline_report(max_train)`: `<new>` -> `300000`
- `nig_timenet_torch.eval.verification_candle.candle_baseline_report(min_auc_gain)`: `<new>` -> `0.005`
- `nig_timenet_torch.eval.verification_candle.candle_baseline_report(n_boot)`: `<new>` -> `300`
- `nig_timenet_torch.eval.verification_candle.candle_baseline_report(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.verification_latest.latest_by_target(n)`: `<new>` -> `1`
- `nig_timenet_torch.eval.verification_latest.latest_by_target(n_coins)`: `<new>` -> `3`
- `nig_timenet_torch.eval.verification_trading.rr_trading_report(min_trades)`: `<new>` -> `200`
- `nig_timenet_torch.eval.verification_trading.rr_trading_report(rr)`: `<new>` -> `2.0`
- `nig_timenet_torch.eval.verification_trading.rr_trading_report(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.verification_trading.rr_trading_report(stop_mode)`: `<new>` -> `'nig'`
- `nig_timenet_torch.eval.verification_trading.selective_direction_report(min_n)`: `<new>` -> `200`
- `nig_timenet_torch.eval.verification_trading.selective_direction_report(target_acc)`: `<new>` -> `0.65`
- `nig_timenet_torch.eval.verification_trading.simulate_rr_trades(stop_k)`: `<new>` -> `1.0`
- `nig_timenet_torch.workflow.flat_run.anchors_by_name(dataset)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.run_common.decode_anchors(truth)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.settings.EvalSettings(verification)`: `<new>` -> `VerificationSettings(enabled=True, target_acc=0.65, rr=2.0, stop_mode='nig', n_boot=500, run_candle=True, latest_n=1, latest_coins=3)`
- `nig_timenet_torch.workflow.settings.RunSettings(evaluation)`: `EvalSettings(chunk=20000, health_max_rows=100000, battery=BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1, day_metrics=False, verdict_trim='best', drop_flat=True, account_books=0))` -> `EvalSettings(chunk=20000, health_max_rows=100000, battery=BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1, day_metrics=False, verdict_trim='best', drop_flat=True, account_books=0), verification=VerificationSettings(enabled=True, target_acc=0.65, rr=2.0, stop_mode='nig', n_boot=500, run_candle=True, latest_n=1, latest_coins=3))`
- `nig_timenet_torch.workflow.settings.RunSettings(split)`: `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None, exclude_coins=(), cross_section=False, top_coins=None, coin_channel=False)` -> `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None, exclude_coins=(), cross_section=False, top_coins=None, coin_channel=False, target_mode=None, entry_close_reg='abs_return', target_clip=None)`
- `nig_timenet_torch.workflow.settings.SplitSettings(entry_close_reg)`: `<new>` -> `'abs_return'`
- `nig_timenet_torch.workflow.settings.SplitSettings(target_clip)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.SplitSettings(target_mode)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.VerificationSettings(enabled)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.VerificationSettings(latest_coins)`: `<new>` -> `3`
- `nig_timenet_torch.workflow.settings.VerificationSettings(latest_n)`: `<new>` -> `1`
- `nig_timenet_torch.workflow.settings.VerificationSettings(n_boot)`: `<new>` -> `500`
- `nig_timenet_torch.workflow.settings.VerificationSettings(rr)`: `<new>` -> `2.0`
- `nig_timenet_torch.workflow.settings.VerificationSettings(run_candle)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.VerificationSettings(stop_mode)`: `<new>` -> `'nig'`
- `nig_timenet_torch.workflow.settings.VerificationSettings(target_acc)`: `<new>` -> `0.65`
- `nig_timenet_torch.workflow.verification_run.verification_step(verbose)`: `<new>` -> `False`
