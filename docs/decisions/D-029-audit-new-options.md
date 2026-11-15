# D-029: new options and helper defaults of the model-first audit (PRs #55 and #56), all at the old behaviour

Date: 2026-11-15
Owner decision: `docs/audit/FABLE_DESIGN_AUDIT_PROMPT.md` (owner: "the session may edit anything"; "keep the old
behaviour reachable only if it is cheap") and the owner's requests of 2026-11-15 in the PR #55 sessions (attention
head experiment, `steps_per_execution`, removing SwiGLU, a transformer with the useful parts of the other encoders).
Evidence: `docs/audit/2026-11-14-model-first/NOTES.md` (N-01 .. N-25), `TRANSFORMER_HEAD_RESULTS.md`,
`QUICK_ARMS_RESULTS.md`; tests under `tests/doctor/`, `tests/model/`, `tests/trainer/`.

## What changes behaviour
- Nothing in this file: every line below is a NEW default (a new option or a new helper's argument) whose value keeps
  the behaviour before it existed. The default CHANGES of the audit are in D-026 (model-first defaults) and D-027
  (`steps_per_execution`). This file exists because the CI step `python -m tests.pin_diff origin/main` asks for a
  decision line for every key the PR adds to `tests/pins/defaults.json`; the PR was red on it since its first push.

## New defaults (pinned as they are)
- `nig_timenet_torch.colab_ops.recipes.arms_line(battery)`: `<new>` -> `True`
- `nig_timenet_torch.colab_ops.recipes.arms_line(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.colab_ops.recipes.arms_line(repo)`: `<new>` -> `'/content/nig-timenet-torch'`
- `nig_timenet_torch.colab_ops.recipes.arms_line(spe)`: `<new>` -> `1`
- `nig_timenet_torch.colab_ops.recipes.doctor_line(batch_size)`: `<new>` -> `128`
- `nig_timenet_torch.colab_ops.recipes.doctor_line(repo)`: `<new>` -> `'/content/nig-timenet-torch'`
- `nig_timenet_torch.colab_ops.recipes.doctor_line(seed)`: `<new>` -> `0`
- `nig_timenet_torch.colab_ops.recipes.env_script(repo)`: `<new>` -> `'/content/nig-timenet-torch'`
- `nig_timenet_torch.colab_ops.recipes.env_script(torch_key)`: `<new>` -> `None`
- `nig_timenet_torch.colab_ops.recipes.runs_line(code)`: `<new>` -> `'/content/nig_uhub/src'`
- `nig_timenet_torch.colab_ops.recipes.runs_line(device)`: `<new>` -> `'cuda'`
- `nig_timenet_torch.colab_ops.recipes.runs_line(script)`: `<new>` -> `'/content/run_models.py'`
- `nig_timenet_torch.colab_ops.recipes.setup_script(artifact)`: `<new>` -> `True`
- `nig_timenet_torch.colab_ops.recipes.setup_script(dest)`: `<new>` -> `'/content/nig-timenet-torch'`
- `nig_timenet_torch.colab_ops.recipes.setup_script(install)`: `<new>` -> `True`
- `nig_timenet_torch.colab_ops.recipes.setup_script(torch_key)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.vision_slice.DEFAULT_COINS`: `<new>` -> `('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'TRXUSDT', 'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'ETCUSDT', 'FILUSDT', 'XLMUSDT', 'AAVEUSDT', 'ALGOUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'EOSUSDT', 'XTZUSDT', 'THETAUSDT', 'VETUSDT', 'ICPUSDT', 'GRTUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT', 'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'SNXUSDT', 'COMPUSDT', 'ZECUSDT', 'DASHUSDT', 'ENJUSDT', 'CHZUSDT', 'KAVAUSDT', 'GALAUSDT', 'IMXUSDT', 'LDOUSDT', 'FETUSDT', 'EGLDUSDT', 'HBARUSDT', 'QNTUSDT', 'NEOUSDT', 'IOTAUSDT', 'ZILUSDT', 'BATUSDT', 'ONEUSDT', 'CELOUSDT', 'ANKRUSDT', 'SUSHIUSDT')`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(horizon)`: `<new>` -> `2`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(interval)`: `<new>` -> `'4h'`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(stride)`: `<new>` -> `1`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(time_seal)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(window)`: `<new>` -> `32`
- `nig_timenet_torch.data_io.vision_slice.fetch_coin(get)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.vision_slice.fetch_month(get)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.Check(action)`: `<new>` -> `''`
- `nig_timenet_torch.doctor.common.Check(baseline)`: `<new>` -> `''`
- `nig_timenet_torch.doctor.common.Check(detail)`: `<new>` -> `{}`
- `nig_timenet_torch.doctor.common.Check(verdict)`: `<new>` -> `'info'`
- `nig_timenet_torch.doctor.common.DoctorContext(notes)`: `<new>` -> `{}`
- `nig_timenet_torch.doctor.common.DoctorContext(quick)`: `<new>` -> `True`
- `nig_timenet_torch.doctor.common.DoctorContext(seed)`: `<new>` -> `0`
- `nig_timenet_torch.doctor.common.fresh_trainer(model)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.fresh_trainer(seed)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.fresh_trainer(train)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.judge(fail_above)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.judge(fail_below)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.judge(warn_above)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.judge(warn_below)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.make_context(architecture)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.make_context(batch_size)`: `<new>` -> `512`
- `nig_timenet_torch.doctor.common.make_context(epochs)`: `<new>` -> `3`
- `nig_timenet_torch.doctor.common.make_context(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.make_context(model)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.make_context(no_preset)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.common.make_context(quick)`: `<new>` -> `True`
- `nig_timenet_torch.doctor.common.make_context(run_dir)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.make_context(seed)`: `<new>` -> `0`
- `nig_timenet_torch.doctor.common.make_context(train)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.make_context(trained)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.common.predict_outputs(chunk)`: `<new>` -> `4096`
- `nig_timenet_torch.doctor.common.predict_outputs(train_mode)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.common.probe_scores(squares)`: `<new>` -> `True`
- `nig_timenet_torch.doctor.common.render_table(only)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.common.ridge_oof(lambdas)`: `<new>` -> `(1.0, 10.0, 100.0, 1000.0, 10000.0)`
- `nig_timenet_torch.doctor.common.time_folds(embargo)`: `<new>` -> `0`
- `nig_timenet_torch.doctor.common.time_folds(n_folds)`: `<new>` -> `4`
- `nig_timenet_torch.doctor.common.write_report(header)`: `<new>` -> `''`
- `nig_timenet_torch.doctor.data_checks.alignment(max_pairs)`: `<new>` -> `20000`
- `nig_timenet_torch.doctor.data_checks.batch_independence(rows)`: `<new>` -> `64`
- `nig_timenet_torch.doctor.data_checks.ceiling(boost)`: `<new>` -> `True`
- `nig_timenet_torch.doctor.info_flow.collect_streams(chunk)`: `<new>` -> `2048`
- `nig_timenet_torch.doctor.info_flow.collect_streams(edit)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.info_flow.info_flow(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.doctor.metric_checks.metric_checks(include_test)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.readiness.run_cards(cards)`: `<new>` -> `('C', 'B', 'A', 'D')`
- `nig_timenet_torch.doctor.readiness.run_cards(include_test)`: `<new>` -> `False`
- `nig_timenet_torch.doctor.trainer_checks.constant_loss(rows)`: `<new>` -> `20000`
- `nig_timenet_torch.doctor.trainer_checks.constant_loss(stage)`: `<new>` -> `'val'`
- `nig_timenet_torch.doctor.trainer_checks.lr_range_test(hi)`: `<new>` -> `1.0`
- `nig_timenet_torch.doctor.trainer_checks.lr_range_test(lo)`: `<new>` -> `1e-06`
- `nig_timenet_torch.doctor.trainer_checks.lr_range_test(steps)`: `<new>` -> `80`
- `nig_timenet_torch.doctor.trainer_checks.mode_gap(passes)`: `<new>` -> `8`
- `nig_timenet_torch.doctor.trainer_checks.mode_gap(rows)`: `<new>` -> `2048`
- `nig_timenet_torch.doctor.trainer_checks.overfit_one_batch(no_regularisation)`: `<new>` -> `True`
- `nig_timenet_torch.doctor.trainer_checks.overfit_one_batch(steps)`: `<new>` -> `300`
- `nig_timenet_torch.doctor.trainer_checks.resume_check(rows)`: `<new>` -> `2048`
- `nig_timenet_torch.model.attn_heads.LinearNIGHead(dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(alpha_min)`: `<new>` -> `2.0`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(beta_min)`: `<new>` -> `0.01`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(l2)`: `<new>` -> `1e-05`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(mode)`: `<new>` -> `'joint'`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(nu_min)`: `<new>` -> `0.1`
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(scale)`: `<new>` -> `None`
- `nig_timenet_torch.model.attn_heads.ScalarOutputAttention(dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.attn_heads.ScalarOutputAttention(l2)`: `<new>` -> `1e-05`
- `nig_timenet_torch.model.attn_heads.ScalarOutputAttention(mode)`: `<new>` -> `'joint'`
- `nig_timenet_torch.model.attn_heads.ScalarOutputAttention(num_outputs)`: `<new>` -> `4`
- `nig_timenet_torch.model.builder.NigTimeNetV2(ffn)`: `<new>` -> `'swiglu'`
- `nig_timenet_torch.model.builder.NigTimeNetV2(ffn_dim)`: `<new>` -> `None`
- `nig_timenet_torch.model.builder.NigTimeNetV2(input_scaler)`: `<new>` -> `'none'`
- `nig_timenet_torch.model.builder.NigTimeNetV2(nig_head)`: `<new>` -> `'dense'`
- `nig_timenet_torch.model.builder.NigTimeNetV2(nig_head_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(ffn)`: `<new>` -> `'swiglu'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(ffn_dim)`: `<new>` -> `None`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(input_scaler)`: `<new>` -> `'none'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(nig_head)`: `<new>` -> `'dense'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(nig_head_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.encoders.EncoderDims(ffn)`: `<new>` -> `'swiglu'`
- `nig_timenet_torch.model.encoders.EncoderDims(ffn_dim)`: `<new>` -> `None`
- `nig_timenet_torch.model.layers.TransformerBlock(ffn)`: `<new>` -> `'swiglu'`
- `nig_timenet_torch.model.layers.TransformerBlock(ffn_dim)`: `<new>` -> `None`
- `nig_timenet_torch.model.scalers.InputScaler(clip)`: `<new>` -> `5.0`
- `nig_timenet_torch.model.taps.stream_taps(edit)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.trainer.FitSettings(steps_per_execution)`: `<new>` -> `1`
- `nig_timenet_torch.workflow.settings.TrainSettings(steps_per_execution)`: `<new>` -> `1`
## Added by the merge of #51, #52 and #56 (D-023 `fitted_scale` reaches the attention head; off = the old behaviour)
## Added by the merge of #51, #52 and #56 (this file only carries the keys that exist after the merge)
- `nig_timenet_torch.model.attn_heads.ScalarNIGHead(fitted_scale)`: `<new>` -> `False`
