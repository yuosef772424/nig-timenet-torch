# D-006 — autoregressive candle stream (feat/ar-stream): new defaults

Date: 2026-11-06
Owner decision: the owner's session prompt (2026-11-05/06): an autoregressive next-candle stream in the
spirit of a text Transformer — a free window (32 context + a FREE horizon, 16 by default), the model eats its
own predictions step by step; free coin selection; free evaluation horizon (may differ from training); a
panel variant in its own notebook; a pluggable trunk (transformer | gru). The T1 rule applies: nothing
below changes an existing default; every key is NEW (the `<new>` format of docs/decisions/README.md), pinned
via `python -m tests.defaults_pin --add`.

Design notes: the loss is SELECTABLE (`loss`): the default "czar" scores each return in units of its
window's LOCAL volatility (vol_floor in fractions of the train-head median |return|) but PAYS in return
units (the magnitude term is multiplied by that local vol, w_mag=0.15): regime-fair in value while the
gradient stays ~O(1) — a calm window is never dwarfed by a wild one and the magnitude lesson never drowns
the direction lesson. CZAR (Pfeffer et al. 2026, arXiv:2609.36061): a zero prediction on a real move stays
expensive (anti-cowardice), a shock is bounded by the Cauchy kernel, near-zero truths are softened by a
zero-agnostic floor (czar_epsilon, czar_tau). The direction BCE excludes flat truths and ramps its weight
with the move in local-vol units (w_move). loss="huber" keeps the v1 robust loss: full-weight direction,
Huber-capped magnitude with the adaptive outlier weight (outlier_softness in units of the train-head return
scale). Scheduled sampling feeds the model's own DETACHED predictions so it does not learn from noise; the
shuffled-labels control's expected value is the honest null (the majority agreement of a biased model,
not a flat 50%).

New defaults:

- `nig_timenet_torch.data.ar_windows.CoinWindows(end_times)`: `<new>` -> `None`
- `nig_timenet_torch.data.ar_windows.build_windows(index)`: `<new>` -> `None`
- `nig_timenet_torch.data.ar_windows.build_windows(stride)`: `<new>` -> `1`
- `nig_timenet_torch.data.ar_windows.fit_volume_median(train_share)`: `<new>` -> `0.7`
- `nig_timenet_torch.data.ar_windows.raw_candles(interval)`: `<new>` -> `'1h'`
- `nig_timenet_torch.data.ar_windows.raw_candles(root)`: `<new>` -> `'.'`
- `nig_timenet_torch.data.ar_windows.raw_candles(seed)`: `<new>` -> `0`
- `nig_timenet_torch.data.ar_windows.resolve_coin_list(root)`: `<new>` -> `'.'`
- `nig_timenet_torch.data.ar_windows.split_panel(test_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.data.ar_windows.split_panel(val_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.data.ar_windows.split_windows(test_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.data.ar_windows.split_windows(val_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.eval.ar_stream.direction_control(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.eval.ar_stream.direction_control(horizon_len)`: `<new>` -> `None`
- `nig_timenet_torch.eval.ar_stream.direction_control(n_shuffles)`: `<new>` -> `5`
- `nig_timenet_torch.eval.ar_stream.direction_control(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.ar_stream.panel_direction_control(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.eval.ar_stream.panel_direction_control(horizon_len)`: `<new>` -> `None`
- `nig_timenet_torch.eval.ar_stream.panel_direction_control(n_shuffles)`: `<new>` -> `5`
- `nig_timenet_torch.eval.ar_stream.panel_direction_control(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.ar_stream.panel_rollout_metrics(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.eval.ar_stream.panel_rollout_metrics(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.eval.ar_stream.panel_rollout_metrics(horizon_len)`: `<new>` -> `None`
- `nig_timenet_torch.eval.ar_stream.rollout_metrics(batch_size)`: `<new>` -> `1024`
- `nig_timenet_torch.eval.ar_stream.rollout_metrics(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.eval.ar_stream.rollout_metrics(horizon_len)`: `<new>` -> `None`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(attn_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(context_len)`: `<new>` -> `32`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(d_model)`: `<new>` -> `64`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(encoder)`: `<new>` -> `'transformer'`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(horizon_len)`: `<new>` -> `16`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(max_rel_pos)`: `<new>` -> `64`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(num_kv_heads)`: `<new>` -> `2`
- `nig_timenet_torch.model.ar_candle_transformer.ArCandleTransformer(num_layers)`: `<new>` -> `2`
- `nig_timenet_torch.model.ar_candle_transformer.rollout(horizon_len)`: `<new>` -> `None`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(attn_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(context_len)`: `<new>` -> `32`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(cross_layers)`: `<new>` -> `1`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(d_model)`: `<new>` -> `64`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(encoder)`: `<new>` -> `'transformer'`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(horizon_len)`: `<new>` -> `16`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(max_rel_pos)`: `<new>` -> `64`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(num_kv_heads)`: `<new>` -> `2`
- `nig_timenet_torch.model.panel_ar_candle.PanelArCandleTransformer(num_layers)`: `<new>` -> `2`
- `nig_timenet_torch.model.panel_ar_candle.build_panel_ar_candle(context_len)`: `<new>` -> `32`
- `nig_timenet_torch.model.panel_ar_candle.build_panel_ar_candle(horizon_len)`: `<new>` -> `16`
- `nig_timenet_torch.model.panel_ar_candle.rollout(horizon_len)`: `<new>` -> `None`
- `nig_timenet_torch.model.panel_ar_candle.rollout(mask)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(context_len)`: `<new>` -> `32`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(czar_alpha)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(czar_epsilon)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(czar_tau)`: `<new>` -> `0.05`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(d_model)`: `<new>` -> `64`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(encoder)`: `<new>` -> `'transformer'`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(epochs)`: `<new>` -> `10`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(grad_clip)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(horizon_len)`: `<new>` -> `16`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(huber_delta)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(loss)`: `<new>` -> `'czar'`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(lr)`: `<new>` -> `0.001`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(num_kv_heads)`: `<new>` -> `2`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(num_layers)`: `<new>` -> `2`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(outlier_softness)`: `<new>` -> `8.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(patience)`: `<new>` -> `4`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(ss_end)`: `<new>` -> `0.5`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(ss_start)`: `<new>` -> `0.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(verbose)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(vol_floor)`: `<new>` -> `0.25`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(w_dir)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(w_mag)`: `<new>` -> `0.15`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(w_move)`: `<new>` -> `0.5`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(w_sigma)`: `<new>` -> `0.1`
- `nig_timenet_torch.trainer.ar_stream.ArStreamSettings(weight_decay)`: `<new>` -> `0.0001`
- `nig_timenet_torch.trainer.ar_stream.czar_loss(alpha)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.czar_loss(epsilon)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.ar_stream.czar_loss(tau)`: `<new>` -> `0.05`
- `nig_timenet_torch.trainer.ar_stream.rollout_direction_accuracy(batch_size)`: `<new>` -> `1024`
- `nig_timenet_torch.trainer.ar_stream.rollout_direction_accuracy(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(context_len)`: `<new>` -> `32`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(cross_layers)`: `<new>` -> `1`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(czar_alpha)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(czar_epsilon)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(czar_tau)`: `<new>` -> `0.05`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(d_model)`: `<new>` -> `64`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(encoder)`: `<new>` -> `'transformer'`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(epochs)`: `<new>` -> `10`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(grad_clip)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(horizon_len)`: `<new>` -> `16`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(huber_delta)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(loss)`: `<new>` -> `'czar'`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(lr)`: `<new>` -> `0.001`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(num_kv_heads)`: `<new>` -> `2`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(num_layers)`: `<new>` -> `2`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(outlier_softness)`: `<new>` -> `8.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(patience)`: `<new>` -> `4`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(ss_end)`: `<new>` -> `0.5`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(ss_start)`: `<new>` -> `0.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(verbose)`: `<new>` -> `True`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(vol_floor)`: `<new>` -> `0.25`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(w_dir)`: `<new>` -> `1.0`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(w_mag)`: `<new>` -> `0.15`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(w_move)`: `<new>` -> `0.5`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(w_sigma)`: `<new>` -> `0.1`
- `nig_timenet_torch.trainer.panel_ar_stream.PanelArStreamSettings(weight_decay)`: `<new>` -> `0.0001`
- `nig_timenet_torch.trainer.panel_ar_stream.panel_rollout_direction_accuracy(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.trainer.panel_ar_stream.panel_rollout_direction_accuracy(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.workflow.ar_stream.run_ar_stream(changes)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.ar_stream.run_ar_stream(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.workflow.ar_stream.run_ar_stream(test_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.workflow.ar_stream.run_ar_stream(val_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.workflow.ar_stream.run_panel_ar_stream(changes)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.ar_stream.run_panel_ar_stream(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.workflow.ar_stream.run_panel_ar_stream(test_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.workflow.ar_stream.run_panel_ar_stream(val_share)`: `<new>` -> `0.15`
