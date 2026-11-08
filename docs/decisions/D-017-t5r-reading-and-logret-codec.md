# D-017: T5r reading tools, the log-close target codec and the decode guard (ROADMAP T5r)

Date: 2026-11-08
Owner decision: step T5r (manager message of 2026-11-08: "PHASE 2 (PR 2): step T5r, reading the baseline", and the owner's
addition "Model error analysis for improvement"; "report any mismatch as a defect"; "do not change any model default").

What changes behaviour (no MODEL or training default changes; training reads the same labels as before):

- B-051 fix: `workflow.run_common.codec_state` now picks the codec with `data_io.target_codec.codec_name`: an artifact whose
  `reg_target_mode` is "return" but whose `price_norm_mode` is a TF logret_* mode ("logret", "logret_std") is decoded with
  the new `logret_close` codec (label = log(future / last_close) for every head, the anchor of every head is last_close).
  The TF artifact `c572b1b36960` is such an artifact. Before, its labels were decoded as own-anchor simple returns, which
  moved every decoded high / low price by the last bar's high-close (low-close) gap. Outputs that change for this artifact:
  the T-sharp `share_std_below_move` / `median_std_over_move` of the test and health reports, and every battery table
  column built from a decoded price (y_ret, pred_high / fut_high, pred_low / fut_low, sigma_*), so every battery section.
  Rank IC, coverage, AUC and T-conf are computed in label space and do not change. An artifact without a logret mode
  (the fixture, the native pipeline) decodes exactly as before.
- `workflow.run.prepare` calls `run_common.check_decode` (rule B3 guard): it refuses an artifact whose stored labels the
  codec does not decode back to the future prices of last_candles (2000 strided rows, relative error > 1e-4).
- `workflow.flat_run.anchors_by_name(codec_state=None)`: without a codec state the own last price, as before; the run steps
  now pass the run's codec state, so the anchor follows the codec.

New defaults (each a new option; none changes an existing behaviour):

- `workflow.run.restore(allow_changes=("code_sha",))`: a finished run folder is re-read at another commit; settings,
  seed and data must be identical (RunIdentity, check_resume).
- `run_common.check_decode(n=2000, rtol=1e-4)`.
- Reading tools (new modules, analysis only): `eval.vol_baseline`, `eval.strategy_book` (cost 0.0008 per side, TP at
  0.5 of the predicted extreme, filters at the VAL quantiles 0.5 / 0.8 / 0.9, CI 95 % day-block bootstrap, seed 0),
  `eval.strategy_diagnosis`, `eval.error_map`, `workflow.reading`.

- `nig_timenet_torch.eval.error_map.bias(n)`: `<new>` -> `10`
- `nig_timenet_torch.eval.error_map.error_map(coin_info)`: `<new>` -> `None`
- `nig_timenet_torch.eval.error_map.error_map(heads)`: `<new>` -> `('high', 'low', 'close')`
- `nig_timenet_torch.eval.error_map.uncertainty(n)`: `<new>` -> `10`
- `nig_timenet_torch.eval.error_map.worst_cases(coin_info)`: `<new>` -> `None`
- `nig_timenet_torch.eval.error_map.worst_cases(k)`: `<new>` -> `15`
- `nig_timenet_torch.eval.strategy_book.book(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.eval.strategy_book.book(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.strategy_book.limit_extreme_trades(pred_high)`: `<new>` -> `None`
- `nig_timenet_torch.eval.strategy_book.limit_extreme_trades(pred_low)`: `<new>` -> `None`
- `nig_timenet_torch.eval.strategy_book.summarize(level)`: `<new>` -> `0.95`
- `nig_timenet_torch.eval.strategy_book.summarize(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.strategy_book.tp_trades(need_close_down)`: `<new>` -> `True`
- `nig_timenet_torch.eval.strategy_book.tp_trades(pred_high)`: `<new>` -> `None`
- `nig_timenet_torch.eval.strategy_book.tp_trades(pred_low)`: `<new>` -> `None`
- `nig_timenet_torch.eval.strategy_book.tp_trades(share)`: `<new>` -> `0.5`
- `nig_timenet_torch.eval.strategy_diagnosis.days(k)`: `<new>` -> `3`
- `nig_timenet_torch.eval.strategy_diagnosis.payoff(n_worst)`: `<new>` -> `5`
- `nig_timenet_torch.eval.vol_baseline.head_vs_baselines(sign)`: `<new>` -> `1.0`
- `nig_timenet_torch.eval.vol_baseline.interval_vs_baseline(level)`: `<new>` -> `0.8`
- `nig_timenet_torch.workflow.flat_run.anchors_by_name(codec_state)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.reading.reading_step(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.run.restore(allow_changes)`: `<new>` -> `('code_sha')`
- `nig_timenet_torch.workflow.run_common.check_decode(n)`: `<new>` -> `2000`
- `nig_timenet_torch.workflow.run_common.check_decode(rtol)`: `<new>` -> `0.0001`
