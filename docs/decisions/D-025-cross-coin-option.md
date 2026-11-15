# D-025: cross-coin option of the default architecture (`cross_coin_layers`, `SplitSettings.cross_section`)

Date: 2026-11-15
Owner decision: owner request 2026-11-15 ("improve the panel model that works on several coins at the same time and
test it"; "400 lines is not a condition, do not stick to the rules").
Evidence: `tests/model/test_cross_coin.py` (identity at init, a row depends on its own timestamp only, the default
builds the same keys, batches and chunks hold whole timestamps), `tests/workflow/test_run.py` (a cross-section run
end to end); measured arms in `docs/audit/2026-11-14-model-first/TRANSFORMER_HEAD_RESULTS.md` section 5.

## What changes behaviour
- Nothing at the defaults: `cross_coin_layers = 0` builds no module, `cross_section = False` keeps the row order, the
  row subsample, the input width and the shuffled batches as they were. `Trainer.group_channel` is False unless the
  run sets `cross_section`.
- `model/panel.py` (the TF port: its own heads, a (day, pos) grid from the caller, no training path here) stays as it
  is; the new block is the trained route to a multi-coin model.
- Rule G3: `NigTimeNetV2.__init__` passes 80 lines (ratchet 4 -> 5), accepted by the owner's message above.
- Rule G3, module budget: 400 -> 1000 lines (owner 2026-11-15: "remove the 400-line rule, make it 1000"); the one
  size exception (`doctor/trainer_checks.py`) is no longer needed and is removed from `tests/ratchets.json`.

## Changed default reprs (a new field at its old behaviour)
- `nig_timenet_torch.workflow.settings.RunSettings(split)`: `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None)` -> `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None, cross_section=False)`
- `nig_timenet_torch.workflow.settings.RunSettings(split)`: `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None, cross_section=False)` -> `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None, cross_section=False, top_coins=None)`

## Added 2026-11-15 (same owner request: "choose the best 100 coins", "features in the differences between coins")
- `SplitSettings.top_coins` (None = all coins): keeps the N coins with the most TRAIN rows in every split.
- Model option `cross_input` (`none` default | `relative`): each coin's scaled window is extended, bar by bar, with
  its difference from the timestamp's mean, that difference in units of the timestamp's sd, and the mean itself.
- Selection metrics `val_rank_ic_reg` and `val_rank_ic_<target>_reg` (cross-sectional rank IC of the NIG mean; the
  trainer passes the timestamps on a cross-section run). The default stays `val_raw_loss`.

## New defaults (pinned as they are; each keeps the behaviour before the option existed)
- `nig_timenet_torch.model.builder.NigTimeNetV2(cross_coin_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.builder.NigTimeNetV2(cross_coin_kind)`: `<new>` -> `'attention'`
- `nig_timenet_torch.model.builder.NigTimeNetV2(cross_coin_layers)`: `<new>` -> `0`
- `nig_timenet_torch.model.builder.NigTimeNetV2(cross_coin_peer_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.builder.NigTimeNetV2(cross_input)`: `<new>` -> `'none'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(cross_coin_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(cross_coin_kind)`: `<new>` -> `'attention'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(cross_coin_layers)`: `<new>` -> `0`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(cross_coin_peer_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(cross_input)`: `<new>` -> `'none'`
- `nig_timenet_torch.model.cross_coin.CrossCoin(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.model.cross_coin.CrossCoin(kind)`: `<new>` -> `'attention'`
- `nig_timenet_torch.model.cross_coin.CrossCoin(layers)`: `<new>` -> `1`
- `nig_timenet_torch.model.cross_coin.CrossCoin(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.cross_coin.CrossCoin(peer_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.cross_coin.CrossCoinBlock(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.model.cross_coin.CrossCoinBlock(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.cross_coin.CrossCoinBlock(peer_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.model.cross_coin.CrossSectionPool(dropout)`: `<new>` -> `0.1`
- `nig_timenet_torch.model.cross_coin.CrossSectionPool(num_heads)`: `<new>` -> `4`
- `nig_timenet_torch.model.cross_coin.CrossSectionPool(peer_dropout)`: `<new>` -> `0.0`
- `nig_timenet_torch.trainer.groups.group_batches(shuffle)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.flat_run.flat_inputs(time_channel)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.settings.SplitSettings(cross_section)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.settings.SplitSettings(top_coins)`: `<new>` -> `None`
