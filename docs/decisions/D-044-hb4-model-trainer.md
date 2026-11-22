# D-044 — Hardening 2026-11b, package HB4a: model and trainer clean-ups (owner 2026-11-21)

Date: 2026-11-22
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB4); owner message 2026-11-22: "continue the work, preferably with Sonnet agents for each task"
Why: the audit `docs/hardening/audit-2026-11b/A_model_trainer_stats_core.md` (rows A-04..A-06, A-10..A-15, A-21..A-24, A-26, A-27, A-33, re-verified by the agent against the code) found dead scaffolding (the Keras-port `PanelModel` used by no production code, three evidential options registered only to raise, the always-empty constraints return value, `set_determinism`, `Trainer.batch_indices`), the same helper written several times (NIG `alpha - 1` floor, Huber, the 11-argument loss call, `_device`, the "did you mean" message, the reseeding constants) and stale "P34" texts.

## What changes behaviour
- No number changes: the trainer config / run identity pin (`tests/pins/trainer_config.json`), the model assembly pin
  and the oracle suites are byte-unchanged; every helper is bit-identical (same floors, same seeds).
- Removed: `model/panel.py` (`PanelModel`, `CrossAssetBlock`, `KerasMultiHeadAttention`, its oracle tests; G8 ratchet
  1 -> 0); the evidential options `normalize_reg`, `huber_weight`, `alpha_floor_straight_through` (they raised
  NotImplementedError when set; a target naming one now fails as an unknown key) and the dead `normalize` /
  `stop_grad` arguments of `nig_regularizer`; `core.seeds.set_determinism` (an alias of `seed_everything`).
- A non-empty `constraints` list in a trainer config raises at validation (it raised at the first loss before);
  `compute_losses` returns four values (the fifth was always `{}`).
- New: `Trainer.device`, `model.nig_layers.alpha_minus_one`, `core.seeds` epoch / stage-probe / BN-refresh seed
  helpers, `core.registry.did_you_mean` / `unknown_names_text`.
- Not done here (later package or owner decision): A-02 / A-03 / A-25 (builder split), A-07 / A-20 (shared fit /
  staged machinery), A-09 (always-True model keys: they are hashed into every run identity), A-18 (which label-scale
  component stays: an owner decision).

- `nig_timenet_torch.core.registry.unknown_names_text(match)`: `<new>` -> `<class 'str'>`
- `nig_timenet_torch.core.seeds.set_determinism(seed)`: `0` -> `<removed>`
- `nig_timenet_torch.model.nig_layers.alpha_minus_one(floor)`: `<new>` -> `0.01`
- `nig_timenet_torch.model.panel.CrossAssetBlock(attention)`: `True` -> `<removed>`
- `nig_timenet_torch.model.panel.CrossAssetBlock(attn_dropout)`: `0.1` -> `<removed>`
- `nig_timenet_torch.model.panel.CrossAssetBlock(d_model)`: `64` -> `<removed>`
- `nig_timenet_torch.model.panel.CrossAssetBlock(dropout)`: `0.25` -> `<removed>`
- `nig_timenet_torch.model.panel.CrossAssetBlock(ff_mult)`: `2` -> `<removed>`
- `nig_timenet_torch.model.panel.CrossAssetBlock(num_heads)`: `4` -> `<removed>`
- `nig_timenet_torch.model.panel.KerasMultiHeadAttention(dropout)`: `0.0` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(attn_dropout)`: `0.1` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(class_head_hidden)`: `32` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(cross_attention)`: `True` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(d_model)`: `64` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(dropout)`: `0.25` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(ff_mult)`: `2` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(head_hidden)`: `64` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(n_cross_layers)`: `1` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(num_heads)`: `4` -> `<removed>`
- `nig_timenet_torch.model.panel.PanelModel(targets)`: `('high', 'low', 'close')` -> `<removed>`
- `nig_timenet_torch.trainer.tasks.nig_regularizer(normalize)`: `False` -> `<removed>`
- `nig_timenet_torch.trainer.tasks.nig_regularizer(stop_grad)`: `True` -> `<removed>`
- `registry:task:evidential.default_options.alpha_floor_straight_through`: `False` -> `<removed>`
- `registry:task:evidential.default_options.huber_weight`: `0.0` -> `<removed>`
- `registry:task:evidential.default_options.normalize_reg`: `False` -> `<removed>`
