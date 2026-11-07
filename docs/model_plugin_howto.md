# How to add a model

A model plugs into this repo by matching a boundary contract, not by sharing internals. The trainer
(`trainer/trainer.py`), `workflow/prediction_table.py` and `stats/metrics.py` only see the output dict.

1. Implement a `torch.nn.Module` with `forward(x) -> dict[str, Tensor]`. `x` has shape `(batch, seq_len, n_features)`.
   The output keys are the `HEAD_REGISTRY` names for every head you emit (`model/heads.py`):
   - `nig_regression`: `y_{t}`, `y_{t}_nu`, `y_{t}_alpha`, `y_{t}_beta`, `y_{t}_confidence`, plus
     `y_{t}_epistemic` and `y_{t}_aleatoric`.
   - `binary_classification`: `y_{t}_class_logits`. This is a probability despite the name.

   Each output is `(batch, 1)` float32. The simplest route is to reuse the head modules in `model/nig_layers.py`
   and `heads.classification_head_forward`, as `TinyMLP` does; assemble the output dict with `heads.nig_outputs` and
   `heads.class_outputs`, never from literal `y_...` strings. If you have kernel regularisers, add
   `regularization_loss()`; the trainer adds it to the loss when present.

   The health report (`eval/nig_floor.py`, the layer probe, the baseline job) reads two members of the model, so
   provide both: `head_modules() -> {target: NIGHead}` (the floors are read from them; no module-name convention) and
   `default_probe_points`, a tuple of module names for the layer probe (nig_timenet_v2: `("trunk_norm",)`; TinyMLP:
   `("fc2",)`).
2. Write a `build_fn(seq_len, n_features, **config)` and register it (all registries and their names: the Registries
   table of `docs/MAP.md`):
   ```python
   ARCHITECTURES.register("my_model", build_my_model, version=1, build_fn=build_my_model,
                          default_config={...})
   ```
3. Check it. `check_contract(my_model, contract_of(reference_model, x), x)` (from `model/contract.py`) raises
   with a field-level diff when a name, shape, dtype or head kind differs
   (`architecture_contract(ModelSettings(), seq_len, n_features)` gives the default architecture's contract). Select it
   with
   `ModelSettings(architecture="my_model")` and `build_model(settings, seq_len, n_features)`.

You are done at that point: the same `Trainer`, `predict_table`, metrics and `health_report` run unchanged. See
`tests/model/test_tiny_mlp_plugin.py` and `tests/workflow/test_health_report_architectures.py` (tiny_mlp, GRU, TCN) for
the end-to-end proof.

## How to add an encoder or a head kind to nig_timenet_v2

- Encoder: one class in `model/encoders.py` (subclass `Encoder`: `build(model, add, sfx, ...)` adds its modules with
  `add(name, module)` and returns the output width, `encode(model, tok, sfx)` is its forward) plus one
  `ENCODERS.register("name", Instance(), version=1, supports_input_skip=...)` line. The builder, the oracle weight
  loader and `build_nig_timenet_v2(encoder="name")` need no edit. Module names are checkpoint keys: keep them stable
  and extend `tests/pins/model_assembly.json` with a case (`UPDATE_MODEL_PINS=1`).
- Classification head kind: a forward `fn(h, fc, out, drop)` plus one `HEAD_REGISTRY.register(...)` line with
  `task_type="classification"` and `output_width` (1, or None for `n_classes`) in `model/heads.py`. The builder and
  TinyMLP find it by its task type; nothing else names it.
- Input-skip variant: a class in `model/parts.py` with the flags `adds_modules` / `replaces_patch_embed` and one
  `INPUT_SKIP_VARIANTS.register` line.

## How to add a probe point

`model/probe.py:layer_probe_report(model, x, y, probe_points=[...])` probes any module by its name in
`model.named_modules()`, such as `"patch_embed"`, `"block_1"` or `"trunk_norm"` (for a GRU encoder, `"step_proj"`,
`"gru_1"`). To add a probe point, give the activation you want its own named submodule and put that name in
`probe_points`; it appears as a row in the next report run. `"raw_input"` (the flattened raw window) is always the
first row. The report gives numbers only: no conclusion is drawn from them in this repo. The default points of an
architecture are its `default_probe_points`.
