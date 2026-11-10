# D-023: grad_stack experiment (model B trained on model A's per-layer gradients)

Date: 2026-11-10
Owner decision: owner message of 2026-11-10 (Arabic, translated): "build a model trained on half of the training data
(time order) that outputs its predictions and the gradients of every learning layer; then train a second model on the
other half with the first model's gradients as input (gradients only, and gradients + data); then apply the project's
test" and "add it as a function in the jobs notebook".
Why: a quick experiment outside the roadmap; results in `experiments/grad_stack/RESULTS.md`.

## What changes behaviour
- Nothing in existing code paths. New module `workflow/grad_stack.py`, job entry `jobs/grad_stack.py`, and section
  `2b. Grad stack` of `notebooks/colab_nig_timenet_torch.ipynb` (runs on Run all while `RUN_GRAD_STACK = True`).

## New encoders (owner request 2026-11-10: "apply iTransformer and TimesNet to these experiments")
- `itransformer` (`model/encoders.py` ITransformerEncoder): one token per input channel (Linear over the whole window),
  attention across channels.
- `timesnet` (TimesNetEncoder / TimesBlock): top-2 FFT periods chosen PER ROW (the paper picks them from the batch
  mean, which would make a row's output depend on its batch), 2D inception convs (kernels 1 / 3) per period.
- Both are opt-in (`encoder` override); the default model (encoder "gru") is unchanged.
- `GradStackConfig.model_overrides` passes e.g. `{"encoder": "timesnet"}` to every variant of the experiment.

## New defaults
- `nig_timenet_torch.jobs.grad_stack.CONFIG`: `<new>` -> `<class 'nig_timenet_torch.workflow.grad_stack.GradStackConfig'>`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(architecture)`: `<new>` -> `'tiny_mlp'`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(batch_size)`: `<new>` -> `256`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(data)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(e1)`: `<new>` -> `{'n_bars': 2000, 'n_coins': 32}`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(epochs)`: `<new>` -> `5`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(grad_target)`: `<new>` -> `'output'`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(max_rows)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(seed)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(variants)`: `<new>` -> `('A', 'data_h2', 'grad', 'grad_data', 'grad_full', 'full')`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(with_preds)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.grad_stack.grad_stack_battery(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.grad_stack.grad_stack_battery(show)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.grad_net.GradNet(dim)`: `<new>` -> `32`
- `nig_timenet_torch.workflow.grad_net.GradNet(outputs)`: `<new>` -> `'per_head'`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(full_grad_dim)`: `<new>` -> `32`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(full_grad_outputs)`: `<new>` -> `'per_head'`
- `nig_timenet_torch.workflow.grad_stack.fit_variant(build)`: `<new>` -> `<callable nig_timenet_torch.workflow.run.build>`
- `nig_timenet_torch.model.encoders.TimesBlock(top_k)`: `<new>` -> `2`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(model_overrides)`: `<new>` -> `{}`
