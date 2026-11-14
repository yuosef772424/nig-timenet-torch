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

## New architecture chart_cnn (owner 2026-11-10: "a big difference, a different structure, preferably none of the
existing layers, since they may be the cause")
- `model/chart_cnn.py`: each window drawn as a 2-channel chart image (range bars + close line, min-max per window,
  so an outlier bar cannot blow up the input), a 2D CNN, one shared head written in the module (no NIGHead /
  OrderedMeans / ConfidenceHead / instance norm); same 24-key output contract. Opt-in (`architecture="chart_cnn"`).

## New architectures conv1d_net / no_dense_net (owner 2026-11-10: "a conv1d structure, and one with no Dense layer
anywhere, to find where memorization starts and where learning breaks")
- `model/scratch_nets.py`: conv1d_net (dilated Conv1d blocks, avg + max pooling, dense head) and no_dense_net (no
  Linear and no 1x1 conv anywhere; the outputs are Conv1d(k=3) maps averaged over time). Inputs squashed with
  sign(x) log1p|x|. Both opt-in. The head maths (`chart_cnn.nig_params` / `output_dict`) is shared by the three
  scratch models.
- grad_stack writes `learning_curves.csv` and prints, per variant, the best epoch, `overfit` (val_last - val_best)
  and `gap` (val_last - train_last). GradNet pools a layer whose gradient size varies per row (TimesNet) to its probe
  size.

## Learning rate and encoder names (owner 2026-11-10: "fix the errors and make the learning rate e5 to e7")
- `GradStackConfig.lr` (default 1e-3 = TrainSettings) and `lr_min` (None = constant; a value = cosine decay from lr at
  the first epoch to lr_min at the last, `FitSettings.lr`); the notebook uses 1e-5 -> 1e-7 over 10 epochs.
- `architecture` may be an encoder name (gru / transformer / tcn / itransformer / timesnet): the default model with
  that encoder (the owner's run passed "itransformer" as an architecture and got UnknownNameError).
- chart_cnn pools with `grid_pool` (deterministic CUDA backward; AdaptiveAvgPool2d failed under
  use_deterministic_algorithms on the owner's GPU).

## reuse (owner 2026-11-11: training finished but the results were lost on a page refresh)
- `run_grad_stack(cfg, out, reuse=True)`: a variant whose folder holds a checkpoint is RESUMED instead of deleted (a
  finished run trains no epoch, restores its best weights and writes the report, curves and battery tables again).

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
- `nig_timenet_torch.workflow.grad_net.GradNet(dim)`: `<new>` -> `32`
- `nig_timenet_torch.workflow.grad_net.GradNet(outputs)`: `<new>` -> `'per_head'`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(full_grad_dim)`: `<new>` -> `32`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(full_grad_outputs)`: `<new>` -> `'per_head'`
- `nig_timenet_torch.workflow.grad_stack.fit_variant(build)`: `<new>` -> `<callable nig_timenet_torch.workflow.run.build>`
- `nig_timenet_torch.model.encoders.TimesBlock(top_k)`: `<new>` -> `2`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(model_overrides)`: `<new>` -> `{}`
- `nig_timenet_torch.model.chart_cnn.CHART_CNN_DEFAULT_CONFIG.dropout`: `<new>` -> `0.3`
- `nig_timenet_torch.model.chart_cnn.CHART_CNN_DEFAULT_CONFIG.height`: `<new>` -> `32`
- `nig_timenet_torch.model.chart_cnn.CHART_CNN_DEFAULT_CONFIG.hidden`: `<new>` -> `64`
- `nig_timenet_torch.model.chart_cnn.CHART_CNN_DEFAULT_CONFIG.n_blocks`: `<new>` -> `3`
- `nig_timenet_torch.model.chart_cnn.CHART_CNN_DEFAULT_CONFIG.out_scale`: `<new>` -> `0.01`
- `nig_timenet_torch.model.chart_cnn.CHART_CNN_DEFAULT_CONFIG.price_targets`: `<new>` -> `['high', 'low', 'close']`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(dropout)`: `<new>` -> `0.3`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(feature_cols)`: `<new>` -> `(0, 1, 2)`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(head_types)`: `<new>` -> `None`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(height)`: `<new>` -> `32`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(hidden)`: `<new>` -> `64`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(n_blocks)`: `<new>` -> `3`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(out_scale)`: `<new>` -> `0.01`
- `nig_timenet_torch.model.chart_cnn.ChartCNN(price_targets)`: `<new>` -> `('high', 'low', 'close')`
- `registry:architecture:chart_cnn.default_config.dropout`: `<new>` -> `0.3`
- `registry:architecture:chart_cnn.default_config.height`: `<new>` -> `32`
- `registry:architecture:chart_cnn.default_config.hidden`: `<new>` -> `64`
- `registry:architecture:chart_cnn.default_config.n_blocks`: `<new>` -> `3`
- `registry:architecture:chart_cnn.default_config.out_scale`: `<new>` -> `0.01`
- `registry:architecture:chart_cnn.default_config.price_targets`: `<new>` -> `['high', 'low', 'close']`
- `nig_timenet_torch.model.scratch_nets.CONV1D_NET_DEFAULT_CONFIG.channels`: `<new>` -> `64`
- `nig_timenet_torch.model.scratch_nets.CONV1D_NET_DEFAULT_CONFIG.dropout`: `<new>` -> `0.2`
- `nig_timenet_torch.model.scratch_nets.CONV1D_NET_DEFAULT_CONFIG.hidden`: `<new>` -> `64`
- `nig_timenet_torch.model.scratch_nets.CONV1D_NET_DEFAULT_CONFIG.n_blocks`: `<new>` -> `3`
- `nig_timenet_torch.model.scratch_nets.CONV1D_NET_DEFAULT_CONFIG.out_scale`: `<new>` -> `0.01`
- `nig_timenet_torch.model.scratch_nets.CONV1D_NET_DEFAULT_CONFIG.price_targets`: `<new>` -> `['high', 'low', 'close']`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(channels)`: `<new>` -> `64`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(dropout)`: `<new>` -> `0.2`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(head_types)`: `<new>` -> `None`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(hidden)`: `<new>` -> `64`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(n_blocks)`: `<new>` -> `3`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(out_scale)`: `<new>` -> `0.01`
- `nig_timenet_torch.model.scratch_nets.Conv1DNet(price_targets)`: `<new>` -> `('high', 'low', 'close')`
- `nig_timenet_torch.model.scratch_nets.NO_DENSE_NET_DEFAULT_CONFIG.channels`: `<new>` -> `64`
- `nig_timenet_torch.model.scratch_nets.NO_DENSE_NET_DEFAULT_CONFIG.n_blocks`: `<new>` -> `4`
- `nig_timenet_torch.model.scratch_nets.NO_DENSE_NET_DEFAULT_CONFIG.out_scale`: `<new>` -> `0.01`
- `nig_timenet_torch.model.scratch_nets.NO_DENSE_NET_DEFAULT_CONFIG.price_targets`: `<new>` -> `['high', 'low', 'close']`
- `nig_timenet_torch.model.scratch_nets.NoDenseNet(channels)`: `<new>` -> `64`
- `nig_timenet_torch.model.scratch_nets.NoDenseNet(head_types)`: `<new>` -> `None`
- `nig_timenet_torch.model.scratch_nets.NoDenseNet(n_blocks)`: `<new>` -> `4`
- `nig_timenet_torch.model.scratch_nets.NoDenseNet(out_scale)`: `<new>` -> `0.01`
- `nig_timenet_torch.model.scratch_nets.NoDenseNet(price_targets)`: `<new>` -> `('high', 'low', 'close')`
- `registry:architecture:conv1d_net.default_config.channels`: `<new>` -> `64`
- `registry:architecture:conv1d_net.default_config.dropout`: `<new>` -> `0.2`
- `registry:architecture:conv1d_net.default_config.hidden`: `<new>` -> `64`
- `registry:architecture:conv1d_net.default_config.n_blocks`: `<new>` -> `3`
- `registry:architecture:conv1d_net.default_config.out_scale`: `<new>` -> `0.01`
- `registry:architecture:conv1d_net.default_config.price_targets`: `<new>` -> `['high', 'low', 'close']`
- `registry:architecture:no_dense_net.default_config.channels`: `<new>` -> `64`
- `registry:architecture:no_dense_net.default_config.n_blocks`: `<new>` -> `4`
- `registry:architecture:no_dense_net.default_config.out_scale`: `<new>` -> `0.01`
- `registry:architecture:no_dense_net.default_config.price_targets`: `<new>` -> `['high', 'low', 'close']`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(lr)`: `<new>` -> `0.001`
- `nig_timenet_torch.workflow.grad_stack.GradStackConfig(lr_min)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.grad_stack.fit_variant(lr)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.grad_stack_report.grad_stack_battery(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.grad_stack_report.grad_stack_battery(show)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.grad_stack.fit_variant(reuse)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.grad_stack.run_grad_stack(reuse)`: `<new>` -> `False`
