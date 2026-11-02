# Reference oracle — NIG-TimeNet v2 (TensorFlow)

Frozen numerical reference produced by the ORIGINAL TensorFlow code, for checking the future PyTorch port.
No PyTorch code here. Regenerate with:

```bash
.venv/bin/python reference/export_reference.py      # writes reference/data/*
```

## Provenance

| item | value |
|---|---|
| upstream repo | https://github.com/yuosef772424/crypto-signal-prediction (cloned to `./upstream`, read only) |
| upstream commit | `aeafc564f25c78f252936affbd374f68aa49b8d9` |
| Python | 3.12.15 (venv `.venv`, created with `uv venv -p python3.12`) |
| numpy | 2.4.6 (same exact pin as upstream `requirements-ci.txt`) |
| tensorflow | 2.21.0 (upstream pins `tensorflow-cpu~=2.21.0`; installed the `tensorflow` wheel, same 2.21) |
| keras | 3.15.1 (pulled in by tensorflow 2.21; not installed separately) |
| pytest | 9.1.1 (upstream `~=9.1.1`) |
| device | CPU only, `enable_op_determinism()`, 1 intra/inter-op thread, seeds python/numpy/tf = 0 |

## Model config

Built through the upstream entry point, exactly as `main.ipynb` does:
`model.build_model_fn(32, 6, config={"d_model": 16, "num_layers": 2})`, i.e.
`build_nig_timenet_v2(seq_len=32, n_features=6, **{**MODEL_CONFIG, d_model=16, num_layers=2})`.
One timeframe (scalar seq_len -> single input `input_sequence`, shape `(B, 32, 6)`), batch 8.

Non-default: `d_model=16`, `num_layers=2`. Everything else = upstream `MODEL_CONFIG`:
num_heads=4, num_kv_heads=2, patch_len=4, stride=2 (-> 15 tokens, pad 0), kernel_sizes=(3,5,9,17), max_rel_pos=16,
dropout=0.1, attn_dropout=0.0, causal=False, window=None, instance norm / decomposition / linear path on,
stats_mode="full", norm_eps=1e-4, input_clip=None, linear_path_l2=1e-4, price_targets=(high, low, close),
head_types = `nig_regression` + `binary_classification` for every target (the MODEL_CONFIG default), enforce_order=True,
head_hidden=128, class_head_hidden=64, nu_min=0.1, alpha_min=2.0, beta_min=0.01, unc_max=20.0.
Keys not in MODEL_CONFIG take the builder defaults: encoder="transformer", architecture="fused", level_film=False,
n_coins=0. Full kwargs: `data/config.json` -> `constructor_kwargs`.

## Ordered layer classes (functional graph, `model.layers` order)

| # | name | class |
|---|---|---|
| 0 | input_sequence | InputLayer |
| 1 | instance_norm | InstanceNorm (2 outputs: z, stats) |
| 2 | decomp | CausalMultiScaleDecomp (2 outputs: trend, seasonal) |
| 3 | trend_seasonal | Concatenate |
| 4 | patch_embed | PatchEmbedding |
| 5 | block_1 | TransformerBlock |
| 6 | block_2 | TransformerBlock |
| 7 | final_norm | RMSNorm |
| 8 | last_token | LastToken |
| 9 | attn_pool | AttentionPool |
| 10 | stats_proj | Dense (gelu) |
| 11 | readout_concat | Concatenate |
| 12 | flatten_window | Flatten |
| 13 | readout_fc | Dense (gelu) |
| 14 | linear_path | Dense (L2 1e-4) |
| 15 | add_linear_path | Add |
| 16 | trunk_norm | RMSNorm |
| 17 | trunk_drop | Dropout |
| 18–20 | class_fc_close / _high / _low | Dense (relu) |
| 21–23 | nig_close / nig_high / nig_low | NIGHead (4 outputs: mu_raw, nu, alpha, beta) |
| 24–26 | class_drop_close / _high / _low | Dropout |
| 27 | ordered_means | OrderedMeans (inputs [close, high, low] mu_raw; outputs mu_high, mu_close, mu_low) |
| 28 | unc_close | NIGUncertainty (outputs epistemic, aleatoric) |
| 29 | y_close_class_logits | Dense (sigmoid) |
| 30 | conf_close | ConfidenceHead |
| 31 | unc_high | NIGUncertainty |
| 32 | y_high_class_logits | Dense (sigmoid) |
| 33 | conf_high | ConfidenceHead |
| 34 | unc_low | NIGUncertainty |
| 35 | y_low_class_logits | Dense (sigmoid) |
| 36 | conf_low | ConfidenceHead |

## Files (`reference/data/`)

| file | content |
|---|---|
| `config.json` | seed, shapes, overrides, full constructor kwargs, trainer targets/loss config |
| `inputs.npz` | `input_sequence` float32 (8, 32, 6): per-sample level + scale × random walk (numpy `default_rng(0)`) |
| `labels.npz` | `y_{t}_reg` float32 (8,1) ~ N(0, 0.5²); `y_{t}_class` float32 (8,1) in {0,1} (feed-time form), t ∈ high/low/close |
| `weights.npz` / `weights.json` | 72 model variables (all trainable) keyed by Keras `variable.path` + 3 trainer Kendall `log_var_*` (prefix `trainer/`). JSON: name, shape, dtype, trainable, `attr_path` (stable attribute path, e.g. `block_1.attn.wq.kernel`; Keras sublayer names like `dense_14` are auto-numbered), owning top-level layer and its class |
| `layer_outputs.npz` / `layers.json` | output of every graph layer, training=False; key `{index:03d}_{name}` (+ `__{k}` for multi-output layers). JSON: index, name, class, input layer names (`name[k]` = output k of a multi-output layer), output keys and shapes |
| `outputs.npz` | final model output dict (24 keys), training=False |
| `losses.json` | per-task losses, stats, total/raw loss (value + float32 hex) for two lambda-schedule scenarios (see below) |
| `grads.npz` / `grads.json` | d(total loss)/d(var) for every trainable variable (model + trainer Kendall log_vars), PRIMARY scenario; no gradient is None |
| `provenance.json` | upstream SHA and library versions |

## Losses

Computed by the upstream trainer itself: `trainer.GenericTrainer(model, trainer.build_config(cfg))._compute_losses(x, y,
None, training=False)` under a `tf.GradientTape`, where `cfg` = what `workflow.run.make_training_config` builds:
targets from `workflow.build_target_configs(price_targets)` (evidential `y_{t}_reg` with calibration loss +
classification `y_{t}_class`, label_smoothing 0), Kendall weighting on (classification tasks only, evidential excluded),
lambda schedules from `workflow.TrainSettings` defaults (`lambda_reg` 0→0.05, `lambda_calib` 0→0.1), no constraints.
Total loss = Kendall(class tasks) + Σ evidential tasks + model L2 regularization losses (`model.losses`).

- PRIMARY scenario (`grads.npz`): scheduled vars set to their `end` values (lambda_reg=0.05, lambda_calib=0.1), so the
  evidential regularizer and the confidence calibration loss (and the ConfidenceHead gradients) are exercised.
- Second scenario: `start` values (both 0, the value at epoch 0) — losses only.

## Upstream functions not called directly

- `workflow.run.make_training_config` was not called (it needs a RunSettings/plan/info/val split, run_dir and computes
  class baselines from a validation set). Instead the script calls the pieces it uses — `build_target_configs`,
  `TrainSettings()` defaults, `trainer.build_config` — with the same structure (run/callbacks parts omitted; they do not
  affect the loss).
- The loss is taken from the private `GenericTrainer._compute_losses` (the body of `train_step`/`test_step`) because
  `train_step` also applies an optimizer update. Gradients are taken with the same tape pattern as `train_step`, over
  `GenericTrainer.trainable_variables`.
- `model/diagnostics.py` and `model/layer_report.py` are excluded from `model.load_into` (they import pandas, which is
  not in the allowed install list; the builder does not use them); `model/selftests.py`, `trainer/smoke_test.py` and
  `trainer/kfold.py` are excluded as in `docs/research/audit/_nbload.py`.
- `np.savez` was replaced by an equivalent writer with fixed zip timestamps (np.savez stamps the current time, so
  files would never be byte-identical). Load with plain `np.load`.

## sha256 (two consecutive runs produced byte-identical files)

```
00956c430dc3d19642a9c4a5629b3d8342aa831ac6cd6171fbe941c21f436e96  config.json
8782f1e20416dd660c51b20e5e17a5c9f8130bbe684e54eda4a51f0633eb96ba  grads.json
613ee4ae7ae9410e67e49e42d4525a4c74651e2ac4eb2c603d9670dd12005c04  grads.npz
c51817ea76ac6584d9065d588ebad356ea2c0bf48f36b76ab1ece8a96aaee5bf  inputs.npz
7c48431f488cd25f36647e94990be7a6d0bb71c137d9bb3764d19aa113bed79d  labels.npz
6f8ce201e7c490211871bb9080bc91f9d5a1e9d807ea6f3ffc245919e3ba8716  layer_outputs.npz
7e3a966d2242a7ba66412edca0b6f00fd1b0cffdaf5943867c51526c1f2ef6bd  layers.json
f0d0b90041f0ac7ceb38256f4c852b083080bafe9df87f824ae6a072b3ecde01  losses.json
ad54ec4f72a59ba141145eafbaadaf2fdf50bd71196c09afbf0e55374c8d9ad9  outputs.npz
73b7179915d7cb5539bfb8269d3bd6bcfc7214da58e3497e92049a4e943d1250  provenance.json
11c078b627197f20eceb8a304af2e79793012a1f0355e4872566cab990e143f5  weights.json
5856d27bc61379abe1fa8afaa2cfa6301a8f61d63a2a7d0bc757d073dc2b9939  weights.npz
```

## Sub-layer internals (`sublayers.npz` / `sublayers.json`)

Added after the first export (step 8 of the script, run after everything else; the 12 files above are unchanged,
sha256 identical). For every `TransformerBlock` (block_1, block_2) and every `NIGHead` (nig_close, nig_high, nig_low),
the upstream sub-layer objects of the already-built model are called directly, training=False, on the exact tensor each
receives inside the parent's `call` (the parent input is taken from `layer_outputs.npz`). Key = `{parent}/{step}`;
the JSON gives block, step, shape, dtype and how each value was obtained.

- TransformerBlock steps: `00_block_input`, `01_norm1_out` (`norm1`), `02/03/04_attn_q/k/v_proj` (`attn.wq/wk/wv`,
  before reshape and GQA repeat: q (8,15,16), k/v (8,15,8)), `05_attn_scores_pre_softmax` and `06_attn_weights`
  ((8,4,15,15), the input and output of the single `tf.nn.softmax` call, captured by temporarily wrapping `tf.nn.softmax`
  while calling `block.attn`; scores include q·kᵀ/√dh + relative bias; no mask is applied because causal=False and
  window=None), `07_attn_out` (`attn`, after `wo`), `08_residual1` (x + `drop1`), `09_norm2_out` (`norm2`),
  `10_swiglu_out` (`ffn`), `11_block_output` (residual1 + `drop2`).
- NIGHead steps: `00_head_input` (trunk_drop output), `01_fc_out` (`fc`, swish, 128), `02_raw_params`
  (`out(drop(fc_out))`, columns mu, nu, alpha, beta, before softplus/floor) and `03_raw_*` (its columns),
  `04_softplus_nu/alpha/beta` (outputs of the 3 `tf.nn.softplus` calls captured the same way), `05_final_*`
  (`head(h)` outputs = softplus + nu_min/alpha_min/beta_min; `05_final_mu` is the raw mu BEFORE `OrderedMeans`).
- The two residual additions are the only arithmetic done in the script (the `+` of `TransformerBlock.call`; there
  is no sub-layer object for them).
- Checks (the script raises if any fails, compared byte for byte): recomputed `11_block_output` == `layer_outputs`
  entry of the block, `block(x, training=False)` == the same entry, every `05_final_*` == the NIGHead entries in
  `layer_outputs`, softplus inputs == `raw_params` columns. All passed.

sha256 (two consecutive runs byte-identical):

```
b0f0158a660a6b820ebc6c468d328194cb80792e4f8406412f979407ad808808  sublayers.json
a8a9ed4ba2eb61ed9399486bf106e60885cb2a47c36bb6053cee691fa156a0ac  sublayers.npz
```
