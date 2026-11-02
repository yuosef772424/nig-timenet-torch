# Second reference oracle — real `main.ipynb` default settings (BUILD_PLAN.md P01)

**Real `main.ipynb` default architecture/training settings: anti_memorization=True, GRU trunk,
label_smoothing=0.1 — EXCEPT price_targets, which this repo keeps at (high, low, close) by owner decision
(2026-11-02), overriding main.ipynb's own close-suspension; see BUILD_PLAN.md V1.**

Used by M6 (P33b, GRU / anti-memorization encoder variant). M2-M5 use `reference/data/` (Transformer path). Do
not blend the two.

```bash
.venv/bin/python reference/export_reference_default.py   # writes reference/data_default/*
```

## Provenance

| item | value |
|---|---|
| upstream commit | `aeafc564f25c78f252936affbd374f68aa49b8d9` (`./upstream`, read only, still clean after the run) |
| script | `reference/export_reference_default.py` = verbatim copy of `export_reference.py` except `MODEL_OVERRIDES`, `label_smoothing`, output dir |
| environment | the same venv as `reference/data/` (Python 3.12.15, numpy 2.4.6, tensorflow 2.21.0, keras 3.15.1), CPU, same seed (0), batch (8), seq_len (32), n_features (6), determinism flags |

## What differs from `reference/data/`

- `MODEL_OVERRIDES = dict(mns["ANTI_MEMORIZATION_CONFIG"])`, merged over `MODEL_CONFIG` as `workflow/run.py:267-268`
  does: {"d_model": 64, "num_layers": 2, "head_hidden": 64, "class_head_hidden": 32, "dropout": 0.25, "input_clip": 4.0, "stats_mode": "symlog", "input_noise_std": 0.1, "feature_dropout": 0.1, "linear_path_l2": 0.001, "level_passthrough": true, "level_norm": "batch", "encoder": "gru"}.
- `price_targets = merged["price_targets"]` unchanged = ['high', 'low', 'close'] (`workflow/run.py:247-249`'s close
  suspension NOT applied).
- `build_target_configs(price_targets, label_smoothing=0.1)` (`workflow/run.py:316`).
- Training-only layers (SpatialDropout1D feature dropout, GaussianNoise input noise, Dropout 0.25) are identity at
  training=False; the two `BatchNormalization` layers (`level_bn`, `stats_bn`) use their initial moving stats
  (mean 0, var 1), stored in `weights.npz` as the 4 non-trainable variables.
- The script's sub-layer step (`sublayers.*`) covers the 3 NIGHeads only: this model has no TransformerBlock.
  GRU internals are not exported.

Full constructor kwargs: `data_default/config.json`. Totals: 52 graph layers, 70 variables (66 trainable),
66 gradients (none empty); PRIMARY total loss 8.437118530273438.

## Ordered layer classes

| # | name | class |
|---|---|---|
| 0 | input_sequence | InputLayer |
| 1 | instance_norm | InstanceNorm |
| 2 | feature_dropout | SpatialDropout1D |
| 3 | level_symlog | SymLog |
| 4 | input_noise | GaussianNoise |
| 5 | level_bn | BatchNormalization |
| 6 | decomp | CausalMultiScaleDecomp |
| 7 | level_feature_dropout | SpatialDropout1D |
| 8 | trend_seasonal | Concatenate |
| 9 | level_noise | GaussianNoise |
| 10 | shape_and_level | Concatenate |
| 11 | step_proj | Dense |
| 12 | gru_norm_1 | RMSNorm |
| 13 | gru_1 | GRU |
| 14 | gru_drop_1 | Dropout |
| 15 | gru_res_1 | Add |
| 16 | gru_norm_2 | RMSNorm |
| 17 | gru_2 | GRU |
| 18 | gru_drop_2 | Dropout |
| 19 | gru_res_2 | Add |
| 20 | final_norm | RMSNorm |
| 21 | stats_bn | BatchNormalization |
| 22 | last_token | LastToken |
| 23 | attn_pool | AttentionPool |
| 24 | stats_proj | Dense |
| 25 | linear_shape_and_level | Concatenate |
| 26 | readout_concat | Concatenate |
| 27 | flatten_window | Flatten |
| 28 | readout_fc | Dense |
| 29 | linear_path | Dense |
| 30 | add_linear_path | Add |
| 31 | trunk_norm | RMSNorm |
| 32 | trunk_drop | Dropout |
| 33 | class_fc_close | Dense |
| 34 | class_fc_high | Dense |
| 35 | class_fc_low | Dense |
| 36 | nig_close | NIGHead |
| 37 | nig_high | NIGHead |
| 38 | nig_low | NIGHead |
| 39 | class_drop_close | Dropout |
| 40 | class_drop_high | Dropout |
| 41 | class_drop_low | Dropout |
| 42 | ordered_means | OrderedMeans |
| 43 | unc_close | NIGUncertainty |
| 44 | y_close_class_logits | Dense |
| 45 | conf_close | ConfidenceHead |
| 46 | unc_high | NIGUncertainty |
| 47 | y_high_class_logits | Dense |
| 48 | conf_high | ConfidenceHead |
| 49 | unc_low | NIGUncertainty |
| 50 | y_low_class_logits | Dense |
| 51 | conf_low | ConfidenceHead |

## sha256 (two consecutive runs of `export_reference_default.py` produced byte-identical files)

```
ba3844f1551631058726b1d23d9fb9b7de349d99233ca4c41d916c5c234bf54e  config.json
2d79c69d2614a58f2d811e5027280d30432043e383866d8e5e9bc70919fa3c46  grads.json
e9ff10fe96b7b4ef37194c0ca1b412332be196b0e882f7b5baa232b4679383fc  grads.npz
c51817ea76ac6584d9065d588ebad356ea2c0bf48f36b76ab1ece8a96aaee5bf  inputs.npz
7c48431f488cd25f36647e94990be7a6d0bb71c137d9bb3764d19aa113bed79d  labels.npz
d2ec6e2bf7824b38ec2a6d7f78fc2ad29ba3f5e7389825ca8110e379d312000a  layer_outputs.npz
c2fbeaa1df27cac3af20502862d2b01d321397ebf8c3102f627cfbf84e26c93d  layers.json
95c0ecf7e52c7007b0059275d8c668fac52f083b03a3225b040a7e84c95c4013  losses.json
1cdff8f9a0fb0f7df61a0decf83f1fcc08b403bd37f49b330a9b3aab5d44de6b  outputs.npz
73b7179915d7cb5539bfb8269d3bd6bcfc7214da58e3497e92049a4e943d1250  provenance.json
a5a2b9d3ffb240a0d0e35eadb199b6de8efea6a0e7282413f8fc9c323cb55c5b  sublayers.json
2fb20991faf0fc86ef14d38e8ea1773f4627aba5e722c27f7ee8e23b6e11f1a8  sublayers.npz
3e4090e04487b8386349f056ce86b249c8e72b37a9a759debce212351eb5df65  weights.json
9cddd840242797ee6dafbfd839deba6be258c3696ed26745e51752ab832ad9c2  weights.npz
```

Same sanity as `ORACLE_REVIEW.md` (two-run byte identity) only; not re-audited in full (optional future work).
