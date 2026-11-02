# Reference oracle: model variants (B9)

Small TF oracles for the variants ported in batch B9 (`docs/BUILD_PLAN.md` section 14, P40-P43). All values come from
the UPSTREAM code (`./upstream`, read only, commit `aeafc564f25c78f252936affbd374f68aa49b8d9`). They are loaded
through the loaders in `reference/export_reference.py`; its helpers are imported, not copied. Same environment as
`README.md` (`.venv`, TF 2.21.0, keras 3.15.1, CPU, op determinism, seeds 0). Regenerate with:

```bash
.venv/bin/python reference/export_reference_variants.py      # writes reference/data_variants/<case>/*
```

| case | built with | checks |
|---|---|---|
| `tcn` | `build_model_fn(32, 6, config={d_model: 16, num_layers: 2, encoder: "tcn"})` | 4 causal dilated Conv1D blocks, 24 outputs |
| `multitf` | `build_model_fn({"1d": 32, "4h": 24}, {"1d": 6, "4h": 5}, config={d_model: 16, num_layers: 2})` | two suffixed branches + `branch_concat`, 24 outputs |
| `multiclass` | as `tcn` but transformer, close head `multiclass_classification`, `n_classes=4` | softmax head + upstream `classification_task_loss(binary=False)`, ls 0 / 0.1, weighted / unweighted (`losses.json`) |
| `panel`, `panel_no_attention` | `cross_asset/model.py` `build_panel_model(extract_encoder(build_model_fn(32, 6, {d_model: 16, num_layers: 2})), ...)`, `cross_attention` True / False | 10 rows over 4 days, `logit` / `mu` (10, 3) |

Each case directory contains:

- `config.json`, with the full `constructor_kwargs`, i.e. `MODEL_CONFIG` with the config merged over it.
- `weights.npz`, keyed by attr path:
  - NIG models use the same attr paths as `data/weights.json`.
  - For the panel cases, the encoder's weights are `encoder.<attr path>` and the panel's own weights use the
    Keras variable `.path`.
- `inputs.npz` and `outputs.npz` (computed with `training=False`).

The parity tests are in `tests/model/test_variants_vs_oracle.py`. Weights must match exactly; outputs and losses
must match within the section 0 tolerances.

## sha256 (two consecutive runs produced byte-identical files)

```
e40ac635d5e57eebe0fe494c73b4c198ac492c4d07a1cd145205c0b3efd1f353  data_variants/multiclass/config.json
2f1a1b4630176be57e008ffcdd6939f49da05fb8631de19f4e1186fe6eccb66f  data_variants/multiclass/inputs.npz
eaf2e2f8fd77e96b8f40f123ca0dcab52d16d2f71c29abed8a714b4c6bfbfc6d  data_variants/multiclass/losses.json
b38f6dba70445ea40161b5b79109cd6618d64976799bcb9015a98fee9650e89c  data_variants/multiclass/outputs.npz
34dfbc38f34d667ad8e1b45691191951eaace9f24eb0c533e07af48799a3ad77  data_variants/multiclass/weights.npz
ae54014422cea5e8883cd5ecea194430bc489c04712df94c666323348d44a3af  data_variants/multitf/config.json
0eccad8e0e8935ac1b3c8f8d5a82d834e5acd24b2a6ebc989a3a0894ac660618  data_variants/multitf/inputs.npz
a48f57b26160b0d31f9a8b6facc19bb00b0d9f36efda5896e817b687e12105b2  data_variants/multitf/outputs.npz
3dfc1bc40cc854e123dd0629e8925d580f3a43dba637f0a0e22245397e225b94  data_variants/multitf/weights.npz
94f21528543ac7f5afd55305ffb98b7a7409432635fac9515e8c36e84d6807ed  data_variants/panel/config.json
c9d5e126373ee672e084a0a3af7fb1742774eb8fa7a5f1b4ced6d9310b7cb136  data_variants/panel/inputs.npz
d6f62306e8498515873c9e000c9277d7b832c8ed602411bcc6025c4f23892db4  data_variants/panel/outputs.npz
5972d1a327aedb11959657c5b3aa4a6810431bc6064cbe7e18aaa857c138cf99  data_variants/panel/weights.npz
601aedcdfaba83b334e0cf7ffc15007b53c5353564d33ba2230d4ae4c9cc6fe5  data_variants/panel_no_attention/config.json
53b8ce26f6852683879d0601adecb75cd6864424fb62d35943d835fa546209cf  data_variants/panel_no_attention/inputs.npz
90ad775d8329eb49e86142f4e3034d8dddb16f0d56800b756887d8db54036d7a  data_variants/panel_no_attention/outputs.npz
9fa7c7563174a902b142e5e98716629552bca23b8da566e747c6bcf9307868b3  data_variants/panel_no_attention/weights.npz
699b69432475457a3f198ee451a4503f2f791ac98b94c02532736bea1a4ebe1c  data_variants/tcn/config.json
d2d294cdbf1465c2eea371ca39d134172258a99d9a05fc3dd5e571906ed159a4  data_variants/tcn/inputs.npz
d8b4b2a0617f688205d7201cec80e0bfac4fed6f2f01442864f2563309dcedf4  data_variants/tcn/outputs.npz
861fefc558cb2ce5b81f1ea42ecc6ab1cac351b8f6d06ef5ac6a6a597fba2b72  data_variants/tcn/weights.npz
```
