# grad_stack — model B learns from model A's per-layer gradients (quick experiment, owner 2026-11-10)

Outside the roadmap and the repo rules (owner asked for the fastest path). One script: `grad_stack.py`.

1. TRAIN rows split by time: H1 (earlier half, rows whose label crosses into H2 purged) and H2 (later half).
2. Model A trained on H1. On H2 / val / test it outputs its predictions and, per row, the gradient at every
   learning layer (gradient w.r.t. the layer output; plus the input gradient), summarised as norm/mean/std/max|g|.
   Saved to `<out>/modelA_grads_preds.npz`.
3. Model B trained on H2 with those gradients as input: `grad` (gradients only) or `grad_data` (gradients + the
   window). Controls: `data_h2` (same rows, data only) and `full` (all train rows, data only).
4. Every variant is scored with the project's own test: `run.stage_report` (target_report on test) and the gate rows
   `eval.gates.stage_rows` (on synthetic E1 judged against the oracle).

`--grad-target output` (default) = gradient of model A's outputs: label-free, usable live.
`--grad-target loss` = gradient of model A's loss: uses the row's label, so on val/test it LEAKS the target
(upper bound only).

```bash
PYTHONPATH=src python experiments/grad_stack/grad_stack.py --out /tmp/gs                 # synthetic E1, tiny_mlp
PYTHONPATH=src python experiments/grad_stack/grad_stack.py --out /tmp/gs --arch ""       # project default model
PYTHONPATH=src python experiments/grad_stack/grad_stack.py --out /tmp/gs --data <artifact dir> --device auto
```
Outputs: `test_report_by_variant.csv`, `gate_rows.csv`, `gate_verdicts.csv`, `summary.json`, `modelA_grads_preds.npz`.
