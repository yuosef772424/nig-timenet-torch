# NOTES — nig-timenet-torch

History: these notes were written with the TensorFlow reference oracle (`reference/`, task 1). The PyTorch port was built
and merged afterwards (batches B1-B10, `docs/BUILD_PLAN.md`). Re-triaged in H6 (2026-11-07, audit finding HF-36): each
item below is RESOLVED or still OPEN with its bottleneck id.

## OPEN

1. "Default head types": the oracle uses `MODEL_CONFIG["head_types"]` (NIG regression + binary classification); the
   NIG-only variant would need a second export. Settled for the port (the registry default config is THE default,
   H3 card PITFALLS); a second export is only needed if someone asks for NIG-only parity (no bottleneck row).
9. The oracle covers single timeframe, transformer encoder only. The other encoders (GRU, TCN, two-tower, FiLM), coin
   embeddings, multi-timeframe inputs and level passthrough are ported (`model/encoders.py`) and tested for shapes and
   the pinned state dict (`tests/pins/model_assembly.json`), but NOT against TF numerically. Open as a risk, not a task.
10. oneDNN / CPU last-bit differences: the port compares with tolerances (done); no action. Kept as a note.

## RESOLVED

- (was OPEN 2) Loss lambdas: the port reproduces both scenarios (schedule start and end values) in the oracle tests
  (`test_total_loss_matches_oracle_primary_scenario` / `..._start_scenario`); the run uses `LAMBDA_SCHEDULES` in
  `workflow/flat_run.py`.
- (was OPEN 3) Losses/gradients with `training=False`: the parity tests use eval mode on purpose; training-mode
  behaviour (dropout, flooding) is covered by unit tests, not by the oracle. Accepted.
- (was OPEN 5) Kendall `log_var_*` are trainer state (`trainer/task_weighting.py`), not model weights; a registered
  weighting component is the later item B-036.
- (was OPEN 6) Model L2: `regularization_loss()` in `model/builder.py` reproduces `l2 * sum(w^2)` (no 0.5 factor); proven
  by the oracle suite.
- (was OPEN 7) Weight layout and GELU: the converter and the exact-erf GELU are tested (B-batches); no open item.
- (was OPEN 8) `tensorflow` vs `tensorflow-cpu`: the oracle is frozen and read-only; no action.
- (was OPEN 4) Internals of composite layers: `reference/data/sublayers.npz` / `sublayers.json` now hold every
  TransformerBlock step (norm1, q/k/v projections, pre-softmax scores, attention weights, attention output,
  residual 1, norm2, SwiGLU, block output) and every NIGHead step (fc, raw params, softplus, final), obtained by
  calling the upstream sub-layer objects directly; the softmax/softplus values are captured by wrapping
  `tf.nn.softmax` / `tf.nn.softplus` during the upstream call. The recomputed block and head outputs equal
  `layer_outputs.npz` byte for byte. Still not exported: the CausalMultiScaleDecomp gate and the attention
  context before `wo`.
