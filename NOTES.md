# NOTES — nig-timenet-torch

History: the notes below were written with the TensorFlow reference oracle (`reference/`, task 1). The PyTorch port
has been built and merged since (`src/nig_timenet_torch/`, batches B1-B10 in `docs/STATUS.md`); the items under OPEN
are the questions raised then. They have NOT been re-triaged since the port (several were settled by it, see
`docs/BUILD_PLAN.md`): read them as history, not as current status.

## OPEN

1. "Default head types" is ambiguous upstream. `builder.DEFAULT_HEAD_TYPES` = NIG regression only, but the real entry
   point `build_model_fn` uses `MODEL_CONFIG["head_types"]` = NIG regression + binary classification for each target.
   The oracle uses the `MODEL_CONFIG` default (what `main.ipynb` trains). If the port must also match the NIG-only
   variant, a second export is needed.
2. Loss lambdas: the trainer's scheduled vars start at 0 (`lambda_reg`, `lambda_calib` both 0 at epoch 0), which makes
   the evidential regularizer and confidence-calibration terms vanish. `grads.npz` uses the schedule END values
   (0.05 / 0.1) so those terms (and the ConfidenceHead gradients) are covered; `losses.json` has both scenarios.
   Confirm this is the scenario the port should be validated against.
3. Losses and gradients are computed with `training=False` (dropout off, deterministic). Training-mode behaviour
   (Dropout masks, label flooding `flood_level`) is NOT in the oracle — dropout masks cannot be matched across
   frameworks anyway.
5. Kendall `log_var_*` (trainer, not model) are included in weights/grads with prefix `trainer/`. The port must decide
   whether these belong to the model or to the training loop.
6. `model.losses` (L2: NIGHead fc l2=1e-5, linear_path l2=1e-4) is part of the total loss. In Keras 3 the L2 term is
   `l2 * sum(w²)` (no 0.5 factor); the port must reproduce this exactly (value in `losses.json`
   -> `model_regularization_losses`).
7. Weight layout differences to handle in the port (not verified here): Keras Dense kernel is (in, out); Conv1D
   kernel is (kernel, in, out); `patch_proj` is a Conv1D. GELU in Keras Dense(activation="gelu") is the exact (erf)
   form (`keras.activations.gelu(x, approximate=False)` in Keras 3.15.1, checked).
8. The installed package is `tensorflow` 2.21.0 (upstream CI pins `tensorflow-cpu~=2.21.0`). Same major.minor; CPU
   numerics should be the same, but this was not cross-checked against `tensorflow-cpu`.
9. Multi-timeframe inputs, coin embeddings, two-tower / FiLM / GRU / TCN encoders, level passthrough and anti-
   memorization config are not covered (single timeframe, transformer encoder only, as requested).
10. oneDNN is on (TF default on CPU). The outputs are byte-identical between runs on this machine; another CPU may give
    last-bit differences. The port should compare with tolerances, not bitwise.

## RESOLVED

- (was OPEN 4) Internals of composite layers: `reference/data/sublayers.npz` / `sublayers.json` now hold every
  TransformerBlock step (norm1, q/k/v projections, pre-softmax scores, attention weights, attention output,
  residual 1, norm2, SwiGLU, block output) and every NIGHead step (fc, raw params, softplus, final), obtained by
  calling the upstream sub-layer objects directly; the softmax/softplus values are captured by wrapping
  `tf.nn.softmax` / `tf.nn.softplus` during the upstream call. The recomputed block and head outputs equal
  `layer_outputs.npz` byte for byte. Still not exported: the CausalMultiScaleDecomp gate and the attention
  context before `wo`.
