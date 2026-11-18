# Alpha / beta on their floors, and training speed (owner task 2026-11-18)

Owner question (Arabic, 2026-11-18): take some models from Google Drive, check the alpha and beta outputs, and find out
why alpha and beta sit on their lower bounds in 100 % of the samples, or what the exact share is. Second part: find
ways to make training faster, try settings, and design a model version that trains fast.

Everything below is exploratory (rule C5). The real-data numbers come from a PUBLIC Binance stand-in
(`data_io/vision_slice.py`: 59 liquid USDT perpetuals, 4h, 2025-06 .. 2026-09, 170,274 windows), not from the owner's
artifact `c572b1b36960`, which cannot be downloaded here (1.16 GB; the Drive tool stops at 10 MB). The trained weights
ARE the owner's: `best.pt` of `nig_timenet_runs/notebook-main2` (commit `94b1167`, T4, 2 epochs, default model;
label scale 0.03618) and of its rolling fold 1 seed 0. The model is evaluated on the stand-in's VAL / TEST rows
(2026-06-26 .. 2026-09-30), which come after the model's own train end (2026-02-24).

## 1. The exact shares

| model (Drive) | head | alpha: share within 5 % of floor 2.0 | alpha - 2 < 0.01 | alpha median | beta: share the report printed | beta: share within 5 % of the TRUE floor | beta median / true floor |
|---|---|---|---|---|---|---|---|
| notebook-main2, own health report (100k VAL rows of the owner's artifact) | high | 0.9998 | n/a | 2.0021 | 1.00 | 0 (min beta 2.39e-5 = 1.8 x the true floor 1.31e-5) | 21.6 |
| | low | 0.9999 | n/a | 2.0013 | 1.00 | 0 (min 2.2 x) | 22.5 |
| | close | 0.9959 | n/a | 2.0272 | 1.00 | 0 (min 3.6 x) | 41.1 |
| notebook-main2 on the stand-in, VAL (15,045 rows) | high | 0.998 | 0.983 | 2.0017 | 1.00 | 0.000 | 11.7 |
| | low | 1.000 | 0.999 | 2.0006 | 1.00 | 0.000 | 12.1 |
| | close | 0.987 | 0.005 | 2.0256 | 1.00 | 0.000 | 22.3 |
| notebook-main2 on the stand-in, TEST (15,163 rows) | high | 1.000 | 0.981 | 2.0020 | 1.00 | 0.000 | 21.4 |
| | low | 1.000 | 0.968 | 2.0011 | 1.00 | 0.000 | 21.0 |
| | close | 0.992 | 0.002 | 2.0269 | 1.00 | 0.000 | 39.2 |

Answer: **beta is not on its floor at all (0 %)**: the "100 %" was a measurement bug. **Alpha really is on its floor**:
99.6-100 % of rows by the report's rule (within 5 %), 97-99.9 % of the high / low rows within 0.01; the close head sits
a little above it (median 2.026, < 1 % of rows within 0.01).

## 2. Beta: the floor-hit report compared two different units (fixed in this PR)

`eval/nig_floor.py::nig_floors` returned `beta_min` = 0.01 as the floor. Since FX2 (`nig_scale_fit: train_std`, the run
default) the head outputs beta = s^2 (softplus + beta_min) in LABEL units, with s the fitted label scale (0.0362 in
notebook-main2), so the real floor is 0.01 x 0.0362^2 = 1.31e-5. Every beta (median 2.8e-4) is below 0.01, so the old
rule `beta - 0.01 <= 0.05 x 0.01` was true for every row. A target normalisation (D-033) multiplies the floor again by
std_h^2. Fix: the floor is `beta_min x s^2 x std_h^2`; guard test `tests/eval/test_nig_floor.py::
test_floors_are_in_the_units_of_the_outputs` checks, for every scale path, that `nig_floors` equals the model's own
output with softplus driven to 0 (4 of its 5 cases fail on the old code).

The older Drive reports show both regimes:

| run (Drive health report) | label scale | alpha median (high / low / close) | beta median (high / low / close) | reading |
|---|---|---|---|---|
| baseline-001-s0-r2 (2026-11-07) | none | 24.9 / 22.6 / 14.7 | 0.0100 / 0.0100 / 0.0101 | beta REALLY on its floor 0.01 (B-052); alpha carries the scale |
| notebook-main (2026-11-09) | none | 25.0 / 22.8 / 15.2 | 0.0100 / 0.0100 / 0.0100 | same |
| fx1-real-002 / -004 (2026-11-10) | none | 23.7 / 23.6 / 14.7 | 0.0100 / 0.0100 / 0.0101 | same |
| fx1-real-001 (2026-11-10) | nig_scale | 2.002 / 2.001 / 2.001 | 1.2e-4 / 1.5e-4 / 3.3e-4 | beta free (report said 1.00: same unit bug), alpha on its floor |
| notebook-main2 (2026-11-18) | train_std 0.0362 | 2.002 / 2.001 / 2.027 | 2.8e-4 / 2.9e-4 / 5.4e-4 | same |

So in every run ONE of the two parameters is dead: before the label scale, beta on its floor and alpha carrying the
scale; after it, the reverse. The predictive variance is beta (1 + nu) / (nu (alpha - 1)); alpha's only role that beta
cannot take over is the tail (Student-t df = 2 alpha).

## 3. Alpha: why it falls to 2.0 and stays there

Measured (scripts in `scripts/`, outputs in `alpha_beta_notebook_main2_on_vision.json`):

1. **Known answer (E1, light tails):** the default model trained from scratch on E1 raises alpha: close 2.59 -> 5.97,
   high / low 2.71 -> 3.16 / 3.27 after 800 steps; no row within 0.01 of the floor. The head and the loss CAN move alpha
   up. It is not a code bug.
2. **Real candles (stand-in), same model, same steps:** alpha falls steadily: high 2.69 -> 2.12, low 2.71 -> 2.14,
   close 2.61 -> 2.48 after 800 steps; the softplus slope at the median row falls 0.50 -> 0.11. In the trained Drive
   model alpha - 2 = 0.0017, i.e. raw alpha about -6.4 and a softplus slope about 0.0017.
3. **The tails of the data are about where the floor is:** a Student-t fitted to the model's standardised residuals
   (y - mu) / scale has df 3.8 / 7.3 / 5.0 (VAL high / low / close) and 2.3 / 4.9 / 3.5 (TEST); excess kurtosis of the
   residuals 10 to 500. alpha = df / 2 would be 1.1-3.7, around the floor 2.0 (df 4). Early in training, when mu and
   the scale are still poor, many residuals are large against the scale and the NIG NLL lowers alpha (heavier tails).
4. **Then the parameter dies:** at alpha - 2 ~ 0.002 the gradient reaching the raw output is multiplied by
   sigmoid(raw) ~ 0.002. In the trained model the NLL now wants alpha UP on average (mean dNLL/dalpha = -0.015 to
   -0.096 at the outputs, 72-80 % of rows ask for a larger alpha; the predicted scale is 7-21 % too wide), but the
   step it can take is ~500 x smaller than for beta. Beta then carries all of the scale.
5. **The evidence regulariser plays no part:** lambda |y - mu| (2 nu + alpha) gives d/dalpha = lambda |err| ~ 1e-4,
   against an NLL gradient of 0.01-0.1. It is in label units (errors ~ 0.03) and so ~1000 x weaker than in TF
   (labels x 100): a second unit-ful constant, noted, not the cause.

**One named cause:** heavy-tailed residuals push alpha to its floor in the first epochs, and the softplus + floor
parametrisation then freezes it there (vanishing gradient), so the model cannot raise it later when the NLL asks for it.

What it costs: with df pinned at 4 the 80 % intervals are too wide where the tails are lighter (VAL coverage 0.834 /
0.875 / 0.818 against 0.80, health report), and alpha carries no per-row tail information, so the uncertainty is
one number (beta) per row.

## 4. Next hypotheses (each with the measurement that would refute it)

| # | hypothesis | cheapest test | refuted if |
|---|---|---|---|
| A1 | a lower alpha floor (alpha_min 1.05, df 2.1) lets alpha settle at the data's df instead of hugging 2 | E1 + stand-in, 800 steps, 2 seeds: share within 0.01 of the floor and residual df vs alpha median | alpha again hugs the new floor on the stand-in, or the 80 % coverage moves away from 0.80 |
| A2 | the freeze is the parametrisation: alpha = alpha_min + exp(raw) (or a softplus with a larger beta / a straight-through floor, TF `alpha_floor_straight_through`) keeps a gradient | same look; gradient share reaching raw alpha at step 800 | the gradient share stays < 1 % and alpha stays at the floor |
| A3 | alpha and beta are redundant for the variance: predict (mu, log sd, df) directly (a Student-t head) and keep the NIG only for nu | gate C (coverage 75-85 %, T-sharp) on E1 / E3, 3 seeds | no gain in NLL or coverage over the NIG head at equal steps |
| A4 | the evidence regulariser should be in head units (divide the error by the label scale) | E1 gate A with lambda in head units | no change in the nu / alpha behaviour |

None of these is a default change; each would be a registered switch (rule B1) with its gate run (rule C6).

## 5. Training speed: where the time goes

Facts: notebook-main2 (default model, 102,310 parameters, GRU encoder, batch 256) took 94 s per epoch on a T4
(`trace_epoch_seconds`). The model is tiny: on a GPU one step is a few thousand very small kernels (encoder, the
decomposition and stats branches, three NIG heads, three class heads, three confidence heads, the ordered means, the
Kendall weights, six losses), so the GPU waits on kernel launches and Python, not on arithmetic (`nvidia-smi` showed
~15 % use of an L4 for one GRU job, `docs/COLAB_OPS.md`). On such a model the time per epoch follows the NUMBER of
steps, almost not the rows per step. CPU profile here (E1, batch 256, 4 threads): 108 ms per step, 2,369 rows/s;
the largest self times are matrix products (GRU loop), elementwise mul / add / copy (thousands of calls per step) and
dropout masks (9 %).

### Methods, ranked by expected gain for THIS model (GPU numbers are expectations, not measured here)

| # | method | how in this project | expected gain | risk / cost |
|---|---|---|---|---|
| S1 | larger batch with lr scaled (sqrt rule) | `train.batch_size` 1024-4096, `train.lr` x 2-4 | ~ linear in the batch on a launch-bound GPU (4-16 x fewer steps) | generalisation: measured below at equal epochs |
| S2 | smaller model ("lite": d_model 32, 1 GRU layer, heads 32 / 16) | `model.overrides` | fewer and smaller kernels; 4 x fewer parameters | capacity: measured below |
| S3 | fewer, decorrelated rows per epoch | `split.max_rows` (evenly strided rows); stride-1 windows of 4h bars overlap 31 / 32 bars | linear in the row share | each epoch sees a different subset only with a rotating stride (not built) |
| S4 | work on the local disk, push to Drive at the end | notebook `WORK_ON = "local"` | the per-epoch checkpoint to Drive (1.3 MB, FUSE) leaves the loop | a cut runtime loses the epochs since the last push |
| S5 | `steps_per_execution` 16-32 | `train.steps_per_execution` | one host sync per group instead of per step | none (logs only) |
| S6 | data resident on the GPU (all X once, index on the device) | not built: `Trainer._train_epoch` copies each batch from the CPU | removes the CPU gather + H2D copy per step | 2.5 M x 32 x 3 float32 = 0.97 GB: fits T4 / L4 |
| S7 | CUDA graphs / `torch.compile(mode="reduce-overhead")` | not built | the largest one for a launch-bound model (often 2-5 x) | compile time, fixed shapes (drop the last batch), the double-backward of the grad report (#23) |
| S8 | mixed precision (fp16 on T4, bf16 on L4 / A100) and fused AdamW (`fused=True`) | not built | small for this model (launch-bound), large for the transformer encoders | NLL in fp16 needs care (lgamma, log of tiny beta): keep the loss in fp32 |
| S9 | drop heads a run does not use (class heads, confidence heads, Kendall) | a head_types override plus the trainer's target config | fewer kernels | changes the model; a separate arm |

S1-S5 are settings only (no code). S6-S9 are code changes and need their own switch, test and GPU measurement.
