# Snapshot panel model: design (2026-11-18)

Input of one moment: S coin slots x T bars x C channels (S = 128 slots for the ~100 coins of a fold, T = 32, C = 3
log-move channels). Output: NIG (mu, nu, alpha, beta) + confidence + class probability for high / low / close of every
coin present (100 x 3 x NIG). Code: `src/nig_timenet_torch/model/snapshot.py`.

| part | what | size (default) | why (accuracy) |
|---|---|---|---|
| input scaler | per channel (x - mean) / sd on the train rows, clip 5 | - | the label / input units problem (N-01) never comes back |
| coin slots | asset index -> slot, the 128 coins with most train rows; others "unknown" | buffer | a fixed meaning per input / output position, fitted on train only (rule B4) |
| joint path (the owner's design) | per bar: [z_1..z_S, present_1..present_S] (S x (C + 1) = 512 values) -> Linear 512 -> 256 (per-coin input weights) -> LayerNorm -> GELU -> GRU 256 x 2 layers -> RMSNorm: the moment state m | ~0.92 M | every coin's details reach every coin: lead / lag, per-coin beta to the market (the E4 signal is only readable after removing beta_i x market) |
| per-coin output weights | `slot_out`: m -> raw NIG parameters of coin s with its own weights (S x 256 x 12), zero at start | ~0.40 M | the "300 outputs with their own weights" of the owner's design; zero init = starts from the shared readout |
| shared path (switch) | relative inputs within the timestamp -> Linear -> GRU 64 x 2 -> RMSNorm, the same weights for every coin | ~0.05 M | keeps what today's panel model reads (it is learned from 100 x more rows than any per-coin weight) |
| coin embedding | 32 per slot (+ unknown) | 4 k | a coin's own offset / scale |
| readout | [shared state, m, embedding] -> Linear -> GELU -> dropout -> RMSNorm (128) | ~45 k | one representation per coin for every head |
| heads | NIGHead per target (floors, fitted label scale, raw + per-coin terms), ordered means (high >= close >= low), NIG uncertainty, confidence, binary class heads | ~30 k | the same output contract as every architecture: the trainer, the battery, the reports, the rolling entry run unchanged |

Regularisation for few independent samples (one moment = one joint sample; ~14k moments in 6.5 years): dropout 0.25
(GRU between layers, readout, heads), whole-coin dropout 0.1 on the joint input (no reliance on one coin or on a fixed
universe), input noise 0.1, weight decay 1e-4, early stop on `val_rank_ic_reg` (patience 3), the cross-section rank
term in the loss (ranking inside the moment). Arms `joint` (shared path off) and `hybrid` (both) in
`PREREGISTERED.md`.

Batch: ~32 whole moments (3200 rows), so the joint encoder sees 32 sequences per step.
