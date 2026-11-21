# Pre-registration I1: the indicator channels trained like R (owner 2026-11-20)

Written before any number of the run. Exploratory (rule C5). Hypothesis I1 of `docs/results/arm-i-001.md` (owner:
"repeat the hypothesis"): arm I's indicator channels were never given a fair test, because arm I also changed the
encoder (tcn) and the training budget (lr 1e-5 x 16 epochs, best epoch = the last in 14 / 14 runs).

## Arm
I1 = arm R of `rolling-ind-001` with ONLY the artifact changed: `/content/art_ind` (art1's 3 price channels + RSI 14,
stochastic %K / %D, relative quote volume; Drive copy `crypto_model/torch_artifacts/preprocessing_output_1_plus_ind`).
GRU 64 x 2, lr 3e-4, 8 epochs, patience 3, 400k train rows, top 100 coins, 7 expanding folds (TEST 2023-01 ..
2026-06), seeds 0 and 1. Jobs `docs/jobs/rolling-indr-001.json` (config equal to R's `out_R_s0/manifest.json` apart
from artifact and run_dir, checked 2026-11-20) and the smoke `rolling-indr-smoke-001` (rule E5), run by `i1_chain.sh`.

Reference R = `rolling-ind-001/out_R_s{0,1}` (on Drive). R ran on an earlier commit (manifest 31796e5); between it and
this run only the artifact tools in `data_io/` changed (`btc_relative`, `market_residual`, `indicator_channels`
column names), not the model, trainer or evaluation, so the arms are compared row by row on the same folds (the
one-commit rule C4 is relaxed for that reason, and said so in the results note).

## Accept rule (both seeds)
1. The hypothesis: close rank IC by timestamp of I1 above R's in at least 5 of the 7 folds on each seed.
2. The use: the long book of `PREREGISTERED.md` (VAL 90th percentile of mu, net = y_ret - 0.16 %) on I1 has a mean
   day net above R's on the same seed, and its mean without the best 5 % of days above 0.
Reported next to it, information only: CI of the long book (`read_arms.py`, level 1 - 0.05 / 2), the market- and
BTC-hedged books, the best epoch per fold (`read_detail.py` training table), the overlap of I1's and R's trades.
If (1) holds and (2) fails, the channels help ranking but not the book (the cost wall again); if (1) fails, the
channels add nothing at R's budget either. Either fail gets the failure reading of rule C7 before any next arm.

## Amendment 1 (owner 2026-11-20, before any number): folds split over processes
The owner asked to run the folds in parallel (each fold trains an independent model). `i1_chain.sh` now runs each
seed as three processes over fold groups {0, 1, 2}, {3, 4}, {5, 6} (`RollingSettings.fold_ids`, D-038); the model,
data and accept rule are unchanged (the fold geometry and identities are the same: tested in
`tests/workflow/test_rolling_run.py`). Also measured: wall time per fold against M3's walk-forward of today (same
configuration and size, 4 processes, one per seed).
