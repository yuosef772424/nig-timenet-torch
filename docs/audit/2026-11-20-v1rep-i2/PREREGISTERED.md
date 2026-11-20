# Pre-registration: V1-rep and I2 (owner 2026-11-20: "run the proposals, in parallel, free RAM under 10 GB")

Written before any number of these runs. Exploratory (rule C5). One commit for all runs; folds split over processes
(`RollingSettings.fold_ids`, D-038), `chain.sh` in this folder.

## V1-rep: is V1 a seed accident? (hypothesis V1-rep of `docs/results/m3-full-001.md`)
Runs: arm R (art1) and M3 (art_mres) on seeds 2 and 3, the configuration of `rolling-ind-001` arm R otherwise (jobs
`rolling-r23-001`, `rolling-mres23-001`). V1 is built per seed from R and M3 of the SAME seed and commit, exactly as in
`docs/audit/2026-11-19-arch-indicators/m3_readd.py` (score = M's mu + beta x the mean of R's mu at the timestamp; VAL
90th percentile).
Accept "V1 replicates": on BOTH seeds 2 and 3, V1's day mean without the best 5 % of days is above 0, AND V1's mean
day is above V0's (R's own long book). Read with `m3_readd.py` and `m3_v1_read.py` (amendment 1 of
`PREREGISTERED_M3_FULL.md`: hedged views, beta, per fold / year, timing-only), information next to the rule.
Also reported: M3's residual IC on seeds 2 / 3 (feasibility repeated) and R's long book on seeds 2 / 3.

## I2: do the residual indicator channels carry M3's ranking? (hypothesis I2 of `docs/results/i1-001.md`)
Run: M3 with `art_mres_noind` = `art_mres` without `res_rsi_14`, `res_stoch_k_14_3`, `res_stoch_d_3`,
`volume_rel_30` (kept: `res_high`, `res_low`, `res_close`, `mkt_close`, `mkt_rsi_14`, `beta`; `drop_channels.py`),
seeds 0 and 1 (job `rolling-mres-noind-001`, smoke `rolling-mres-noind-smoke-001`). Reference: M3 seeds 0 / 1
(`rolling-mres-001`, commit aa738f3; model / trainer / evaluation code unchanged since).
Rule: "the channels carry information" if the close rank IC by timestamp against the residual label falls by more than
0.005 (from 0.093 / 0.094) on BOTH seeds; "they do not" if it is within 0.005 on both; mixed otherwise. Reported next
to it: the top-decile residual gross per trade and the decile-10 excess over the timestamp mean.

## Amendment 1 (2026-11-20, after the start, before any number): seeds 4 and 5 for V1-rep
The owner asked to use the RAM (free RAM under 10 GB); the 12 processes left 26 GB free with the GPU at 99 %, so
more processes add information, not speed. Added: R and M3 on seeds 4 and 5 (jobs `rolling-r45-001`,
`rolling-mres45-001`), same commit and configuration. The rule is unchanged and now applies to seeds 2, 3, 4 and 5:
"V1 replicates" if it holds on at least 3 of the 4 new seeds (one seed of four may fail by chance), and V1 is NOT
recommended for live use if it fails on 2 or more.
