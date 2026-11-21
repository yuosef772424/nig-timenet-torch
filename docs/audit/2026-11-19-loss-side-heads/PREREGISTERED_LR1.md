# LR1 pre-registration: cosine learning-rate decay (written 2026-11-19 19:11 UTC, before any number of arm C is read)

Gap: the run path never passed an lr schedule (every run trained at a constant lr); in `rolling-ls1-a0-001` the best
epoch was the last of 8 in 4 of 7 folds. E1 gate passed (`E1_LR_COSINE.txt`: cosine beats constant on mu ranking on 3
of 3 seeds). Arm C = `rolling-lr1-cos-001` (A0 + `lr_schedule` "cosine", `lr_min_ratio` 0.1, the same 8 epochs),
seeds 2 and 3; reference A0 = `rolling-ls1-a0-001` (same config and seeds; the default path is byte-equal).

Read with `read_ls1.py` (the same books) and the fold rows. Accept C as a candidate default (a hypothesis for a
confirming walk-forward on new seeds, rule C5) only if, on BOTH seeds:

1. the close rank IC by timestamp (fold mean) of C is above A0's; and
2. the mean selected validation loss (`best` of `best_meta.json`, lower is better) is lower than A0's in at least 5 of
   the 7 folds; and
3. no book of `read_ls1.py` that A0 has (LONG / SHORT top 5 % by mu, E) loses more than 0.05 % per trade against A0.

Reported beside it, not part of the rule: the best epochs, the high / low rank ICs, the scorecard columns of each book.
