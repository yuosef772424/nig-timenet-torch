# Rolling walk-forward, CPU smoke check on E1 (ROADMAP F2, builder half)

**This is a smoke check of the machinery, not a result (rule C5).** It shows that the folds, the per-fold scalers, the
seeds loop and the summaries run end to end and what the output looks like. The environment is the synthetic E1 (8 coins,
3,200 bars, a planted close IC of 0.05), a small model (3 epochs) and 2 seeds: nothing here says anything about the real
data, the T8 model or the stability of any skill.

Command (CPU, 25 s): `python docs/results/cpu_smoke_st1_f2.py WORK rolling '{"train_span":"150D","test_span":"40D",
"step":"40D","embargo":"1D","max_folds":4,"seeds":[0,1]}'`. Expanding folds, `retrain_every_k` 1, `val_share` 0.15 (the last 15 %
of each train window selects the epoch, D-031), battery off. Each fold trains on rows with `label_end` <= its cut minus
the validation tail; the test window starts after the 1-day embargo and is read only by the report.

4 folds; per fold: mean over seeds (sd over seeds); last rows: across folds (mean of the fold means, sd over folds).

| metric | fold 0 | fold 1 | fold 2 | fold 3 | across folds |
|---|---|---|---|---|---|
| targets.close.sharp.coverage | 0.8252 (0.0086) | 0.7864 (0.0137) | 0.8107 (0.0154) | 0.7749 (0.0146) | 0.7993 (0.0228) |
| targets.close.sharp.share_std_below_move | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| targets.close.skill.rank_ic_by_timestamp | -0.0162 (0.0291) | 0.0491 (0.0149) | -0.0096 (0.0136) | 0.1514 (0.0232) | 0.0437 (0.0776) |
| targets.high.sharp.coverage | 0.8647 (0.0060) | 0.8683 (0.0009) | 0.8853 (0.0094) | 0.8574 (0.0009) | 0.8689 (0.0118) |
| targets.high.sharp.share_std_below_move | 0.3453 (0.3012) | 0.5091 (0.1021) | 0.4842 (0.0566) | 0.3859 (0.2437) | 0.4311 (0.0781) |
| targets.high.skill.rank_ic_by_timestamp | 0.1888 (0.0207) | 0.2443 (0.0167) | 0.2594 (0.0026) | 0.2665 (0.0042) | 0.2397 (0.0352) |
| targets.low.sharp.coverage | 0.8975 (0.0180) | 0.8362 (0.0034) | 0.8744 (0.0077) | 0.8325 (0.0103) | 0.8601 (0.0313) |
| targets.low.sharp.share_std_below_move | 0.3228 (0.0618) | 0.2045 (0.0644) | 0.2609 (0.0206) | 0.2803 (0.0154) | 0.2671 (0.0491) |
| targets.low.skill.rank_ic_by_timestamp | 0.3206 (0.0234) | 0.3031 (0.0167) | 0.2152 (0.0174) | 0.3309 (0.0048) | 0.2924 (0.0527) |

| fold | train end | test window | test rows | trained |
|---|---|---|---|---|
| 0 | 2023-06-16 00:00:00 | 2023-06-17 00:00:00 .. 2023-07-27 00:00:00 | 824 | [True, True] |
| 1 | 2023-07-26 00:00:00 | 2023-07-27 00:00:00 .. 2023-09-05 00:00:00 | 824 | [True, True] |
| 2 | 2023-09-04 00:00:00 | 2023-09-05 00:00:00 .. 2023-10-15 00:00:00 | 824 | [True, True] |
| 3 | 2023-10-14 00:00:00 | 2023-10-15 00:00:00 .. 2023-11-24 00:00:00 | 824 | [True, True] |

How to read: per fold, the number is the mean of the test report's scalar over the 2 seeds and the bracket the seed sd; the
last column is the mean of the 4 fold means and the sd over folds. With 2 seeds and 4 folds the sds are 1-4 numbers: they
are a look at the spread, not an interval. The E1 close head's planted IC is small, so a close rank IC that moves
from -0.02 to +0.15 across folds is exactly the fold-to-fold noise the study exists to show; the high / low rank ICs
(0.19-0.33) are far larger than the planted close IC; this smoke does not say why (not investigated).

Not verified: any real-data number, the battery per fold (switched off here), `retrain_every_k` > 1 on a long run (covered
by the tests), the spans against the real artifact's date range (the job specs carry placeholders).
