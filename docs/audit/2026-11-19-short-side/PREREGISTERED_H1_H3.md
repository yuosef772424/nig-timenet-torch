# Pre-registration: price-unit short rules H1 / H3 on saved tables (owner 2026-11-19, written before any number)

Owner request 2026-11-19: test, without retraining, (1) a decision rule in price units: no short unless the expected
edge beats s^2/2 plus cost, and a higher confidence bar for shorts than for longs, fitted on VAL and tested on TEST
(H1 of short-side-001); (2) no shorts in the most uncertain fifth (sigma q5) and no low-confidence shorts (H3).

## Universe and notation
- One fold = its VAL table (thresholds) and its TEST table (judgment). Battery tables (`eval.battery_adapter`):
  `mu` and `y_ret` are simple returns, `sigma` the decoded width, `p_dir` the close class head.
- Side = +1 if `p_dir > 0.5` else -1. Confidence `c = |p_dir - 0.5|`. Log-space sd `s = log(1 + sigma / (1 + mu))`.
  Round trip `rt = 0.0016` (0.0008 per side). Net of a trade = `side * y_ret - rt`.
- Universe U = scored rows (`dir_true != 0`) in the top 20 % |mu| (threshold = VAL 0.8 quantile of |mu|): the "top"
  rows where the short side loses (short-side-001). The LONG side is never changed by a rule.

## Rules (each changes the SHORT side only)
| id | short kept when | thresholds from VAL |
|---|---|---|
| R0 | every short in U (baseline arm) | none |
| H1a | `-log(1 + mu) > s^2 / 2 + rt` (the predicted log fall beats the exp() part plus cost) | none (model outputs only) |
| H1b | `c >= t_short`, where `t_short` is the VAL quantile q in {0.5, 0.6, 0.7, 0.8, 0.9} of `c` over the VAL shorts in U that maximises the VAL short net per trade | q chosen on VAL |
| H3 | `sigma < VAL 0.8 quantile of sigma` (all VAL rows) AND `c >= VAL median of c` over the VAL shorts in U | both quantiles |
| H13 | H1b AND H3 | as above |

5 rules x the folds; every rule counts as a variant (CI level `1 - 0.05 / 5` for the book rows).

## What is reported (per seed, TEST pooled over the 7 folds)
For each rule: (a) the kept shorts: n, share of R0's shorts kept, win rate, net per trade, mean day with a 5-day-block
CI, busy-day mean; (b) the DROPPED shorts: n and net per trade (a good rule drops shorts that lose); (c) the book =
R0's longs + the kept shorts: net per trade, mean day [CI], winning days, without the best 5 % of days, busy days.

## Accept rule (fixed now)
A rule is accepted on a dataset only if, on EVERY seed of it:
1. the kept shorts' net per trade is above 0 AND above R0's short net per trade;
2. the dropped shorts' net per trade is below the kept shorts' net per trade;
3. the book's net per trade AND its busy-day mean are both above R0's book.

## Datasets
- **A, development**: `rolling-single-001` seeds 0 and 1. The hypotheses were formed on these TEST days, so a pass here
  is only a look. Folds 0-1 use their real VAL tables. Folds 2-6 use the TEST table of fold k-1 as VAL (VAL tables
  above 10 MB are out of the Drive connector's reach), an earlier period.
- **B, the test**: `rolling-single-002` seeds 2 and 3 (running), with each fold's own VAL table, read on Colab. A rule
  counts only if it passes on B. Even then it is exploratory (rule C5): same data period, new model seeds.
