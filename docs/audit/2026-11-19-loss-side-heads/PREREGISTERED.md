# LS1 pre-registration: a loss per trade side (written 2026-11-19, BEFORE any number of these arms)

Owner task 2026-11-19: test improving the loss function. mu stays on the symmetric NIG log loss (an asymmetric weight
on the one shared mu would lean every prediction one way). Two new heads on the close target (D-038, off by default):

- `side_payoff`: two outputs, each regressed (squared error / 0.02) on what that side is paid after a round trip of
  0.16 %: long `e^y - 1 - 0.0016`, short `1 - e^y - 0.0016` (y = the log-return label). The short's exp() cost is
  inside its own target.
- `upper_quantile`: the 0.9 quantile of y (pinball loss), for a "no short into a likely squeeze" veto.

The cheap competitor that needs no training is the closed form of the NIG outputs (H1a, short-side-001):
`E_long = exp(log(1 + mu) + s^2 / 2) - 1 - 0.0016`, `E_short = -(exp(log(1 + mu) + s^2 / 2) - 1) - 0.0016`, with
`s = log(1 + sigma / (1 + mu))` the predictive sd in log units (battery columns mu, sigma). A head that does not beat
it adds nothing.

## 1. Known answer first: E1 (CPU, `nig-envs` E1 defaults, 3 epochs, seeds 0 / 1 / 2)

Arms: `base` (the default preset), `R` (`aux_heads` side_payoff + upper_quantile, `aux_trunk_grad` 0), `S`
(side_payoff, `aux_trunk_grad` 0.3). E1's close label is Gaussian in log units with a known mean m and sd v per row,
so the oracle payoffs are `exp(m + v^2 / 2) - 1 -/+ ...` and the oracle 0.9 quantile is `m + 1.2816 v`. Read on the
TEST rows. Accept, on EVERY seed:

| # | check | pass |
|---|---|---|
| E1-a | the short head ranks the oracle short payoff | Spearman(pay_short, oracle) >= Spearman(E_short of the same arm, oracle) - 0.05, R and S |
| E1-b | the long head likewise | the same with pay_long / E_long |
| E1-c | the payoff heads are calibrated in level | abs(mean pred - mean oracle) <= 0.3 sd(oracle), each side |
| E1-d | the quantile head covers and ranks | coverage of q in [0.87, 0.93] and Spearman(q, oracle q90) >= 0.8 (R) |
| E1-e | the read-out leaves mu alone | R's Spearman(close mu, oracle mean) within the base arm's min / max over seeds +- 0.02 |

A failure here is read (rule C7) before any GPU time is spent.

## 2. The walk-forward (GPU, one job, one commit)

The configuration of `rolling-single-001/002` unchanged (single-coin GRU, lr 3e-4, 100 coins per fold, 400k train
rows, 7 expanding folds, 8 epochs, patience 3, selection `val_raw_loss`), seeds 2 and 3, arms:

- `A0`: the baseline (no auxiliary head);
- `R`: side_payoff + upper_quantile, `aux_trunk_grad` 0 (read-outs: the trunk and the other heads train as A0 up to
  the random path);
- `S`: side_payoff, `aux_trunk_grad` 0.3 (the owner's "partial stop of the gradient"), `aux_loss_weight` 1.

Every threshold of fold k is fitted on fold k's VAL battery table and applied to its TEST table; the TEST trades of the
7 folds are pooled per seed. Cost 0.0008 per side. Rows with `dir_true == 0` are not traded. "Top 5 %" = rows whose
score is at or above the VAL 0.95 quantile of that score; ALL those rows are traded on the named side (no class
head). Per book: trades, net per trade, mean day with a day-block CI (99 %), mean of busy days (days with at least the
median number of trades), mean without the best 5 % of days: `eval.battery_scorecard.day_profile`.

| # | test (primary unless marked) | books compared (same arm, same rows) | accept on BOTH seeds |
|---|---|---|---|
| P1 | the short head picks better shorts | R: SHORT top 5 % by `pay_short` against SHORT top 5 % by `-mu` and SHORT top 5 % by `E_short` | net per trade AND busy-day mean of the `pay_short` book above both others |
| P2 | letting it shape the trunk helps | S's SHORT top 5 % by `pay_short` against R's | net per trade above R's, and S's close rank IC by timestamp (fold mean) not below A0's by more than 0.01 |
| P3 | the quantile veto removes the squeezes | R: universe = SHORT top 5 % by `-mu`; drop its rows with `q_up` at or above the VAL 0.8 quantile of `q_up` over the VAL universe, against dropping the same share by the NIG upper quantile `log(1 + mu) + 1.2816 s` | kept net per trade above the NIG veto's kept net, and the dropped rows' net below the kept rows' |
| P1L (secondary) | the long head picks better longs | R: LONG top 5 % by `pay_long` against LONG top 5 % by `mu` and by `E_long` | as P1 |
| H (health, not a test) | the read-out does not harm the baseline | R's close rank IC against A0's; `q_up` TEST coverage | R within 0.01 of A0; coverage in [0.85, 0.95] |

Three primary tests (multiple testing is counted in the reading). Profit is not an accept condition here (the question
is whether the loss helps the decision); each book's scorecard verdict is reported beside it. A P that passes is a
hypothesis for the next walk-forward on new seeds, not a result (rule C5). A P that fails gets the failure reading of
`docs/PHILOSOPHY.md` section 3 before the next arm.

## Amendment 1 (2026-11-19, after reading the E1 rows of R and S, before any walk-forward number)

E1 showed the side-payoff heads (plain squared error) ranking the oracle payoff at 0.43-0.64 against 0.59-0.73 for
the closed form of mu (E1-a / E1-b fail on every seed), while the quantile head passes (E1-d) and the read-out leaves
mu alone (E1-e). Named cause, to be checked: the squared error on a heteroscedastic payoff is dominated by the
volatile rows' noise, while the NIG loss weights each row by its inverse variance. New switch
`TrainSettings.aux_variance_weight` (rows weighted by the detached NIG predictive variance^-v, batch mean 1; the
minimiser stays the conditional mean payoff). New arm `W` = R with `aux_variance_weight` 1. E1 is re-run from fresh
folders for ALL arms (the first run's seeds 0 and 1 resumed from the checkpoints of an interrupted run) with the same
checks E1-a .. E1-e (E1-d / E1-e also on W). If W passes E1-a / E1-b, W joins the walk-forward as a fourth arm with
the tests P1, P3 and P1L read on W exactly as on R; A0 / R / S already run on commit b8e0051, W runs on a later commit
that differs only by the off-by-default switch (P1 / P3 / P1L compare books inside one arm, so the commit difference
does not enter them).

## Amendment 2 (2026-11-19, after the clean E1 run, the read-out diagnostic and the 10-epoch E1 run; no walk-forward number read)

E1 results (`E1_OUTPUT.txt`, `E1_READOUT.txt`, `E1_10EPOCHS.txt`): the side-payoff heads fail E1-a / E1-b on every seed
for R, S and W, also at 10 epochs (head 0.43-0.64 against the closed form 0.70-0.82); the quantile head passes E1-d
and the read-out leaves mu alone (E1-e). A ridge of the realised payoff on the trunk h reaches the closed form
(0.64-0.70), and a ridge on (mu, s) beats it (0.73-0.75): the information is in the model, the small nonlinear head
loses it on a target whose per-row signal is ~1 % of its variance. Consequences, fixed now:

- W is NOT run on the GPU (it failed its E1 gate, MODEL_DEV_PROTOCOL section 2). R and S run as planned; their P1 /
  P1L / P2 are read and are expected to fail.
- New primary test **P4** (the owner's "train each side on its real payoff", with three parameters per side instead of
  a head): per fold, least squares on the fold's VAL rows of each side's realised payoff after cost (long `y - RT`,
  short `-y - RT`, y = the realised return) on `[1, log(1 + mu), s, s^2]`; the score `lin_short` / `lin_long` is that
  fit applied to TEST. Books: SHORT top 5 % by `lin_short` against SHORT top 5 % by `-mu` and by `E_short`; accept on
  BOTH seeds as P1 (net per trade and busy-day mean above both). Read on A0 (the baseline model, so a pass needs no new
  training) and on R. LONG top 5 % by `lin_long` is reported as P4L (secondary). Four primary tests now (P1, P2, P3, P4).
