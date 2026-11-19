# ls1-side-heads-001: a loss per trade side (exploratory, rule C5)

Owner task 2026-11-19: test improving the loss function. mu stays on the symmetric NIG log loss. Two auxiliary close
heads (D-038, off by default): `side_payoff` (each side regressed on what it is paid after a 0.16 % round trip: long
`e^y - 1 - c`, short `1 - e^y - c`) and `upper_quantile` (the 0.9 quantile of y, pinball loss). Pre-registration:
`docs/audit/2026-11-19-loss-side-heads/PREREGISTERED.md` (with amendments 1 and 2, both written before any
walk-forward number). Outputs: `E1_OUTPUT.txt`, `E1_READOUT.txt`, `E1_10EPOCHS.txt` (known answer), `WF_OUTPUT.txt`
(walk-forward), `FACTS_OUTPUT.txt` (the reading below), all in that folder.

Walk-forward: the `rolling-single-002` configuration (single-coin GRU, lr 3e-4, 100 coins per fold, 400k train rows, 7
expanding folds, 8 epochs, patience 3), seeds 2 and 3, on a Colab L4 at commit b8e0051, after the smoke
`rolling-ls1-smoke-001` on the same commit (rule E5). Arms: A0 baseline, R (both heads, `aux_trunk_grad` 0 = read-outs),
S (side_payoff, `aux_trunk_grad` 0.3). Thresholds from each fold's VAL table, the TEST trades of 7 folds pooled per seed.

## Verdict

- **Every pre-registered test fails on both seeds.** P1 (the short head picks better shorts), P1L (the long head picks
  better longs), P2 (letting it shape the trunk helps), P3 (the quantile veto beats the NIG upper-quantile veto), P4 (a
  VAL-fitted linear payoff per side on mu, s, s^2). The health checks pass: R's close rank IC 0.019 / 0.018 against
  A0's 0.022 / 0.021 (within 0.01), q_up TEST coverage 0.898 / 0.896.
- SHORT top 5 % net per trade, seeds 2 / 3: by `pay_short` -0.259 / -0.101 % (R), by `-mu` -0.151 / -0.029 %, by
  `E_short` -0.164 / -0.037 %. The short head picks WORSE shorts than mu.
- Known answer (E1) predicted it: the side heads rank the oracle payoff at 0.43-0.64 against 0.59-0.82 for the closed
  form of mu, on every seed, also with inverse-variance weights (arm W) and at 10 epochs; the quantile head passes
  (coverage 0.87-0.89, Spearman 0.95-0.97 with the oracle quantile).

## What the model learned

- The quantile head learns the upper quantile well: coverage 0.90 on real TEST rows of every fold, and its value is
  2.6-2.9 x higher on the worst squeezes (6.9-7.6 %) than on the median row (2.6 %).
- The side heads learn a noisy copy of mu: Spearman(pay_short, -mu) 0.63 / 0.66, with the predictive sd -0.05 / -0.09.
  Their rank correlation with the realised short payoff is 0.014 / 0.020, against 0.048 / 0.046 for -mu.
- E1 shows the information is in the model: a ridge of the payoff on the trunk h reaches the closed form (0.64-0.70),
  and a ridge on (mu, s) beats it (0.73-0.75). The small nonlinear head loses it on a target whose per-row signal is
  ~1 % of its variance.

## Where it is right and where it fails

- By decile of `pay_short` (R, pooled TEST, realised short net per trade, low -> high), seed 2: -0.376 -0.214 -0.237
  -0.175 -0.113 -0.152 -0.117 -0.095 -0.085 -0.184 %; seed 3: -0.427 -0.238 -0.212 -0.158 -0.152 -0.146 -0.150 -0.084
  -0.092 -0.090 %. The head separates the worst decile from the rest, but the top decile is not the best and no decile
  is above 0.
- The top and bottom deciles are the most volatile (log sd 0.031-0.034 against 0.025-0.028 in the middle): the head
  ranks volatility at both ends, and the squeezes sit in both.
- The long side is unchanged: LONG top 5 % by mu +0.34 .. +0.39 % per trade on every arm and seed, mean day CI just
  above 0 on seed 3, negative without the best 5 % of days (as in rolling-single-002).

## The samples it failed on

The worst trades of SHORT top 5 % by `pay_short` are the same 8 h squeezes as in short-side-001: DENTUSDT 2026-02-26
04:00 (+88.5 %), DENTUSDT 2026-04-21 (+64.7 %), TRXUSDT 2024-12-03 (+67.6 %), UNFIUSDT 2024-10-23 12:00 and 16:00
(+57.6 / +59.8 %), FILUSDT 2025-11-07 (+57.8 %), ALPHAUSDT 2024-10-22 (+40.6 %). Their q_up was 4-11 %: above the
median row, but every move above +20 % exceeded its q_up (a +60 % bar is far beyond any 0.9 quantile).

## Bias

- Levels are right: mean `pay_short` -0.153 / -0.182 % against a realised -0.175 %; mean `pay_long` -0.167 / -0.138 %
  against -0.145 %. The head predicts a positive short payoff on 16 / 11 % of rows and a positive long payoff on 14 /
  21 %: the lean changes with the seed, as the class head's does.
- The loss sits on the short side and in its most volatile rows, as before.

## Selective use

Four primary rules and one secondary, all written before reading TEST (PREREGISTERED.md, 5 tests x 2 seeds). The
quantile veto drops worse rows than it keeps on both seeds (dropped -0.218 / -0.186 % against kept -0.133 / +0.015 %),
but it is no better than the same veto on the NIG model's own upper quantile `log(1 + mu) + 1.28 s` (kept -0.132 /
-0.005 %): the head adds nothing over sigma. No kept short book is above 0 with a CI (the positive "mean day" of the
short books is the thin-day effect of short-side-001; their busy-day means are near 0 or negative).

## Why it failed

**Signal there but not read by the new heads, and the short side's loss is not in the inputs.** The side heads see the
same trunk as mu and, trained on a noisy price-unit target, end as a noisier version of mu (E1: the information is in
h; real data: rank correlation with the realised short payoff 0.014-0.020 against mu's 0.046-0.048). The quantile head
reads the tail correctly but only as well as the NIG sigma already does. The worst short losses are squeezes no input
announces (B-077, B-055): a loss term cannot create that information.

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| L1 | The constant learning rate under-trains the model (best epoch = last in 4 / 7 A0 folds); a cosine decay raises the close rank IC | arm C does not pass `PREREGISTERED_LR1.md` | running: `rolling-lr1-cos-001` |
| L2 | The pumps are announced by volume / liquidity (not in the artifact, B-055): with volume features the quantile head's veto beats the sigma veto | q_up with volume does not drop the squeezes better than sigma | a volume artifact, then the R arm again |
| L3 | A side decision needs a market-relative target (the long edge is beta, rolling-single-002 N1): hedged books show whether any side edge is coin-level | hedged top rows lose on every seed | `hedge_and_stop.py` on the A0 tables |

What is kept: the switches stay available (off by default); the quantile head is a correct, cheap read-out of the
upper tail for later rules; arm W's weighting stays as an option (no consistent gain on E1).
