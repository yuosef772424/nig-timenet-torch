# rolling-single-002: the pre-registered tests on new seeds (exploratory, rule C5)

Job `docs/jobs/rolling-single-002.json`: the `rolling-single-001` configuration unchanged (single-coin GRU, lr 3e-4,
100 longest-listed coins per fold, 400k train rows, 7 expanding folds, 8 epochs, patience 3, selection
`val_raw_loss`), **new seeds 2 and 3**. It ran on 2026-11-19 on a Colab L4 at commit cc3b011 as one process per seed,
after the smoke `rolling-single-smoke-001` on the same commit (rule E5). Outputs are on Drive
`nig_timenet_runs/rolling-single-002/out_s{2,3}`. The pooled readings use each fold's OWN VAL table and were run on
Colab: `docs/audit/2026-11-19-short-side/OUTPUT_002.txt` (`read_short_side.py`) and `H1_H3_OUTPUT_B.txt`
(`h1_h3_rules.py`). Pre-registrations: `PREREGISTERED.md` (candidate) and `PREREGISTERED_H1_H3.md` (short rules).

## Verdict

- **The candidate LONG top 20 % confidence FAILS its pre-registered accept rule on both seeds.** Mean day
  +0.143 % [-0.023, +0.307] (seed 2) and +0.138 % [-0.076, +0.285] (seed 3) at 99.81 %; winning days 54.2 / 54.1 %;
  busy days +0.071 / +0.106 %; without the best 5 % of days -0.061 / -0.083 %. Two conditions fail: the CI contains 0,
  and the mean without the best days is negative. The same shape as on seeds 0 and 1, slightly weaker.
- **No short rule H1a / H1b / H3 / H13 is accepted on dataset B** (as on dataset A): the kept shorts lose under every
  rule on both seeds (condition 1). The rules raise the book only by trading fewer shorts.
- **The short-side reading of short-side-001 replicates on the new seeds**: top 5 % |mu| short exp() part -0.118 /
  -0.130 % per trade (predicted by sigma -0.115 / -0.124 %), long +0.102 / +0.101 %; short net -0.236 / -0.226 %, long
  +0.425 / +0.466 %; correlation of the short day with the market day -0.41 / -0.46.
- Model skill as before: rank IC by timestamp (seed 2) close 0.022, high 0.342, low 0.392 across folds
  (`out_s2/rolling_summary.md`).

## What the model learned

- The long side of the top rows calls the direction in log units: top 10 % |mu| long log part +0.321 / +0.301 % per
  trade. The short side much less so on these seeds: +0.056 / +0.129 % (seeds 0 / 1: +0.284 / +0.179 %).
- High / low heads rank volatility (IC 0.34 / 0.39). The close head ranks little across coins (0.022).
- The model's sigma predicts the short's exp() cost within 3-5 % on the top rows.

## Where it is right and where it fails

Per fold, top 10 % |mu| (seed 2 / seed 3), net per trade:

| fold (test from) | SHORT net | SHORT exp() | LONG net |
|---|---|---|---|
| 0 (2023-01) | -0.225 / -0.189 | -0.076 / -0.060 | +0.243 / +0.235 |
| 1 (2023-07) | -0.250 / -0.273 | -0.051 / -0.072 | +0.164 / +0.246 |
| 2 (2024-01) | -0.182 / -0.066 | -0.079 / -0.090 | +0.179 / +0.124 |
| 3 (2024-06) | -0.382 / -0.196 | -0.113 / -0.127 | +0.980 / +1.197 |
| 4 (2024-12) | +0.228 / +0.001 | -0.116 / -0.115 | +0.181 / +0.085 |
| 5 (2025-06) | -0.352 / -0.114 | -0.222 / -0.171 | +0.090 / +0.042 |
| 6 (2025-12) | -0.015 / +0.087 | -0.230 / -0.216 | -0.016 / -0.155 |

- The short loses in 11 of 14 runs. Its exp() part is negative in 14 of 14 and doubles in folds 5-6.
- The long wins in 12 of 14 runs, but fold 3 (the late-2024 rally) carries it: +0.98 / +1.20 % per trade.
- Long top 10 % |mu| has a mean day with a CI above 0 on both new seeds: +0.219 [+0.023, +0.414] and
  +0.227 [+0.011, +0.438]. It was not pre-registered, it fails "without the best days" (-0.074 / -0.071), and seeds 0
  and 1 did not show it (CI contains 0). It is a hypothesis, not a result.

## The samples it failed on

The worst short top-10 % trades are again 8 h pumps: CTSIUSDT 2026-04-02 (+93 %), DENTUSDT 2026-02-26 (+88.5 %, the
same bar as in seeds 0 and 1), NKNUSDT 2025-04-20 (+75 %), TRXUSDT 2024-12-03 (+68 %), ATAUSDT 2024-12-28, FILUSDT
2025-11-07, APEUSDT 2026-04-24. Shared traits: high sigma rank (0.75-0.98) and p_dir 0.40-0.49. On NKNUSDT the side
is short (p_dir 0.49) while mu is POSITIVE (+0.45 %): the two close outputs disagree. Trades below -10 %: 2.4 / 2.5 %;
the worst 1 % of trades hold 15-16 % of all short losses.

## Bias

- The class head calls short on 0.50-0.59 (seed 2) and 0.44-0.57 (seed 3) of rows per fold, against a realised
  falling share of 0.48-0.52. Seed 2 leans short in every fold. The lean changes with the seed: a level the training
  window imprints, not a property of the data.
- Which side carries the loss: the short, on rising-market days (short day +2.1-2.2 % in the falling fifth of days,
  -1.5-1.6 % in the rising fifth) and on busy days (busy-day mean of short top 10 %: -0.150 / -0.032 %).

## Selective use

Rules written before TEST (5 short rules, 1 candidate). No rule passes on the new seeds (above). H1a again drops the
worst shorts (dropped -0.649 / -0.286 % per trade against kept -0.112 / -0.058 %), but what remains loses. The book
with the fewest shorts (H13) has the best book numbers (+0.061 / +0.072 % per trade, busy +0.070 / +0.071 %), still
under the long side alone. The dataset A comparison was long only +0.12 % per trade.

## Why it failed

**The direction edge of this model lives on the long side of the top rows, and mostly in market rallies.** Not trading
shorts helps because the short pays the exp() part and fights the market drift. The confident long rows are no better
than the top-|mu| long rows. They add winning days (54 %), but the mean depends on the best 5 % of days on all four
seeds. Evidence: the per-fold table (fold 3 carries the long side), the "without the best days" column on seeds 0-3,
and the hedged result of short-side-001 (both sides lose against the market move).

## Next hypotheses

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| N1 | The long top rows' edge is a market-beta: a long book hedged with the equal-weight market (or BTC) loses on every seed | hedged long top 10 % net stays above 0 on 2 of the 4 seeds | `hedge_and_stop.py` on the 002 tables (Colab) |
| N2 | Long-only top 10 % |mu| with a regime filter written on VAL (trade only when the market's trailing 7 d move is up) keeps the days that pay | the VAL-fitted filter does not raise the "without the best days" mean above 0 on TEST, seeds 2 and 3 | saved tables plus market returns; pre-register first |
| N3 | Rows where p_dir and sign(mu) disagree hold the worst trades (NKNUSDT) | the net of agreeing rows is not above the net of all rows on both sides | saved tables, one rule |
| N4 | A market-relative target (panel relative inputs + rank term) removes the beta and keeps a coin-level edge | panel walk-forward hedged top rows lose like the single-coin model | `hedge_and_stop.py` on `panel-rolling-ref-003` tables |
