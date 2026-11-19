# Lessons from the experiments so far (owner request 2026-11-19)

Source: every results note on `main` and on the open PRs #63 (batch-by), #66 (snapshot / alpha-beta), #67 (panel std,
two frames, more data, memorisation, dominance, BTC hedge), the failure charts, the model-first notes, the panel
handoff, plus today's reading of the single-coin walk-forward tables (`docs/results/short-side-001.md`). A reading
agent collected the numbers; the lead checked the ones quoted here against their notes. All exploratory (rule C5);
no use of the outputs is profitable after cost on any split read so far.

## 1. What the model reliably learns

| Skill | Evidence | Trivial baseline it must beat |
|---|---|---|
| High / low heads rank VOLATILITY | IC high 0.36-0.39, low 0.44 (single-coin and panel, every fold) | HAR / 6-bar mean range ties or beats it (mean6 0.382 / 0.446; ridge beats A0 in 4/4 folds, batch-by-wf-002) |
| Close head: a small cross-sectional rank | relative rank IC 0.06 -> 0.11 with relative inputs + rank term (panel-001), 0.087 over 7 folds (snapshot base) | shuffled 0; momentum / reversal 0.046 |
| Close head: market-direction of the top rows | top-10 % |mu| net +0.27 % (long) per trade pooled | market-hedged it is -0.32 % (short-side-001): the edge is the market move |
| Sigma ranks the error and the move size | sd vs abs error +0.19 .. +0.42 on every head and run | constant sigma 0 |
| Confidence ranks the hit rate | hit 0.50 -> 0.54 over confidence quintiles (single-coin wf) | constant confidence 0 |

## 2. What does not work, and the named cause

| Finding | Cause named | Where |
|---|---|---|
| No use of the outputs pays after 0.16 % per round trip | gross edge of a 4 h book is under half the cost (B-074); low-sigma rows move 1.15 % with hit 0.50 | panel-001, panel-rolling-std-001, short-side-001 |
| The top SHORT rows lose in every run while winning as often as the longs | a symmetric log loss, but a short is paid 1 - e^l: exp() costs it -0.11 .. -0.15 % per trade on the volatile top rows, the long gets the same as a gain | short-side-001 |
| The worst losses are squeezes of thin coins from quiet bases (+40 .. +88 % in 8 h; label clip at log 1 hid larger ones) | no input announces them (volume not in the artifact, B-055); confidence of those shorts is LOW (p_dir 0.37-0.49) | failure-charts REPORT, short-side-001 |
| One to three days decide the sign of a book | per-trade Sharpe hides day concentration | baseline-001 section 5; short-side-001 (positive only without crowded days) |
| The close head leans long or short by fold and seed (predicted up 0.06 .. 0.68 of rows against a realised 0.46 .. 0.51) | a level bet learned from the training window | deep-side-001, panel-001, batch-by-wf-*, two-frames-001 |
| Memorisation after epoch 1 on the single-coin model | train falls, val rises from epoch 2 | baseline-001, har-range-h3-001, no-instance-norm-001 |
| Std target normalisation, two timeframes, more rows, snapshot model, timestamp batches | refuted on the close IC; small gains on high / low only | panel-rolling-std-001, two-frames-001, more-data-flat-001, snapshot-panel-001, batch-by-wf-002 |
| Earlier "profitable" battery cells | a post-hoc best cell of a grid (panel-001), the B-051 decode bug (baseline-001), VAL / TEST sign flips | baseline-001, panel-001 |

## 3. What the evaluation must do (changed in this PR)

1. **Judge by days, per side.** A selection is read by its mean day with a day-block CI, its winning days, worst day,
   CVaR, drawdown, its mean without the best 5 % of days, and its mean on BUSY days (at least the median number of
   trades). The single-coin short side looked +0.30 % a day with a CI above 0 only because thin days (1-3 trades)
   count as much as crowded ones; on busy days it loses.
2. **Read confidence, uncertainty and the top together.** Every selection (top 5 / 10 / 20 % |mu|, top confidence,
   low sigma, |mu| / sigma, and the combinations) uses VAL thresholds and is printed per side; the CI is Bonferroni
   over the rows of the table.
3. **Split every trade into its log part and its exp() part.** A short that is right in log units can still lose;
   the model's sigma predicts the exp() part, so the decision can use it.
4. **Hedge before believing a direction.** A top-row edge that vanishes against the market move of its timestamp is
   a market call (both sides of the single-coin model, short-side-001).

## 4. Where to look next (ranked by evidence / cost)

1. **Long top 20 % confidence** (single-coin wf): mean day +0.195 % [+0.002, +0.420] at 99.8 % (seed 0) /
   +0.200 % [-0.035, +0.380] (seed 1), 56 % winning days, positive on busy days (+0.10 / +0.13 %), but negative
   without its best 5 % of days (-0.04 %). With the panel notes' `top_confidence` (positive in every run, no CI) it
   is the one candidate to PRE-REGISTER for the next walk-forward, with the scorecard verdict as its accept rule.
2. **Side-aware decisions in price units** (short-side-001 H1, H3): a higher bar for shorts fitted on VAL, no shorts
   in the most uncertain fifth. Needs runs whose VAL tables are reachable (the Drive connector stops at 10 MB).
3. **Market-relative scores** (short-side-001 H2, btc-hedged-book-001): the edge that survives a hedge is the
   bottom-20 % names falling behind BTC; the cost wall (B-074) and the turnover are the open questions.
4. **Volume in the artifact** (B-055) for the squeezes; until then, low-confidence shorts on high-sigma rows are the
   rows to avoid.
