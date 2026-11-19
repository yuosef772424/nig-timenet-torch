# Pre-registration: the panel model with std normalisation, walk-forward (owner task 2026-11-18)

Written BEFORE any number of the jobs below is read (`docs/PHILOSOPHY.md` section 2, step 1). Exploratory (rule C5).

## The question
Owner message 2026-11-18 (Arabic): "test the panel model, but using std normalisation of the batch data; use the
rolling training". Read as: the panel arm `relative_rank_loss_logret_std` of `workflow/panel_study.py` (D-033: the
TF `logret_std` normalisation in full, i.e. train-standardised inputs, which is already the run default
`input_scaler: train_std`, PLUS per-head train-standardised labels `model.target_norm: logret_std`), with the
relative inputs (each coin against the cross-section of its timestamp batch) and the cross-section rank term,
walked forward. Does the per-head label standardisation change what the panel model reads, and does it pay?

## Arms, data, folds (one commit, one data fingerprint `c572b1b36960`)
| job | arm | model overrides |
|---|---|---|
| `panel-rolling-std-001` | candidate | `encoder gru`, `cross_input relative`, `target_norm logret_std` |
| `panel-rolling-ref-002` | reference (baseline arm) | `encoder gru`, `cross_input relative` (= `panel-rolling-001`, re-run on this commit) |

Both: train `cross_section_loss rank_ic`, selection `val_rank_ic_reg`, 8 epochs, patience 3, batch 512, lr 3e-4,
`split.cross_section` with the 100 longest-listed coins per fold, at most 400000 train rows per fold, 7 expanding
folds (first train 1095 days, test and step 180 days, embargo 2 days), seeds 0 and 1, day metrics on (test days
pooled per seed, about 1260 days). Smoke first: `panel-rolling-std-smoke-001` (rule E5).
`panel-rolling-001` (commit `ebbd336`, the same reference arm) is a reproducibility check of the reference.

## Metrics and baselines
1. T-profit (the goal): the pooled-days panel report of each seed (`pooled_days_seed_<s>/panel_report.md`): the
   absolute and the market-neutral day books net of 0.0008 per side, next to every baseline of
   `eval/day_metrics.py`, day-block CI.
2. Paired against the reference: per fold and seed, the close / high / low rank IC by timestamp
   (`rolling_folds.csv`); the paired day difference of the neutral book on the pooled days.
3. Diagnostics the change is expected to move: high / low / close coverage of the 80 % interval and
   `share_std_below_move`, the bias of the close mean (share predicted up vs realised).

## Accept rules (fixed now)
- **Profit (T-profit):** the candidate's book beats every day-metric baseline on the pooled days with a CI that
  excludes 0, on BOTH seeds.
- **Diagnostic gain (kept as a switch, not a default):** close rank IC above the reference in >= 6 of 7 folds on
  both seeds, and the pooled mean difference larger than twice the seed spread, with no high / low rank IC loss
  over 0.01.
- **Refuted:** close rank IC not above the reference in at least 4 of 7 folds on either seed.

## Expectation (written before the run)
The rank term is invariant to the per-head affine map, so the close ranking should move little; the change is
expected mostly on the NIG heads' starting point (centred high / low), i.e. on calibration (coverage,
`share_std_below_move`) and on the close bias. A profit win is not expected: the reference's gross edge is under
half of its cost (B-074).

## Known answer
E4 (the synthetic panel, planted specific IC 0.15) through the `panel` entry, arms `relative_rank_loss` and
`relative_rank_loss_logret_std`, seeds 0 and 1, run on Colab beside the walk-forward; read against the oracle's
0.15, not against 1.

## Addendum (owner request 2026-11-18, 18:3x UTC, written before any fold of it was read)
Third arm `panel-rolling-ls-001`: the panel arm `relative_rank_loss_ls` = the reference with the best epoch chosen
by `val_ls_sharpe_close` (the per-timestamp Sharpe of the top-bottom 20 % close book) instead of
`val_rank_ic_reg`; same folds, seeds, data and code (`6c928bb`; its job file was committed later, the code is
the same). Smoke `panel-rolling-ls-smoke-001` passed (51 s). Started while the first two ran, so its fold times
are not comparable. Accept rules:
- **Profit:** as above (beats every day-metric baseline on the pooled days, CI excludes 0, both seeds).
- **Selection gain:** the paired difference of the neutral book (gross, pooled days) against
  `panel-rolling-ref-002` has a CI above 0 on both seeds; the close rank IC may fall by at most 0.01 (the metric
  it no longer selects on).

## Addendum 2 (owner 2026-11-19, written before the chain started): two timeframes, 4h + 1h
Jobs (chain `two_frames_chain.sh`, one commit): `panel-rolling-2tf-001` (reference arm + the 1h window of every row
beside the 4h window, `input_timeframes [4h, 1h]`) against `panel-rolling-ref-003` (the same arm on the same derived
artifact without the 1h input); same folds, seeds and settings as ref-002. `panel-rolling-ref-003` against
`panel-rolling-ref-002` checks that the derived artifact reproduces the 4h run (close rank IC per fold within 0.002).
Accept rules:
- **Profit:** as above (beats every day-metric baseline on the pooled days, CI excludes 0, both seeds).
- **Information gain:** close rank IC above ref-003 in >= 6 of 7 folds on both seeds and the pooled mean difference
  over twice the seed spread, with no high / low rank IC loss over 0.01. **Refuted** if not above in at least 4 of 7
  folds on either seed.
- Expectation: the 1h shape of the last 4h bar (where inside the bar the move happened) adds to the last-bar
  asymmetry (DS1); a small close IC gain at most.

## Addendum 3 (owner 2026-11-19, before any number of it): the USDT supply channel, walk-forward
`panel-rolling-usdt-001` = the reference arm + the channel `usdt_supply_g7` (7-day log growth of the USDT supply,
the day before each bar), same folds / seeds / settings, against `panel-rolling-ref-002` (and `-ref-003`). The
model-free look (dominance-look-001, addendum 2) says the channel is about the market's DIRECTION, so:
- **Accept (direction gain):** the absolute book (metric 1) paired day difference against the reference, gross, on
  the pooled days, has a CI above 0 on both seeds.
- **Profit:** as above (beats every baseline on the pooled days, CI excludes 0, both seeds).
- **Refuted:** the paired absolute-book difference holds 0 on both seeds. Close rank IC is reported, not judged
  (a common channel cannot change the ranking by itself).

## Addendum 4 (owner 2026-11-19, before any number of it): a BTC-relative book against the market-neutral book
No retraining: a common reference does not change the ranking, so the book is built from the saved TEST frames of the
finished walk-forwards (`panel-rolling-ref-002`, `-std-001`, and `-2tf-001` / `-ref-003` / `-ls-001` when done), the
1225 pooled days per seed. Books per timestamp, top / bottom 20 % by the close mean mu:
- `neutral` (today): +top 20 % / -bottom 20 % (half a unit each).
- `btc_hedged`: +top 20 % (half a unit) / -BTC (half a unit).
- `btc_pair_sign` (information): every coin with mu above BTC's mu long, below short, each against BTC.
Costs per side: (a) the flat battery cost 0.0008 on every leg every timestamp (today's convention); (b) per
instrument: alts 0.0008, BTC 0.0004 (deeper book); (c) as (b) with the BTC leg held (charged only on the change of
its size: zero for a constant half unit).
- **Accept:** `btc_hedged` net mean day minus `neutral` net mean day, paired over the pooled days, CI above 0 on both
  seeds under cost model (b) or (c).
- **Refuted:** that CI holds 0 or is below 0 on either seed under (b) and (c).

## Addendum 5 (owner 2026-11-19, before any number of it): does more training data help? (flat split)
`panel-std-order-full-001` (all train rows, about 1.1 M) against `panel-std-order-400k-001` (train capped at 400000
rows, as in a walk-forward fold), arm `relative_rank_loss_logret_std_label_order`, the artifact's default split, seeds 0
and 1, one commit; the same test rows in both.
- **More data helps:** close rank IC on TEST higher with all rows on both seeds, by more than the seed spread of the
  capped job, and no high / low rank IC loss over 0.01.
- **Refuted:** close rank IC not higher on either seed. Reported beside it: high / low rank IC, the pooled day books
  (absolute, neutral) and the kept-row rules with their CIs, epochs to the best epoch (memorisation check).
- A single split is one market regime: a look, not a walk-forward result (`docs/PHILOSOPHY.md` section 2, step 5).
