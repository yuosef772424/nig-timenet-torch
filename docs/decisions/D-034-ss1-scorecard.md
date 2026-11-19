# D-034 — Short-side reading + day scorecard in the battery (owner 2026-11-19)

Date: 2026-11-19
Owner decision: owner message 2026-11-19: "read the reports of the past experiments and what was learned from them,
and make the evaluation take into account the confidence, the distribution of the losses over the days, the size of
the uncertainty and the top; and open a report on why every run's top short lost, checking whether the loss function
computes the short exponentially" (translated from Arabic). Confirmed by merging this PR.
Why: a per-trade Sharpe hides that one to three days decide the sign of a book (baseline-001) and that a side can win
as often as the other and still lose (the short pays the exp() part of 1 - e^l: `docs/results/short-side-001.md`);
the battery judged its verdict on the best per-trade Sharpe only.

## What changes behaviour
- The battery (`eval.battery.full_battery`, so `battery_report.txt` and `battery_from_folder`) gains four sections
  after the T-profit sections and before the verdict: SCORECARD (selection x side by days), UNCERTAINTY AND
  CONFIDENCE BUCKETS, LONG vs SHORT PAYOFF (log / exp split), SCORECARD VERDICT; `battery_profit.json` gains the key
  `scorecard_verdict`. No existing section or number changes; the old `verdict` is unchanged.
- The three new defaults reproduce the battery's existing conventions: the bootstrap seed 0 (as `battery_profit`'s
  `DEFAULT_SEED`) and one variant tried (as `full_battery(n_variants=1)`).

- `nig_timenet_torch.eval.battery_scorecard.day_profile(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.battery_scorecard.scorecard(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.eval.battery_scorecard.scorecard_sections(n_variants)`: `<new>` -> `1`
