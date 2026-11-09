# D-020: the deep-side check (new derived target, read-only) and two new metrics

Date: 2026-11-09
Owner decision: owner task of 2026-11-08 ("deep side" direction, read-only, baseline-002-s0 weights): measure "the side
where price goes deeper over the horizon" (deep = log(fh / lc) + log(fl / lc), predicted mu_high + mu_low) against every
baseline with a day-block CI on VAL and TEST; reusable tool with a card and a test; no default changes.
Why: a new tool needs defaults. None of them changes an existing behaviour: `stats.metrics.rank_ic`, `partial_rank_ic`
(1-D baseline), `rank_ic_se` and `within_group_spearman` give bit-identical results on the real TEST table (checked
against `origin/main`); a 2-D baseline in `partial_rank_ic` is new (it raised before).

## What changes behaviour
- Nothing that existed. New: `eval/deep_side.py`, `workflow/deep_side_run.py`, `stats.metrics.group_rank_ics`,
  `stats.metrics.auc_block_ci`, `partial_rank_ic` accepting several baselines, `workflow.reading.stage_data` (was the
  private `_stage`; now also returns the last input bar `x_last`).
- Thresholds of the check: volatility bucket edges = VAL quintiles of ATR-14; near-tie margin = VAL 20 % quantile of
  |deep|; baseline signs from VAL; CIs = 500 day-block resamples, seed 0.

## New defaults
- `nig_timenet_torch.eval.deep_side.clip_counts(level)`: `<new>` -> `1.0`
- `nig_timenet_torch.eval.deep_side.close_link(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.deep_side.deep_side(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.deep_side.near_ties(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.deep_side.near_ties(share)`: `<new>` -> `0.2`
- `nig_timenet_torch.eval.deep_side.score_columns(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.deep_side.skill(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.deep_side.skill(with_auc)`: `<new>` -> `True`
- `nig_timenet_torch.eval.deep_side.vol_buckets(n)`: `<new>` -> `5`
- `nig_timenet_torch.eval.deep_side.vol_buckets(seed)`: `<new>` -> `0`
- `nig_timenet_torch.stats.metrics.auc_block_ci(level)`: `<new>` -> `0.95`
- `nig_timenet_torch.stats.metrics.auc_block_ci(n_boot)`: `<new>` -> `500`
- `nig_timenet_torch.stats.metrics.auc_block_ci(seed)`: `<new>` -> `0`
- `nig_timenet_torch.stats.metrics.group_rank_ics(baseline)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.deep_side_run.deep_side_step(seed)`: `<new>` -> `0`
