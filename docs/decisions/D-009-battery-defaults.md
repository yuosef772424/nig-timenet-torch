# D-009: battery and T-profit defaults (package I5)

Date: 2026-11-07
Owner decision: package I5 of `docs/hardening/PLAN.md` (rule B1 / T1: a new default needs a decision file and a pin);
cost, quantiles and percents come from the owner's `feat/ar-stream` work (commits 7c9736b, f052cf6, D-006 there). T-profit
(owner 2026-11-07) adds the profit sections.
Why: `eval/battery*.py` is ported as is, so its defaults are the ported ones: cost 0.0008, top percents 0.05 / 0.10 / 0.20,
VAL quantiles 0.3 / 0.5 / 0.7, 5 shock groups. New in this package: `battery_profit` charges the cost PER SIDE
(`DEFAULT_SIDES = 2`, entry and exit), the CI level is `1 - 0.05 / n_variants`, seed 0; the older ported sections charge the
cost ONCE per trade (see `docs/suggestions.csv`: the owner may want one convention). The owner may correct 0.0008.
`stats.metrics.block_bootstrap_ci` (new metric, C2) defaults: level 0.95, 500 resamples, seed 0.

- `nig_timenet_torch.eval.battery.battery(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.eval.battery.confidence_buckets(n_buckets)`: `<new>` -> `5`
- `nig_timenet_torch.eval.battery.flip_test(pct)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.battery.full_battery(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.eval.battery.full_battery(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.eval.battery.limit_execution(pcts)`: `<new>` -> `(0.05, 0.1, 0.2)`
- `nig_timenet_torch.eval.battery.per_coin(min_n)`: `<new>` -> `30`
- `nig_timenet_torch.eval.battery.per_coin(pct)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.battery.shock_split(n_groups)`: `<new>` -> `5`
- `nig_timenet_torch.eval.battery.shock_split(pct)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.battery.show(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.eval.battery.trade_grid(pcts)`: `<new>` -> `(0.05, 0.1, 0.2)`
- `nig_timenet_torch.eval.battery.uncertainty_filter(pct)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.battery.uncertainty_filter(quantiles)`: `<new>` -> `(0.3, 0.5, 0.7)`
- `nig_timenet_torch.eval.battery.weekly_stability(min_n)`: `<new>` -> `50`
- `nig_timenet_torch.eval.battery.weekly_stability(pct)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.battery_adapter.to_battery_table(prev_close)`: `<new>` -> `None`
- `nig_timenet_torch.eval.battery_classics.market_neutral(quantiles)`: `<new>` -> `(0.2, 0.3, 0.4)`
- `nig_timenet_torch.eval.battery_classics.persistence_baseline(pct)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.battery_classics.selective_evaluation(quantiles)`: `<new>` -> `(0.25, 0.5, 0.75)`
- `nig_timenet_torch.eval.battery_profit.DEFAULT_SEED`: `<new>` -> `0`
- `nig_timenet_torch.eval.battery_profit.DEFAULT_SIDES`: `<new>` -> `2`
- `nig_timenet_torch.eval.battery_profit.limit_highlow(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.eval.battery_profit.limit_highlow(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.battery_profit.limit_highlow(sides)`: `<new>` -> `2`
- `nig_timenet_torch.eval.battery_profit.profit_close(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.eval.battery_profit.profit_close(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.battery_profit.profit_close(sides)`: `<new>` -> `2`
- `nig_timenet_torch.eval.battery_profit.profit_sections(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.eval.battery_profit.sigma_sizing(n_variants)`: `<new>` -> `1`
- `nig_timenet_torch.eval.battery_profit.sigma_sizing(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.battery_profit.sigma_sizing(sides)`: `<new>` -> `2`
- `nig_timenet_torch.stats.metrics.block_bootstrap_ci(level)`: `<new>` -> `0.95`
- `nig_timenet_torch.stats.metrics.block_bootstrap_ci(n_boot)`: `<new>` -> `500`
- `nig_timenet_torch.stats.metrics.block_bootstrap_ci(seed)`: `<new>` -> `0`
