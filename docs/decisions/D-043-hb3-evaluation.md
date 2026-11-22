# D-043 — Hardening 2026-11b, package HB3: evaluation, one implementation per trading rule and helper (owner 2026-11-21)

Date: 2026-11-22
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB3); owner message 2026-11-22: "continue the work, preferably with Sonnet agents for each task"
Why: the audit `docs/hardening/audit-2026-11b/B_eval_doctor.md` (rows B-04..B-11, B-20, B-34, B-35, re-verified by the agent against the code) found the battery's shared API exported as private names, the round-trip cost and the market-exit PnL re-derived in about 15 places, two copies of the limit-at-extreme rule, three "top share by |mu|" selectors (two with an unstable sort), six day / block helpers, three Bonferroni levels, the minimum trade count 50 and the cost 0.0008 typed in 8 and 15+ places, the direction rule `p_dir > 0.5` 24 times, and a day-block CI label on a 50-row fallback.

## What changes behaviour
- Numbers: the "top share by |mu|" selection uses a stable sort everywhere (`battery_lib.top_abs`, B-08): rows TIED on
  |mu| may be selected differently (an equal-output harness over 13 frames: only the frames built with ties differ,
  in the sections that select by |mu|). The T-profit note says "50-row blocks (no end_time)" when that fallback is
  used (B-35). Every other battery, strategy book, legacy, day-metric, uncertainty-use and deep-side output is equal.
- Names: the battery helpers are public (`trade_stats`, `scored_rows`, `top_abs`, `direction_pnl`, `section`,
  `format_row`, `HEADER`; no aliases); new `net_pnl`, `bonferroni_level` (+ `ALPHA`), `trade_side`,
  `limit_at_extreme`, `MIN_TRADES`, `eval/day_blocks.py` (`calendar_day`, `day_blocks`, `DAY_BLOCK_DAYS`), and the cost
  default `core/costs.py::DEFAULT_COST_PER_SIDE` (0.0008, read by the trainer's side-payoff loss, the run settings,
  the battery and the reports; a scan test refuses a typed 0.0008 elsewhere).
- Removed (dead): the `sides` parameters of `battery_profit` and `DEFAULT_SIDES` (no caller passed them; the value 2
  is `battery_lib.SIDES`).

- `nig_timenet_torch.core.costs.DEFAULT_COST_PER_SIDE`: `<new>` -> `0.0008`
- `nig_timenet_torch.eval.battery_lib.bonferroni_level(alpha)`: `<new>` -> `0.05`
- `nig_timenet_torch.eval.battery_lib.section(note)`: `<new>` -> `''`
- `nig_timenet_torch.eval.battery_lib.top_abs(mask)`: `<new>` -> `None`
- `nig_timenet_torch.eval.battery_lib.trade_stats(min_n)`: `<new>` -> `50`
- `nig_timenet_torch.eval.battery_profit.DEFAULT_SIDES`: `2` -> `<removed>`
- `nig_timenet_torch.eval.battery_profit.limit_highlow(sides)`: `2` -> `<removed>`
- `nig_timenet_torch.eval.battery_profit.profit_close(sides)`: `2` -> `<removed>`
- `nig_timenet_torch.eval.battery_profit.sigma_sizing(sides)`: `2` -> `<removed>`
