# D-049 — Take-profit at k x ATR in the battery, beside the take-profit at mu (owner 2026-11-22)

Date: 2026-11-22
Owner decision: owner message 2026-11-22 (on the battery's take-profit code): "add another version, but based on the real average ATR instead of mu, so the targets are set at given points of the ATR or its multiples"
Why: the battery judged a take-profit only at the model's own mu (`take_profit_grid`); a target at multiples of the coin's recent volatility is the trader's usual reference and tells whether the model's magnitude adds anything to the volatility itself.

## What changes behaviour
- New section `take_profit_at_atr` (`eval.battery.take_profit_atr_grid`, after `take_profit_at_mu` in
  `SECTION_ORDER`): the same rows as the take-profit at mu (top 5 / 10 / 20 % by |mu| on each side, side from p_dir),
  market entry at the last close, a take-profit at k = 0.5 / 1 / 1.5 / 2 x the row's ATR on the trade's side
  (`battery_lib.take_profit_at_atr`; reached on the realised high / low, else exit at the close), with the median
  |mu| / ATR of the rows. Every existing section and number is unchanged; the battery report gains this section and
  `battery_table_<stage>.parquet` gains the column `atr`.
- The ATR (`workflow.battery_run.stage_atr`, log units per bar, only bars closed by the prediction time): the mean
  true range of the last 14 input bars when the window has high / low price channels, else the window's last
  `NATR_14` (Wilder ATR / close, stored / `data.scaling.PERCENT_SCALE` = 10, a named constant now, value unchanged).
  The ATR is per bar while the label spans the run's horizon: k counts bars' ATR.
- Battery tables saved before this change have no `atr` column: `battery_from_folder` on them prints the section as
  SKIPPED.


- `nig_timenet_torch.eval.battery.take_profit_atr_grid(multiples)`: `<new>` -> `(0.5, 1.0, 1.5, 2.0)`
- `nig_timenet_torch.eval.battery.take_profit_atr_grid(pcts)`: `<new>` -> `(0.05, 0.1, 0.2)`
- `nig_timenet_torch.eval.battery_adapter.to_battery_table(atr)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.battery_run.StageInputs(atr)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.battery_run.stage_atr(timeframes)`: `<new>` -> `()`
