# Audit: the samples the earlier models failed on, read on RAW charts (owner task 2026-11-17)

Exploratory (rule C5): 12 rows chosen BY THEIR OUTCOME from one report, so nothing here is a rate or a result. It
names what the failures look like, which input could have shown them, and what to measure next.

## Question

The earlier reports list their worst trades as numbers only (`docs/results/baseline-001.md` section 5: shorts that lost
the label clip, -172 %). What did the market do before and during those rows, on the prices BEFORE any normalisation,
and what separates them from the best rows of the same rule?

## Data and method

- Rows: the 9 worst and 3 best rows named in `docs/results/baseline-001.md` (run `baseline-002-s0`, artifact
  `c572b1b36960`, rule "SHORT top 5-10 % |mu|"; VAL and TEST). The list is in `samples.csv`, with the source line of
  each row. `panel-001.md` and `deep-side-001.md` name no single rows. The prediction frames on Drive
  (`panel_frame_test.parquet`, about 10 MB each) could not be read from this session: the Drive tool returns a file as
  base64 into the context. The reports give a DATE only, so the event is the UTC day and the exact entry bar is
  unknown.
- Candles: the public Binance futures archive (`data.binance.vision`, the source of the TF data), 4h and 1h, read by
  `eval/sample_charts.py` (`python -m nig_timenet_torch.eval.sample_charts samples.csv charts/`). Each chart has the
  64 bars of 4h before the day, the day, the quote volume, and the day +- 1 day in 1h bars, on a log price scale. The
  numbers are in `charts/facts.csv`.
- Reading: three Haiku agents read the 12 charts with a fixed form and no judgement of the model
  (`haiku_readings.md`). The lead checked HIGHUSDT and BTRUSDT against the 1h candles. The start hours and levels
  match. The peak hour of HIGHUSDT is off by about 3 h (Haiku put it at 10:00-12:00, the candles at 07:00), so the
  hours in the readings are approximate.

## The samples it failed on

| coin | day | reported | 4h vol of the window | largest 4h bar in window | 7 d before | event-day volume / 30 d median | largest 8 h rise in the day | what the chart shows (Haiku, checked) |
|---|---|---|---|---|---|---|---|---|
| AIOTUSDT | 2026-04-02 | -172 % | 5.3 % | 26 % | +15 % | 21x | +231 % | quiet base after a month-long decline; volume burst 1-2 days before; vertical run in the last hours of the day |
| HIGHUSDT | 2026-04-18 | -172 % | 3.2 % | 22 % | +11 % | 1207x | +139 % | flat dead base (0.115-0.125); staircase pump from 19:00 UTC the day before to 4x |
| DAMUSDT | 2026-04-27 | -158 % | 5.0 % | 27 % | -42 % | 179x | +293 % | downtrend, then one vertical candle; 80 % given back the next day; no warning |
| PORTALUSDT | 2026-05-31 | -147 % | 3.5 % | 21 % | +47 % | 270x | +201 % | flat base, volume ramp and breakout 12-24 h before |
| LSKUSDT | 2026-09-12 | -172 % | 2.6 % | 12 % | +38 % | 559x | +833 % | 8-day flat base, slow grind, then staircase and a 10x spike the next night |
| TAIKOUSDT | 2026-07-01 | -172 % | 5.3 % | 36 % | +15 % | 885x | +322 % | base after a decline; staircase then a vertical leg; full retrace the next day |
| BTRUSDT | 2026-08-26 | -172 % | 4.2 % | 24 % | +12 % | 628x | +208 % | 10 days flat at near-zero volume; 5x in 18 h; held |
| SIRENUSDT | 2026-03-22 | day -217 | 6.9 % | 39 % | +61 % | 20x | +176 % | uptrend continuation, wick-heavy squeeze |
| STOUSDT | 2026-04-01 | day -234 | 7.2 % | 39 % | +166 % | 183x | +122 % | rally continuation, late blow-off and dump |
| **best:** POWERUSDT | 2026-03-03 | day +184 | 10.0 % | 44 % | +273 % | 10x | +22 % (fall -84 %) | collapse after a parabolic run-up |
| **best:** VELVETUSDT | 2026-06-12 | day +156 | 11.2 % | 54 % | +1174 % | 518x | +22 % (fall -75 %) | collapse after a parabolic run-up |
| **best:** TAIKOUSDT | 2026-07-02 | day +168 | 17.2 % | 252 % | +624 % | 964x | +22 % (fall -67 %) | fade of the previous day's pump |

## What the failures have in common (on these 12 rows only)

1. **All 9 losses are short squeezes of thin coins.** Price rose 2x to 10x within hours, on 20x to 1200x the usual
   volume. Seven of the nine started from a quiet, flat base: 4h volatility of the window 2.6-5.3 % (median 4.3 %),
   largest bar about 25 %. The other two (SIREN, STO) continued a rally.
2. **The 3 best shorts are the opposite case.** They came after a parabolic run-up (+273 % to +1174 % in 7 days, window
   volatility 10-17 %), and the short caught the collapse. The same "fade the move" rule wins after a blow-off and
   loses in a breakout from a dead base. On these rows the two groups do not overlap on window volatility (losses
   <= 7.2 %, wins >= 10.0 %) or on the 7-day return before (losses <= +166 %, wins >= +273 %). With 12 rows picked
   by outcome, this is a hypothesis, not a filter.
3. **The visible warning is volume, and the model cannot see it.** The readers saw volume rise before the move in 7 of
   the 9 losses, 4 h (HIGH) to 2 days (AIOT) ahead. There was none in DAM and only a weak bump in BTR. The artifact holds high / low / close
   log moves only (B-055): volume, quote volume and trade counts are not inputs. The base itself (a long flat stretch)
   is visible in the inputs.
4. **The reported losses are clipped.** Five of the seven single trades sit at -172 % = e^1 - 1, the label clip
   |log| <= 1 (B-054).
   The largest 8 h rise from a bar close inside those days was +139 % to +833 %, so the real tail of the short side
   is heavier than any report shows.
5. **The fall days are fast.** The wins came from -67 % to -84 % in two 4h bars. The rule's P&L is a few dozen
   coin-days with moves of 100 % and more on either side: one squeeze or one collapse outweighs hundreds of ordinary
   trades. This matches `baseline-001.md`: "one to three days decide the sign".

## Why the shorts failed (one named cause, evidence level stated)

**Cause (suspected, not verified):** the short side of the top-|mu| rule is a mean-reversion bet on coins that have
just moved. It cannot tell a blow-off top (reverts) from a breakout of an illiquid coin (keeps going). The
information that separates them in these charts is volume, which the model does not get, and the length and
quietness of the base before the move.

Verifying needs the exact rows: the bar each trade was entered at, its mu, and the window the model saw. The reports
keep a date only, and the frames that hold the rows are on Drive. `workflow/legacy_report.py` now writes
`legacy_samples.csv` with the exact `end_time` of the worst and best trades, so the next run can be charted bar by
bar (`time=True`).

## What was NOT verified

- The exact entry bar of each loss (the date only). The 8 h move is the largest inside the day: an upper bound for the
  trade.
- That the model's mu on these rows came from the earlier rise (no access to the rows' inputs and mu).
- Any rate: 12 rows picked by outcome. The separation in point 2 can be chance or selection.
- The Haiku readings beyond the two coins the lead checked; their hours are +- 3 h.

## Next hypotheses (each with what would refute it and its cheapest test)

| # | Hypothesis | Refuted if | Cheapest test |
|---|---|---|---|
| H1 | The top-|mu| short side loses on breakouts from quiet bases and wins after blow-offs: a filter "short only when the window 4h volatility > Q and the 7-day return > R", fitted on VAL, improves TEST net per trade | on the battery tables of `baseline-002-s0` (Drive) the VAL-fitted filter does not raise the TEST net per trade outside its block CI, or keeps < 2 % of rows | `legacy_report` + a filter column on the saved battery tables, Colab CPU, minutes; the rule is written before TEST is read |
| H2 | Volume (quote volume, its ramp over 12-72 h) is the missing input for squeezes | a ridge on window features + volume ramp does not rank the 8 h max-up of the next bars better than the same ridge without volume (rank IC, day-block CI) on VAL | export quote volume per window (B-055 fix idea) on a 50-coin slice, then a model-free ridge (`docs/audit/2026-11-16-panel/normalisation_look.py` pattern) |
| H3 | The clipped label hides the short side's tail: with unclipped realised returns the short top-|mu| rows lose more than reported | the unclipped net of the same trades is within the CI of the clipped net | B-054 fix idea: evaluate from `last_candles` prices; one pass over the saved tables |
