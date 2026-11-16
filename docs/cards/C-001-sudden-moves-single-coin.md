# Card C-001: the single-coin trainer and sudden moves (owner request 2026-11-16; to be executed later)

Status: card only, nothing built. ROADMAP "Now" step N4. Follows `docs/PHILOSOPHY.md` section 2: its first step is a
measurement, not a switch.

## The owner's idea

Improve the single-coin trainer so that sudden coin moves, liquidity pumps, crashes and news are treated on purpose.
Warning from the owner: penalising every such case the same way would confuse the model and make it lose good weights.

## What the idea becomes when it is made measurable

A small share of rows carries a large share of the loss. Three kinds of rows look the same to the loss and are not:

| class | what it is | what the model can know | what a uniform penalty would break |
|---|---|---|---|
| market shock | every coin moves together at that timestamp (a crash, macro news) | nothing about the sign from one coin's window; the size partly (volatility clusters) | the scale heads must still learn that these rows are wide |
| specific jump | one coin moves far beyond its own recent volatility and beyond the market (a pump, a listing, coin news) | the sign rarely; an "event risk" sometimes (run-up, range expansion) | the mean is dragged toward a level bet for that coin |
| volatile regime | large moves that the recent bars announce | the size: this is where the high / low heads earn their rank IC (0.30 .. 0.42 on `panel-001`) | clipping or down-weighting large losses removes exactly this signal |

So the question is not "penalise sudden moves" but: **on which rows should the gradient of the MEAN be trusted less,
while the gradient of the SCALE is kept?**

## Step 0: measure before any switch (CPU and one trained checkpoint; no training)

Per row, from the artifact only (labels, the 32-bar input window, the timestamp): `z_own` = |close label| over the
realised volatility of the coin's window; `m_t` = the mean close label of the timestamp over its coins and `z_mkt`
= |m_t| over its trailing volatility; `z_idio` = |label - beta_i * m_t| over the coin's residual volatility (beta and
volatilities from TRAIN rows only). Classes: market shock (`z_mkt` > 3), specific jump (`z_idio` > 4 and not a market
shock), the rest.

| # | Measurement | Tool | It decides |
|---|---|---|---|
| S1 | share of rows, of the total loss and of the trunk's gradient norm per class | `trainer/grad_report.py` on a trained checkpoint, batches drawn per class | whether there is anything to fix: go on only if a class with < 2 % of the rows holds > 15 % of the mean's gradient |
| S2 | per class: does the predicted sd rank the absolute error, and is the 80 % interval covered | `eval/uncertainty_use.py` | whether the scale already explains the class (then leave it alone) |
| S3 | per class: the mean's sign error and the direction bias it leaves on the calm rows that follow | day metrics on the rows after an event | whether events are where the level bet (B-075) comes from |
| S4 | the same three numbers on a synthetic set with PLANTED jumps of both kinds | a jump process added to E1 / E4 (off by default) | the known answer: what a model that ignores the jump's sign scores |

News cannot be recognised as news: the artifact has no news and no volume (3 log-move channels). A news row is
visible only through its footprint (classes above). A volume channel is a data change and a separate card.

## Arms (only after step 0 says go; one change per arm, every switch off by default, rule B1)

| arm | change | predicted movement |
|---|---|---|
| A0 | the single-coin model as it is (`rolling-single-001`) | the reference |
| A1 | market-shock rows: the mean's gradient weighted down, the scale's kept (the NLL evaluated with the mean detached for the down-weighted part) | smaller direction bias and train / val gap; high / low rank IC unchanged |
| A2 | the same for specific jumps | the per-coin predicted-up range narrows (today 0.30 .. 0.70 and wider) |
| A3 | an auxiliary "event risk" output (probability of a class row) instead of touching the loss | the sd ranks the error better across time; the mean unchanged |
| A4 | the uniform control: one robust loss for every large error (the thing the owner warns against) | high / low rank IC and coverage fall; kept as the arm that tests the warning |

Order: E1 / E4 with planted jumps (known answer), the real long slice (2 seeds, a look), then the walk-forward folds
of `rolling-single-001`. Accept rule, written before any number: on both seeds and on the pooled walk-forward days the
close relative rank IC and the high / low rank IC are not below A0 beyond the seed spread, coverage stays in its
band, and the mean day net of metric 1 or metric 2 is above A0 with a paired day-block CI that excludes 0.
