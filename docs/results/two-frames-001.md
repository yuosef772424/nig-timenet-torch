# two-frames-001: the panel model with a 1h window beside the 4h window, walk-forward (owner 2026-11-18)

Pre-registered: `docs/audit/2026-11-18-panel-std/PREREGISTERED.md` addendum 2. `panel-rolling-2tf-001` (reference arm
+ the 32-bar 1h window of every row, `input_timeframes [4h, 1h]`, artifact `02f375cb3286` = c572b1b36960 + X_1h)
against `panel-rolling-ref-003` (same derived artifact, 4h input only), commit `03bafba`, 7 expanding folds x seeds 0
and 1, 1225 pooled TEST days per seed. Read from `rolling_folds.csv` and `pooled_days_seed_<s>/panel_report.md` on Drive
(the reading script did not run: the runtime stopped at about 01:03 UTC on 2026-11-19). Exploratory (rule C5).

**Reproducibility check passed:** `panel-rolling-ref-003` equals `panel-rolling-ref-002` digit for digit in every fold
(close rank IC fold 0 seed 0 0.10679 in both), so the derived artifact reproduces the 4h run exactly.

**Verdict: refuted on the close, a small consistent gain on high / low, no profit.**

| per fold x seed (14 runs) | 2tf | reference | difference |
|---|---|---|---|
| close rank IC, mean | 0.0858 | 0.0860 | -0.0002; above the reference in 3 / 7 folds (seed 0) and 2 / 7 (seed 1) |
| high rank IC, mean | 0.367 | 0.360 | **+0.007, higher in 14 / 14 runs** |
| low rank IC, mean | 0.448 | 0.441 | **+0.006, higher in 14 / 14 runs** |

| pooled TEST days, seed 0 / 1 | 2tf | reference |
|---|---|---|
| metric 1 (sign book) mean day net | -0.073 % / -0.072 % | -0.089 % / -0.090 % |
| metric 1 beats every baseline | no / no | no / no |
| metric 2 (neutral book) mean day net | -0.118 % / -0.113 % | -0.115 % / -0.118 % |
| relative rank IC | 0.0848 / 0.0868 | 0.0853 / 0.0867 |
| predicted up (realised 0.486) | 0.427 / 0.397 | 0.524 / 0.500 |
| top_confidence rows, metric 1 net | +0.031 % / +0.042 % | +0.045 % / +0.050 % |

## What the model learned
The 1h window carries information about the RANGE of the next 8 h: the high and low rank IC rise in every one of the 14
fold x seed runs (+0.006 to +0.009), above the run-to-run noise (the walk-forward is deterministic: ref-003 = ref-002
exactly). The close ranking does not move (relative rank IC 0.085-0.087 in both).

## Where it is right and where it fails
On the pooled days the absolute (sign) book loses less (-0.07 % against -0.09 % a day) and the neutral book is the
same; neither beats every baseline. The error map by groups from the frames was not computed (the reading script
`read_std_rolling.py` writes it to `panel-rolling-2tf-001/reading_vs_panel-rolling-ref-003/` when a runtime is up).

## The samples it failed on
Not yet read (same reading script: `worst_rows_cand`, `worst_days_cand`).

## Bias
The 1h input makes the close mean lean SHORT: predicted up 0.427 / 0.397 against a realised 0.486 (the reference leans
slightly long, 0.524 / 0.500). Part of the smaller sign-book loss may be this lean in a period with more down rows; a
VAL re-centring is in the reading script (`recenter_cand`).

## Selective use
The panel report's kept-row rules: `top_confidence` stays positive after cost (+0.031 / +0.042 %), but below the
reference's (+0.045 / +0.050 %); no CI in this report (the CI columns were added after `03bafba`).

## Why it failed
Named cause: **the 1h window describes the shape of the last 4h bar (where inside it the move happened), which is
volatility information**, so it helps the heads that predict the range (high / low) and not the one that ranks the
close: the close ranking comes from the cross-section (relative inputs, rank term), which the 1h window of the same
coin does not add to. Evidence: high / low up in 14 / 14 runs, close unchanged, relative rank IC unchanged.

## Next hypotheses
1. Use the 1h frame where it helps: the limit-order uses of the high / low heads (orders at the predicted high / low,
   battery section of T-profit). Refuted if the limit-order book of 2tf does not beat the reference's on the pooled days.
2. The close lean: re-centre the close mean on VAL (the reading's `recenter_cand`); refuted if the sign book's net does
   not improve on TEST.
