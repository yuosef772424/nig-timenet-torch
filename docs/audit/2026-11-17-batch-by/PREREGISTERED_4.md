# PREREGISTERED (4) — the H3 test again, without the normalisation layer (owner 2026-11-18)

Written before any number of this study was read. Owner's request (Arabic, 2026-11-18): "try again, removing the
normalisation layer". Exploratory (rule C5).

## Hypothesis

The per-window normalisation (`InstanceNorm`: z = (x - window mean) / window std) erases each window's volatility
level before the encoder; the model gets the level back only through the 6-number stats path and the level
passthrough, and it does not reach the model-free bound (studies 2 and 3). **Without the normalisation the encoder
sees the level directly and the model reaches the bound.** One change only: version **N** = the run default with
`use_instance_norm = False` (`InstanceNorm(normalize=False)`: z = the train-scaled input, soft clip and affine and
the stats path unchanged); NOT combined with `har_range` (one change per version, PHILOSOPHY section 4).

## Design (as PREREGISTERED_3, 4 trainings)

Same data (60-coin slice), same single split (train 2023, test 2024, inner validation = last 15 % of 2023), versions
**A0** (run default) and **N**, seeds 0 and 1, 4 epochs, lr 1e-3, best epoch by `val_raw_loss`, battery off, Colab
L4, all four on one commit. Gates of rule C6 not run (owner: few models); recorded as a deviation.

## Metric, bound and accept rule (the rules of PREREGISTERED_3 with H read as N)

Rank IC by timestamp of the high and low heads per quarter of 2024, mean of the 2 seeds; bound = the better of the
6-bar mean range and the ridge on the HAR features fitted on 2023. **Supported if BOTH:** (a) N >= the bound in at
least 3 of 4 quarters on both heads; (b) N > A0 in at least 3 of 4 quarters on both heads and N > A0 on every seed
and head over the whole year. Refuted otherwise. Read, not judged: the best epoch (the memorisation of study 3), the
gap closed, the test raw loss, the close head and its bias, coverage.
