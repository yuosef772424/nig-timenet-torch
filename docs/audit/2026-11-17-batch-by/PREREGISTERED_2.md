# PREREGISTERED (2) — the two hypotheses of batch-by-wf-001 on 60 coins (owner 2026-11-18)

Written before any number of this study was read. Owner's request (Arabic, 2026-11-18): "test the two hypotheses on
60 coins". The hypotheses are the "Next hypotheses" of `docs/results/batch-by-wf-001.md`. Exploratory (rule C5):
public klines, not the owner's artifact; Colab GPU (L4) through the bridge.

## Data (the same definition as study 1, three times the coins)

`data_io/vision_slice.py` with its 60 DEFAULT_COINS (liquid, unsealed), 4h, 2023-01 .. 2024-12, stride 2, window 32,
horizon 2. Coins listed after 2023-01 (e.g. ARB, SUI) enter when their history starts. Walk-forward geometry as study
1: expanding, first train window 365 days, test 90 days, step 90 days, embargo 1 day, 4 folds (test = 2024 Q1..Q4),
inner validation = last 15 % of each train window; run-default model, AdamW lr 1e-3, 4 epochs, best epoch by
`val_raw_loss`; battery off. **3 seeds (0, 1, 2)**: 12 fold x seed pairs per comparison.

## H1 — one-timestamp batches help the cross-sectional ranking of the low head

Arms: A0 = rows, batch 256 (default); A2 = timestamp (one batch = all ~60 coins of one timestamp); C60 = rows, batch
60 (the size control of A2: same steps per epoch). A1 (day) is NOT run.

Metric: the low head's rank IC by timestamp on the fold's test window (`targets.low.skill.rank_ic_by_timestamp` of the
test report), per fold x seed.

Accept rule (H1 **supported**) — BOTH comparisons A2 vs A0 and A2 vs C60 must pass:
(a) A2's low IC is higher in at least 9 of the 12 fold x seed pairs, and
(b) the mean over the 4 folds of the paired difference (seeds averaged per fold) has a 95 % t-interval (3 df) above 0.
H1 is **refuted** if either comparison's interval includes 0. Read but not judged: the high head's IC (same rule,
reported), the test raw loss (pre-registered rule of study 1, reported), coverage, CPU / GPU time.

## H2 — the model under-reads the range information already in its inputs

Model-free bound on the SAME test rows of every fold: (i) the 6-bar mean range of the input window (sign + for high,
- for low), (ii) a ridge regression on `doctor/common.py::simple_features` (the HAR range terms and the per-channel
means / sds over 1 / 6 / 32 bars) fitted on the fold's TRAINING rows (penalty chosen on the fold's inner validation
from {1, 10, 100, 1000, 10000}), scored by rank IC by timestamp against the high and low labels.

Accept rule (H2 **supported**): the better of (i) and (ii) has a higher rank IC than A0 (mean of its 3 seeds) on BOTH
the high and the low head in at least 3 of the 4 folds. H2 is **refuted** if on either head the model is at or above
the better baseline in 2 or more folds. Read alongside (not judged): the same comparison for A2 and C60, and the
doctor's information-flow probe (`python -m nig_timenet_torch.doctor <artifact> --cards A`, its own short run on the
default split) for where along the model the range signal is lost.

## What follows

H1 supported -> a GPU walk-forward on the owner's artifact (smoke first, rule E5) with A0 / A2 / C60. H2 supported ->
locate the loss with the probe, then ONE switch (e.g. an input skip of the range terms) through the gates (rule C6).
Neither supported -> the failure reading of PHILOSOPHY section 3 in the results note. No default changes from this
study.
