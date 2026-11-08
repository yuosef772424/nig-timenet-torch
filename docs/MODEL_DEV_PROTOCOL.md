# MODEL DEVELOPMENT PROTOCOL — binding for every agent that touches the model (owner, 2026-11-07)

Rule C6 in `docs/RULES.md` makes this file binding. It applies to every PR or job that changes or trains the model:
`model/`, `trainer/`, the loss/target/codec code (`trainer/tasks.py`, `data_io/target_codec.py`), the split
(`data_io/split_plan.py`), the target metrics (`eval/`, `stats/metrics.py`), and every training job beyond the smoke
job. Reading order for such a session: `docs/STATUS.md`, `docs/RULES.md`, this file, then ONLY your ROADMAP step.

Ported in spirit from the TF protocol (`crypto-signal-prediction/docs/research/MODEL_DEV_PROTOCOL.md`: card before the
run, per-layer health report, fixed failure-location order) and its research rules (localize -> test the cause -> fix
-> validate). What is new here: the gates run BEFORE any full training, in fixed test environments, on >= 3 seeds.

Contents: 1 definitions; 2 the gates (A gradient, B confidence, C uncertainty) and the direction rule (D); 3 the test
environments; 4 the confidence and uncertainty methods and how to choose; 5 the memorization study; 6 after training
(model doctor); 7 evidence and records.

---

## 0. The goal (owner decision 2026-11-07)

The north star is **T-profit** (ROADMAP "Targets"): profit after realistic costs from ANY use of the model outputs,
measured by the evaluation battery through `eval/battery_adapter.py`, per seed, with trivial baselines and a
dependence-aware CI, every variant tried counted toward multiple testing, exploratory until the owner asks for strict
confirmation (C5). T-skill, T-conf and T-sharp below are MEANS: diagnostics that explain profit. The gates in
section 2 are health checks that run BEFORE a full training (they prevent wasted GPU time); passing them is not
success and failing a diagnostic gate is not by itself a rejection of a model that shows T-profit. A profit-based
selection metric is a T6a candidate (not built yet; selection stays `val_raw_loss` until then, B-003).

## 1. Definitions

- **Full training**: any run on the full artifact longer than one epoch or 30 GPU minutes. Everything shorter (smoke,
  gate runs, fractional-epoch study arms) is a **short run**.
- **Configuration**: commit + model settings + trainer settings + data artifact fingerprint (the `RunIdentity` settings
  hash). Gates are passed by a configuration, not by an idea: a changed switch is a new configuration.
- **Head**: the three NIG regression heads (`close`, `high`, `low`) and their class heads. "Per head" always means all
  three regression heads, each reported separately.
- **Predicted move** of a head: `|pred_price - last_close|`, where `pred_price` is the head's decoded prediction
  (`TargetCodec.decode`) and `last_close` the last close of the input window (for the high head: the predicted high
  minus the last close, and so on).
- **Predictive std** of a head: the NIG predictive std `sqrt(beta (1 + nu) / (nu (alpha - 1)))` mapped to PRICE units
  through the codec (`decode(mu + std) - decode(mu)`), or the std of whatever method a switch registers.
- **Representative batch set**: `R = 16` batches of the job's batch size drawn from the TRAINING split with a fixed
  seed, stratified so every batch holds rows from >= 32 coins (all coins if fewer) spread over the whole training time
  range. Never "the first rows": the dataset is coin-major, so the first rows are one coin's earliest windows (H0
  audit, `trainer/trainer.py:300-301`).
- **Seed set**: >= 3 seeds (0, 1, 2 unless a card says otherwise). A gate passes only if it passes on EVERY seed.

## 2. The gates (ordered; each runs on every environment of section 3 before the next gate)

A configuration may start a full training only when `gate_report.json` for its settings hash shows gates A, B and C
passed on environments E1, E2 and E3 for every seed. Exception: the **baseline arm** (today's default) is trained as
the reference even when it fails a gate; its failures are recorded in the result file. Until ROADMAP step G2 builds
the job-side refusal, the job card names the gate report (manual check at review). Since G2 a training job calls
`workflow.gate_run.gate_check` after building the model: it records the gate status in the run summary and refuses a
full training of a non-baseline configuration without a passing report of the same configuration key (D-019).

### Gate A — gradient health

Measured with the planned optimizer and LR on the representative batch set at step 0 and again at the end of a short
run (E1: 3 epochs of the synthetic set; E3: 1/4 epoch of the small real sample). Tool: `trainer/grad_report.py`
(built in G2; gradient code must live in `trainer/`, rule G5). Every number below is per parameter tensor unless
stated; "layer" = a top-level submodule (input projection, each encoder layer, readout, `linear_path`, each head).

| # | Check | Pass | Fail |
|---|---|---|---|
| A1 | finite gradients and loss | all finite on all R batches | any NaN / inf |
| A2 | dead parameters | every trainable tensor has grad norm > 1e-12 on at least one batch | a tensor with zero gradient on all R batches (unless the card lists it, e.g. a disabled head) |
| A3 | vanishing / exploding per layer | per-layer RMS gradient (`norm / sqrt(numel)`) within [1e-3, 1e3] x the median layer | outside that band; or a global-norm spike > 100 x its running median during the short run (10 x = warn) |
| A4 | update / weight ratio | median over tensors of `norm(delta w) / norm(w)` for one optimizer step in [1e-4, 1e-2] | median outside [1e-5, 1e-1], or any weight matrix < 1e-6 (frozen in practice) or > 1e-1 |
| A5 | per-head gradient share on the shared trunk (as optimised: loss weights and Kendall factors applied) | every judged head >= 0.02, the largest <= 0.60 | a judged head < 0.02 (it does not shape the trunk) or one head > 0.60 (it owns the trunk) |
| A6 | head-gradient conflict | mean pairwise cosine of two heads' trunk gradients over R batches >= -0.10 for every pair | a pair below -0.10 on every seed (persistent conflict); between -0.10 and 0 = warn |
| A7 | dead units | per activation layer, share of units never active on the batch set <= 0.20 | > 0.50 (0.20-0.50 = warn) |

What to fix when it fails (one change at a time, each a registered switch, then re-run the gate):

| Failure | First fix | Next fix |
|---|---|---|
| A1 non-finite | find the op (anomaly mode on the failing batch); input clip; NIG floors | lower LR, gradient clipping |
| A2 dead tensor | wiring bug (output not used by any loss) | remove the parameter or the head |
| A3 vanishing in early layers | init switch (`keras`: glorot / orthogonal, B-007); normalisation placement | residual / shorter path (architecture card) |
| A3 exploding / spikes | gradient clipping (`clip_norm=1.0`, the TF value, B-018) | LR warmup, lower LR |
| A4 too small | LR up; check weight decay is not dominating | init scale |
| A4 too large | LR down, clipping | LR warmup |
| A5 one head owns the trunk | check Kendall log-variances (trace) and loss scales | loss normalisation per head; GradNorm-style weighting (TASK_WEIGHTINGS registry, B-036) |
| A5 a head starved | its loss scale / Kendall weight | separate small trunk for that head (architecture card) |
| A6 conflict | detach the conflicting auxiliary path (e.g. confidence from mu) | gradient surgery (PCGrad) or separate trunks (architecture card) |
| A7 dead units | init switch; lower dropout; GELU instead of ReLU | smaller LR, narrower layer |

### Gate B — confidence is useful (T-conf)

Confidence is useful when the head's own absolute error falls monotonically across confidence deciles WITHIN COIN
(`stats.metrics.confidence_error_spearman` / `confidence_error_gap`). Trivial baseline: a constant confidence scores 0.

| Env | Pass (every seed, every head) |
|---|---|
| E1 synthetic | decile Spearman <= -0.8, top-minus-bottom gap CI95 lower bound > 0, AND within-coin Spearman(confidence, -planted noise sd) >= 0.7 (the known answer) |
| E2 shuffled labels | the gap CI95 contains 0 (no confidence without signal; a "useful" confidence here is a leak) |
| E3 small real | decile Spearman <= -0.5 and gap CI95 lower bound > 0 (the mechanism works; the full -0.8 target is judged at T7r on the full data) |

Methods to make it useful and how to choose: section 4.

### Gate C — uncertainty is sharp around the predicted value (T-sharp, corrected by the owner 2026-11-07)

PER HEAD (close, high, low; pass/fail for each): the predictive std in price units is smaller than the head's own
predicted move `|pred_price - last_close|` in >= 80 % of samples, AND the empirical coverage of the head's 80 %
interval is within [0.75, 0.85]. Owner example: the model predicts the high +5 % above the close -> its uncertainty is
~2-3 %, the doubt AROUND the predicted value. The coverage condition stops a model from passing by shrinking its std
(a zero std scores 1.0 on the share: trivial baseline, rule C1). The previous text compared the width with
`pred_high - pred_low`; that interpretation is wrong and was never used for a recorded result (code fix: ROADMAP T4b).

| Env | Pass (every seed) |
|---|---|
| E1 synthetic | within-coin Spearman(pred std, planted noise sd) >= 0.8; mean(pred std) / mean(planted sd) in [0.8, 1.25]; coverage in [0.75, 0.85]; per head, the T-sharp share within 0.10 of the ORACLE share on the same data (the generator knows the true conditional distribution) |
| E2 shuffled labels | coverage in [0.75, 0.85] and mean(pred std) / sd(label) in [0.8, 1.25] (an honest model widens to the unconditional spread) |
| E3 small real | coverage in [0.75, 0.85] per head (after the interval-scale switch if on, fitted on a separate calibration slice); T-sharp share >= 0.80 for the high and low heads (close: informational, O-1) |

As built (G2, `docs/decisions/D-019-gates.md`): on E1 a band the ORACLE itself fails on the same rows is judged
relative to the oracle (high / low coverage: the true mean / sd with a symmetric interval covers ~0.88, B-056; tolerance
half the band width, 0.05); high / low coverage on E2 / E3 is information until the owner decides B-056; every gate B
gap CI is a day-block bootstrap (B-025). Tools: `trainer/grad_report.py`, `eval/gates.py`, `workflow/gate_run.py`, job
entry `gates`.

**Known answer and a warning for the close head (scratch probe, H0c, re-built as the T4b synthetic test).** With the
TRUE conditional distribution (random walk, 4h bars of 48 sub-steps, horizon 2, heteroscedastic vol, planted drift
with IC 0.05 or 0.10), the oracle's T-sharp share is 1.00 for high and low (median std / move = 0.81) and 0.00 for
close (median std / move 12.6-18.7). For a calibrated Gaussian close head, a share of 0.80 needs `sd(pred move) /
noise sd >= 1 / 0.253`, i.e. R^2 >= 0.94: no return model reaches that. So the close-head T-sharp is expected to fail
for any honest model. Owner decision O-1 (CLOSED 2026-11-07): the close-head T-sharp is informational only, never a
target or a gate (recorded, never blocking); the code already marks it non-blocking. High and low failures block.

### Rule D — direction (owner decision 2026-11-07)

- Direction is SUSPENDED for the high and low heads: sign, "hit rate", up/down accuracy and the class-head AUC of high
  and low are never evidence and never a selection metric (TF: high/low hit rate ~99 % for any model, F-0085). They may
  be printed, labelled "information only".
- Direction is judged on the close head only, always next to its trivial baselines (rule C1): shuffled prediction,
  constant, last-return sign (momentum) and its negative (reversal). Evidence only when the close head beats the BEST
  trivial baseline beyond its CI on every seed.
- Skill of high and low is their within-timestamp rank IC / per-coin Spearman of the magnitude, shown next to a
  trivial volatility baseline (the last bar's range, `last_high - last_low`, as the prediction): only the excess over
  that baseline is evidence.

### Leakage canary (runs inside every gate on E2)

On shuffled labels every head's within-timestamp rank IC is within 3 standard errors of its shuffled baseline (or
`|IC| < 0.01`), on every seed, and the val raw loss is not below the constant-prediction loss by more than 1 %. A
failure means a leak (labels, time or coin identity reaching the prediction): stop, fix, re-run all gates.

## 3. Test environments (in this order; same code path as the real job: a contract artifact through the loader)

| Env | What | Why | Cost |
|---|---|---|---|
| E1 synthetic, known answer | generator `data_io/synthetic.py` (ROADMAP G1) writes a contract artifact: >= 32 coins x >= 2,000 bars (rows at stride = horizon, one row per horizon block, so 2,000 bars give 1,000 rows per coin; D-018), intrabar random-walk paths so high / low / close are consistent, a planted close-return drift `b * feature_k` with known IC (0.05 and 0.10 variants), heteroscedastic noise whose sd is a known function of another feature and of a per-coin scale, the exact conditional mean and sd per row stored next to the labels | the model must recover the planted signal (rank IC >= 0.7 x the oracle IC), its uncertainty must recover the noise level and its confidence must rank the planted error | CPU minutes |
| E2 shuffled labels | the real small sample (E3) with all label rows permuted globally (across coins and time), inputs untouched | skill and confidence must be ~0: the leakage canary; also measures how fast the model memorises pure noise | CPU / GPU minutes |
| E3 small real sample | a slice of the real artifact: 20 coins x 6 months from the TRAINING era only, its own time split inside (70 / 15 / 15, purged); never touches the full run's val / test or any sealed data | the mechanism works on real data before paying for a full run | GPU minutes |

Each environment runs the seed set; the report keeps one row per (env, seed, gate, check) and the verdict is the
logical AND over seeds.

## 4. Confidence and uncertainty: candidate methods and how to choose

All are registered components (rule B4) chosen by a switch whose default reproduces today.

| Method | Confidence / uncertainty from | Cost | Main risk |
|---|---|---|---|
| NIG evidential (today) | NIG `nu, alpha, beta`; confidence head trained on batch-normalised error + diversity penalty (B-005) | none | floors bind (B-001); diversity penalty forces spread unrelated to error |
| ranking loss on own error | pairwise / listwise loss: within coin, higher confidence must mean lower `|error|` (error detached), no diversity penalty | small | needs enough rows per coin per batch (stratified batches) |
| dedicated error-predictor head | head regressing `log |residual|` on DETACHED residuals (never pushes mu) | small | learns volatility level only; check within-coin ranking, not pooled |
| quantile heads | pinball loss at 0.1 / 0.5 / 0.9 per head: interval directly, confidence = 1 / width | medium | crossing quantiles (sort or monotone parametrisation) |
| seed / deep ensemble | spread of M >= 5 independently trained models | M x training | cost; spread measures model (epistemic) doubt only |
| MC dropout | spread over K stochastic forward passes | K x inference | often under-dispersed; dropout 0.25 is already on |
| post-hoc calibration | `eval/calibrator.py` interval scale (and isotonic on confidence) fitted on a calibration slice of val, measured on the rest | none at train time | fixes coverage, not ranking; must never be fitted and measured on the same rows |
| heteroscedastic two-tower | separate small trunk for the scale (architecture card) | medium | only if gate A6 shows mu / scale conflict |

Choosing by evidence: run the gate battery (B and C on E1-E3, 3 seeds) for each candidate on one commit; keep the
candidates that pass on every seed; among them prefer (1) the one that does not lower T-skill beyond seed spread, (2)
then the best E3 decile Spearman, (3) then the cheapest. Post-hoc calibration may be stacked on any of them. A
candidate that passes only with ensembles must state the inference cost in its card.

## 5. Memorization study (top-priority finding to explain)

Owner observation (2026-11-07, TF-era runs; not yet reproduced here): with 3 features, 64 units and > 2M training
windows the model still memorises. Rule: no fix is proposed before the cause is localised (TF RESEARCH_RULES).
Operational definition used everywhere: the train metric measured in eval() mode on a train subset of the val size
(same metric, same code) exceeds the val metric by more than 2 x the seed spread, AND/OR a shuffled-label run fits
the train set above the linear-model level.

TF failure registry checked before proposing (reopen conditions respected): F-0029 (InstanceNorm erased the level ->
memorisation; fixed by level passthrough + BatchNorm + GRU = today's `ANTI_MEMORIZATION_CONFIG`; reopen only if the
level reaches the head through another channel: H-M3 does NOT re-run the InstanceNorm config, it asks whether the fix
itself leaks coin identity), F-0030 (capacity / encoder sweeps flat on 1d; R2 "fair coin-embedding test" and R3
"feature-count sweep" never run: H-M3 / H-M7 are not repeats), F-0031 (no memorisation at train-test gap ~0.01 on the
1d relative target: so the question is what differs on 4h own-return targets), F-0059 (effective sample size never
recorded: H-M1 records it).

| id | Hypothesis | Discriminating test | Expected if TRUE | Cost | Step |
|---|---|---|---|---|---|
| H-M8 | "memorisation" is partly a measurement artefact: train loss is logged in train() mode (dropout 0.25, input noise 0.1, feature dropout, label smoothing, lambda schedules rising) and compared with eval-mode val loss | eval() train-subset metric vs val metric, same code, on the T5r checkpoint; per-epoch | the eval-mode gap is much smaller than the logged gap | CPU, minutes | M1 (first: it defines the effect) |
| H-M1 | overlapping windows: the effective sample size is far below 2.5M (stride 1, every window shares T-1 bars; horizon-2 labels overlap; ~4.7k windows per coin = 2,520,913 / 535; coins move together) | count distinct timestamps; label autocorrelation per lag; mean pairwise cross-coin label correlation rho at the same timestamp -> ESS ~ timestamps x coins / (1 + (coins - 1) rho); short-run arm training on stride-h subsampled windows | ESS of order 10^4 or less; the stride-h arm reaches the same val loss with the same gap | CPU (counts) + 1 short GPU arm | M1 (counts), M2 (arm) |
| H-M2 | the market-common part of own-return targets is memorised by date: the model recognises the date (vol regime, level) and recalls that timestamp's market move | R^2 of labels on timestamp fixed effects (train); decompose the trained model's train/val error into the timestamp-mean part and the residual; probe predicting a month id from inputs and from the embedding | large timestamp R^2 (> 0.3); the gap sits in the timestamp-mean part; month id decodable far above chance | CPU on one checkpoint | M1 |
| H-M3 | coin identity leaks through price level / level passthrough / batch-norm level path, letting the model memorise per-coin paths | linear probe of coin id from raw inputs, post-passthrough tokens and the final embedding (accuracy vs 1 / n_coins and vs shuffled ids); short arm with `level_passthrough=False` | coin id decodable from the embedding; the gap shrinks without passthrough (while skill may drop: report both) | CPU probes + 1 short GPU arm | M1 (probe), M2 (arm) |
| H-M4 | non-stationarity mistaken for memorisation: later dates differ, any model (also linear) loses skill | (a) train coins A / val coins B over the SAME dates vs (b) same coins, later dates; plus a ridge baseline under both | small gap in (a), large in (b), the ridge baseline shows a similar (b) drop | needs a coin-split option (S, default off) + 2 short GPU arms | M2 |
| H-M5 | label overlap / timing leak at the split boundary (B-020: is `last_candles` time the bar OPEN time? the job uses `embargo_hours=0`) | count train rows whose label end >= val start; recompute label times from raw bars for 3 coins; short arm with embargo = window + horizon (TF default) | non-zero overlap; val loss at early epochs drops when the embargo is removed | CPU + 1 short arm | M1 |
| H-M6 | no effective regularisation / too many steps: AdamW decay `lr x wd = 1e-3 x 1e-4 = 1e-7` per step is a no-op (TF F-0029 invariant), no clipping, constant LR | val metric every 1/4 epoch: when does the gap open? arms: weight decay 1e-2 / 1e-1 with norm/bias exclusions; LR 3e-4 | the gap opens late (after epoch 1) and shrinks with real decay / lower LR | 2-3 short GPU arms | M2 |
| H-M7 | label noise vs capacity: the model can store noise | shuffled-label train curve (how fast the eval-mode train loss falls below the constant loss, in samples); capacity arms d_model 16 / 32 / 64, 1 vs 2 layers | the shuffled curve falls within the epoch budget; the real gap scales with capacity | 1 + 3 short GPU arms | M2 |
| H-M9 | near-duplicate windows (stale bars, dead-coin tails, exact-zero returns; TF `research/stale_bars`) are memorised easily | share of windows with >= k zero-return bars or identical feature hashes; gap with them removed | a non-trivial duplicate share and a smaller gap without them | CPU | M1 |

Every arm is a short run with the seed set where it decides something, one change per arm, on one commit, with a card
(predicted movement + accept rule) written before it runs. A hypothesis that is ruled out gets a row in
`docs/bottlenecks.csv` marked `rejected` with the evidence; a verified cause becomes a switch (T6c) and goes through
the gates. A failed idea is also recorded in the TF failure registry (allowed write, ROADMAP).

## 6. After training: the model doctor

Every full training writes `doctor_report.json` / `.md` next to the checkpoint (ROADMAP D1-D2; until then the T4
`health_report`). Each check states its value, threshold, verdict and trivial reference. Warn thresholds: dead units
> 20 % of a layer; GRU hidden |h| > 0.95 in > 30 % of entries; class-head p < 0.01 or > 0.99 in > 20 % of rows;
attention entropy / log(n_tokens) < 0.2 (transformer only); a NIG parameter on its floor in > 10 % of rows or the
uncertainty at `unc_max` in > 1 %; std(pred) / std(label) < 0.05 for a head (collapse); `linear_path` + level
passthrough > 0.8 of the readout variance or gradient (passthrough dominance); a layer whose zero / permute ablation
changes val loss less than seed noise (dead layer); Kendall log-variance still drifting over the last 5 epochs; train
/ val gap > 2 x seed spread; per-coin coverage: < 60 % of coins inside [0.75, 0.85]; calibration per head (coverage
at 50 / 80 / 95 %, PIT histogram) off by > 5 points.

## 7. Evidence and records

- A card before every run (one change, predicted diagnostic movement, accept rule); the TF template
  `docs/research/templates/` is the model.
- No "X is better" without beating seed noise (C4: >= 2 seeds per arm on one commit, paired test) AND the predicted
  diagnostic moving.
- Every metric next to its trivial baseline (C1); direction per rule D. T-profit numbers come from the battery with
  their baselines and CI, and the number of variants tried is stated in the result file.
- Exploratory until the owner opens the holdout (C5); the holdout is opened once, in ROADMAP F4, only on the owner's
  request.
- Results: `docs/results/<run>.md` with the gate report, the doctor report and the commit; bottleneck rows updated.
