# D-019: gate tooling, gate job and the job-side gate check (ROADMAP G2)

Date: 2026-11-08
Owner decision: step G2 (manager message of 2026-11-08: "PHASE 2 (PR 2): ROADMAP step G2, gate tooling and gate job";
"Manager decision for B-056 (option a, recorded in a decision file): on E1, Gate C coverage for high / low is judged
RELATIVE to the oracle's own score on the same rows ..., with a stated tolerance, not against the fixed [0.75, 0.85]
band; the same 'relative to the oracle' rule for every E1 check where the oracle itself fails a fixed band ... Close keeps
the protocol band ... On E3 and the real data, report coverage against the band but mark the high / low verdict as
informational until the owner decides B-056"; follow-up message: "use the day-block CI for Gate B where the protocol
allows (B-025, suggestion rows 157/159/160), and show HAR as information next to rule D's last-range baseline (B-057)
without changing rule D itself"; "Do not fix B-052").
Why: rule C6 / MODEL_DEV_PROTOCOL section 2 require gates A, B and C on E1, E2 and E3 (3 seeds) before any full training;
until now they were a manual check (B-040).

## What changes behaviour
- `jobs/baseline.py`: after `build` the job calls `workflow.gate_run.gate_check` and prints its result; the record is a new
  key `gate_check` of `baseline_summary.json` (`tests/pins/baseline_fixture.json` updated). The baseline entry always
  trains the default configuration (baseline arm), so it is exempt and never refused; new optional config key
  `gate_report` (default none) is read and recorded. A non-baseline full training that calls `gate_check` without a
  passing report of its configuration key raises `GateRefused` (no such job exists yet; T7r's `all-001` will be one).
  "Full training" = more than one epoch and no `max_rows` (protocol section 1 adds "on the full artifact"; a strided smoke
  is never checked).
- `Trainer.fit(trace_batches=None)`: with index arrays the trace is the mean over them; `TrainSettings.trace_representative`
  (default False = today: the trace on the first `batch_size` training rows) switches the run steps to the
  representative batch set. Training is unchanged either way (test: identical weights).
- `stats.metrics.confidence_error_gap(blocks=None)`: None = today's iid bootstrap; with block ids a block bootstrap.
  `target_report` keeps the iid CI (B-025 stays open for the reports); the gates use day blocks.
- `trainer.trace.head_gradients`: the shared gradient pass of the trace, now also used by gate A5 / A6 (refactor, outputs
  unchanged); a fully detached head now gives no gradient instead of raising.

## New modules and their thresholds (all from MODEL_DEV_PROTOCOL section 2 unless stated)
- `trainer/batch_set.py`: the representative batch set, R = 16 batches, >= 32 coins each (section 1), one row per
  equal-count time stratum of each drawn coin, fixed seed (the run's seed).
- `trainer/grad_report.py` (gate A, one measurement; the gate run measures at step 0 and after the short run):
  A1 finite; A2 grad norm > 1e-12 on some batch, a tensor that gets a gradient once the loss schedules reach their end
  value is "scheduled off" (warn; the confidence heads at epoch 0, `lambda_calib` starts at 0); A3 layer RMS within
  [1e-3, 1e3] x the median layer, batch-norm spike > 10 x median warn / > 100 x fail (measured over the R batches of one
  measurement, not per training step: not logged yet); A4 one real optimizer step, median update / weight in
  [1e-4, 1e-2] (fail outside [1e-5, 1e-1]), every weight matrix in [1e-6, 1e-1], A2's dead tensors left out (state
  restored exactly); A5 every judged head (loss weight > 0) >= 0.02, the largest <= 0.60, shares of the gradient norm on
  the shared TRUNK (parameters at least two heads reach, minus the parameters of `head_modules()`: the ordered means make
  the close NIG head reachable from the high / low losses, and with it the high / low cosine is -0.999, on the trunk
  alone +0.79 on E1; the T4 trace keeps its own definition); A6 mean pairwise cosine >= -0.10 (< 0 warn), a fail
  blocks only when it fails on every seed; A7 share of never-active units per layer <= 0.20 (> 0.50 fail), a unit is
  never active when its pre-activation (the model's `activation_points`, Linear layers feeding ReLU / GELU / SiLU) is
  <= 0 on every row of the batch set.
- `eval/gates.py`: gate B (E1 decile Spearman <= -0.8, gap CI lower > 0, within-coin Spearman(confidence, -oracle sd)
  >= 0.7; E2 gap CI contains 0; E3 decile Spearman <= -0.5 and gap CI lower > 0; every gap CI is the DAY-BLOCK
  percentile bootstrap); gate C (E1 within-coin Spearman(std, oracle sd) >= 0.8, mean std / mean oracle sd in
  [0.8, 1.25], coverage in [0.75, 0.85], T-sharp share within 0.10 of the oracle's; E2 coverage band and mean std / label
  sd in [0.8, 1.25]; E3 coverage band and T-sharp share >= 0.80 for high / low); the leakage canary on E2 (every head's
  rank IC within 3 se of its shuffled baseline or |IC| < 0.01; val raw loss not below the constant-prediction loss by more
  than 1 %, the constant = the model's mean output on every row); E1 signal recovery (rank IC >= 0.7 x the oracle's,
  section 3, information); rule D rows (close direction vs the best of always up / down, shuffled, momentum, reversal
  with a day-block CI; high / low partial rank IC net of the last-bar range) and, as information (B-057), the same net of
  HAR mean6. Rule D, HAR and the signal check never block.
- Relative to the oracle (B-056, option a): on E1, a check whose fixed band or bound the ORACLE itself fails on the same
  rows (with its symmetric 80 % interval mean +- 1.2816 sd) is judged relative to the oracle: two-sided band -> |value -
  oracle| <= half the band width (0.05 for coverage); one-sided bound -> within 0.10 of the oracle (the protocol's
  T-sharp tolerance). In practice this applies to high / low coverage (oracle ~0.88); close keeps the band (oracle 0.80).
  A skewed / quantile interval is a later T6 switch candidate (B-056), not part of G2.
- Information until the owner decides B-056: high / low coverage on E2 and E3 (E2 permutes E3, the same skewed labels).
  Close-head T-sharp is information everywhere (O-1).
- `workflow/gate_run.py` + entry `jobs/gates.py`: E1 (D-018 generator, 3 epochs), E3 (`real_slice` of `e3_source`,
  3 epochs; the protocol's "1/4 epoch" is for gate A only and the run steps have no step limit; a first local run with
  1 epoch showed it is too short for gates B / C: the default `lambda_calib` schedule is 0 in epoch 0, so the confidence
  heads are never trained and the NIG alpha has not grown, coverage 0.999), E2 (the
  label permutation of E3, of E1 when no E3 source is given, 3 epochs), seeds 0, 1, 2; each env is written as a contract
  artifact and read back; each env x seed run folder is rebuilt (gate runs never resume). `gate_report.json` / `.md`
  carry the key `trainer.identity.configuration_hash` (the run identity's settings hash without the seed, the data and
  the commit). `passed` needs every env with >= 3 seeds and every blocking check passing on every seed.

## New defaults
- `nig_timenet_torch.eval.gates.GateStage(last_move)`: `<new>` -> `None`
- `nig_timenet_torch.eval.gates.GateStage(losses)`: `<new>` -> `None`
- `nig_timenet_torch.eval.gates.GateStage(oracle)`: `<new>` -> `None`
- `nig_timenet_torch.eval.gates.GateStage(ranges)`: `<new>` -> `None`
- `nig_timenet_torch.eval.gates.summarize(envs)`: `<new>` -> `('E1', 'E2', 'E3')`
- `nig_timenet_torch.eval.gates.summarize(min_seeds)`: `<new>` -> `3`
- `nig_timenet_torch.jobs.baseline.BaselineConfig(gate_report)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.gates.CONFIG`: `<new>` -> `<class 'nig_timenet_torch.jobs.gates.GatesConfig'>`
- `nig_timenet_torch.jobs.gates.GatesConfig(architecture)`: `<new>` -> `'nig_timenet_v2'`
- `nig_timenet_torch.jobs.gates.GatesConfig(e1)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.gates.GatesConfig(e1_epochs)`: `<new>` -> `3`
- `nig_timenet_torch.jobs.gates.GatesConfig(e2_epochs)`: `<new>` -> `3`
- `nig_timenet_torch.jobs.gates.GatesConfig(e2_permutation_seed)`: `<new>` -> `0`
- `nig_timenet_torch.jobs.gates.GatesConfig(e3_epochs)`: `<new>` -> `3`
- `nig_timenet_torch.jobs.gates.GatesConfig(e3_months)`: `<new>` -> `6`
- `nig_timenet_torch.jobs.gates.GatesConfig(e3_n_coins)`: `<new>` -> `20`
- `nig_timenet_torch.jobs.gates.GatesConfig(e3_source)`: `<new>` -> `''`
- `nig_timenet_torch.jobs.gates.GatesConfig(e3_train_end)`: `<new>` -> `''`
- `nig_timenet_torch.jobs.gates.GatesConfig(model_overrides)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.gates.GatesConfig(seeds)`: `<new>` -> `(0, 1, 2)`
- `nig_timenet_torch.jobs.gates.GatesConfig(train)`: `<new>` -> `{}`
- `nig_timenet_torch.jobs.gates.run_settings(code_sha)`: `<new>` -> `''`
- `nig_timenet_torch.stats.metrics.confidence_error_gap(blocks)`: `<new>` -> `None`
- `nig_timenet_torch.trainer.batch_set.representative_batches(min_coins)`: `<new>` -> `32`
- `nig_timenet_torch.trainer.batch_set.representative_batches(n_batches)`: `<new>` -> `16`
- `nig_timenet_torch.trainer.batch_set.representative_batches(seed)`: `<new>` -> `0`
- `nig_timenet_torch.trainer.grad_report.grad_report(allow_dead)`: `<new>` -> `()`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e1)`: `<new>` -> `{}`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e1_epochs)`: `<new>` -> `3`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e2_epochs)`: `<new>` -> `3`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e2_permutation_seed)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e3_epochs)`: `<new>` -> `3`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e3_months)`: `<new>` -> `6`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e3_n_coins)`: `<new>` -> `20`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e3_source)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.gate_run.GateSettings(e3_train_end)`: `<new>` -> `''`
- `nig_timenet_torch.workflow.gate_run.GateSettings(seeds)`: `<new>` -> `(0, 1, 2)`
- `nig_timenet_torch.workflow.gate_run.run_gates(g)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.TrainSettings(trace_representative)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.settings.RunSettings(train)`: `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto')` -> `TrainSettings(epochs=5, batch_size=256, seed=0, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, selection_metric='val_raw_loss', early_stop=EarlyStopSettings(patience=15, min_delta=0.0001, smoothing='window', window=3, ema_beta=0.7, restore_best_weights=True), device='auto', trace_representative=False)`
