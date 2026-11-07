# Evaluation of `feat/ar-stream` (PR #25) and the integration packages I1..I7

Written by H0b (2026-11-07). Subject: branch `feat/ar-stream`, head `22d240e` (15 commits on top of `7c453af`, which
contains the older `feat/native-data-notebooks-uncertainty`), PR #25 by the owner, open, CI red. Nothing on that
branch or PR was touched. All numbers below were measured on a worktree of the head (`git diff --stat origin/main...`,
AST scans, `pytest`, `ruff`). Rules are `docs/RULES.md`; findings are `docs/hardening/AUDIT-2026-11.md` (HF-nn).

## 1. Size and test results

| item | value |
|---|---|
| diff vs the merge base | 103 files, +20,547 / -41 (of which `data/resources/holdout.json` 8,786 lines, 5 notebooks 2,191 lines of JSON, `D-006` 194, tests ~5,470) |
| new Python in `src/` | ~6,000 lines, 56 files (47 new modules + changes to 9 existing); largest module 351 lines (`data/ar_windows.py`), largest function 65 lines (`Trainer.fit`, existing) |
| tests added | ~190 test functions in ~28 files; the branch suite is 706 tests |
| `pytest -q` on the head, as CI runs it (`pip install -e ".[dev]"`) | **20 failed, 27 errors, 659 passed**: every one is `ModuleNotFoundError: pandas_ta_classic` (the new optional extra `[data]`, which `.github/workflows/*.yml` does not install). PR #25 check runs: both `test` jobs `failure` |
| same, with `pandas-ta-classic==0.8.32` added (installed into a scratch dir, venv untouched) | **705 passed, 1 failed** in 142 s: `tests/data/test_ar_windows.py::test_raw_candles_drive_reaggregates_the_1h_files_up_to_interval`, exact float equality `101.74910193531912 == 101.74910193531913` (`max` over a product vs the stored value; platform-dependent; fix with `approx`) |
| `ruff check src tests` | 1 error: `E501` `tests/notebooks/test_panel_ar_stream_notebook.py:19` (138 chars) |
| E3 (fast suite < 300 s) | full suite 142 s with the extra on this machine: inside budget; `tests/budget.py` is part of the suite and passed |

So the branch is red in CI for two cheap reasons (missing extra in the workflow, one long line) plus one brittle test.
Nothing else failed: the code itself is in good shape and well tested.

## 2. Quality against the rules

**Ratchets (`tests/ratchets.json`)**: only two lines move, both down: `G4_dead_settings_fields` 1 -> 0 and
`G8_orphan_modules_in_registry_folders` 1 -> 0 (HF-29: `RunSettings.rolling`, `model/panel.py` become used). G1 (22), G2 (11),
G3 (1 / 0), G5 (0), G7 (1), C2 (1) are unchanged, i.e. the new code adds no counted violation. Caveat: G5 passes only
because the scan looks outside `trainer/`; `trainer/ar_stream.py:279-283` and `trainer/panel_ar_stream.py:98-102` are two
more hand-written `zero_grad / backward / clip / step` loops with their own `AdamW` and selection logic, inside `trainer/`.
That is a second training system (G5's principle, "one trainer"), invisible to the counter.

**`tests/scan.py`**: adds `"data": {"core", "data_io"}` to `ALLOWED_IMPORTS`, `data` to `_ON_TOP`, and counts `pyproject`
console scripts as entry points (G8). Harmless; collides with H5's `mode_name_comparisons` edit (different hunks).

**Pins / decisions (B1)**: `tests/pins/defaults.json` +423 / -1 lines (422 new keys, **no changed or removed default**, fingerprint
refreshed). `D-006-ar-stream-defaults.md` lists them as `<new>`. The decision number D-006 is the only collision: the
H packages take the next free number at build time. Because the integration is by porting, each I-package writes its OWN
`D-nnn` for its own keys and regenerates its pin with `python -m tests.defaults_pin --add`; the D-006 text is mined for the rationale.

**G1-G8 / B4 / B5 / C2 / E4 check**
- G1: `flat_run.TARGETS = ("high","low","close")` and `panel_run` repeat the head vocabulary; the existing counter does not flag it. Port: derive from `head_spec`.
- G2: no new `== "<registered name>"` in counted form; `ENCODERS = ("transformer","gru")` (`model/ar_candle_transformer.py:18`) is a second hand-written encoder list next to `model/builder.py:46` and the encoder choice is an `if encoder == "gru"` branch inside the module (not through `model/parts` registry). Port: register the trunks.
- G3: all new modules <= 351 lines, all new functions < 65 lines. Good. `workflow/run.py` (251) is the biggest workflow module.
- G5: see above (two extra training loops; `workflow/run.py` correctly uses the one `Trainer` for the flat and panel NIG models).
- G6/E4: notebooks have no `def`; they import normally. Five notebooks have headless tests that execute them on a synthetic market. Good.
- G7: `data` is added as a top layer; `model`/`trainer` do not import `workflow` in the new files. `eval/ar_stream.py` imports `model.ar_candle_transformer` (eval -> model: allowed).
- B4: `data/scaling.py` holds the per-window normaliser and `data/targets.py` a target-mode registry (`retarget` fitted on train rows only): registered, with `fit` on train. The ar-stream `fit_volume_median` / `local_vol` are fitted per call, not stored with the model (B4 gap).
- B5/strictness: `core.config.updated` raises on unknown keys with a did-you-mean; `DataSettings`/`RunSettings` are strict frozen trees. Good.
- C2: the battery imports `stats.metrics.spearman`; it has private `_stats` (win/avg/Sharpe/profit factor) in `eval/battery_lib.py` and a private `_null_score`, which are trade statistics, not in `stats.metrics`. Fine, but `stats/` has no trivial-baseline test for them (C1 test only covers `stats.metrics.__all__`).
- D1 (data integrity): `data_io/loading.py:112-117` runs the D1 checks only when the folder's manifest has an `export` block; a `pkl.gz` or an old folder loads unchecked. This weakens the T2 rule "the loader refuses an artifact without the manifest fields". Fix on port: unchecked loading only with an explicit `allow_unchecked=True` setting.
- Data provenance: `data/resources/holdout.json` is a copy of the TF lock, pinned by sha256 (`data/lock.py:29`) and count: good. The native pipeline does not make the TF export unnecessary for old results (TF export `c572b1b36960` stays the reference), it gives a second route.

**Duplicates of existing code (the three H0 flagged, confirmed)**
1. `model/ar_candle_transformer.py:18` `ENCODERS` next to `model/builder.py:46` (H3).
2. Trainer-config builders `workflow/flat_run.py:41` (`flat_trainer_config`) and `workflow/panel_run.py:46` are copies of `jobs/baseline.py:51-67` logic (H4's `targets_config`). `flat_trainer_config` hard-codes the oracle schedule numbers (`lambda_reg 0 -> 0.05, 5 warmup`, `lambda_calib 0 -> 0.1`).
3. `workflow/run.py` + `run_common.py` + `flat_run.py` are the "run steps" HF-23 asked for, but `jobs/baseline.py` is NOT rewired to them: after adoption there would be two entries running the same recipe. Also missing in `run.py`: the T4 `health_report` (so nothing in `run()` writes `health_report.json`), the D-005 `test_report.json`, the E1 device assertion/validation and the E2 publish contract (those are `colab/` + `jobs/`), and a literal `[:, 3]` timestamp column (`run.py` `run_single`, HF-12) next to `TS_COL` used in `run_common.split_bounds`. It does implement the #21/#22 lessons correctly (input key read from the dataset's timeframes, split bounds from `split_dates` else quantiles) and has a `max_rows` smoke limit (`_limit`, HF-16).
4. `eval/battery.py` builds its own table (`rollout_predictions`: columns `step, mu, sigma, p_dir, y_ret, dir_true, last_ret, coin, end_time`) rather than the repo's `PredictionTable`; the NIG flat model cannot be fed to the battery without an adapter (not written).

**Cards (H1)**: 46 of 47 new modules have a good module docstring that states the purpose, none has the `PURPOSE:/TAGS:/PITFALLS:` shape H1 will define. Each port converts the docstring (cheap) and `tools`-style map entries come from H1's generator.

**Other**: `docs/STATUS.md` row "AR1" says "fourteen-section battery"; the module/PR text says "ten sections" and the code has 9 functions + `verdict` (unclear; the notebook is the source of truth). The branch's `STATUS.md` / CSV edits are append-only and will conflict textually with ours (resolved by hand at port time). The branch changes `PanelModel.forward` into `mix()` + `heads()` and `Trainer` (`n_rows/take/to_device`, panel dict input): behaviour preserved for the tensor path (existing tests pass), a 14-line diff each, minimal collision with H4 (H4 edits `__init__` and the `fit` schedules line).

## 3. Inventory and verdicts

Size = changed lines (additions) as measured; tests = number of test functions. Verdict key: ADOPT / ADOPT-FIX / HARVEST / DROP.

| # | component | files (lines) | purpose | tests | verdict |
|---|---|---|---|---|---|
| C1 | Native data pipeline equal to TF | `data/` 25 modules ex. ar_windows/panel/publish/cli: settings 276, build 270, targets 266, context 264, scaling 203, custom_* 336, fetch 228, windows 153, align 142, artifact_writer 125, indicators 74, sources 161, lock 40 (~2.9k src) + `resources/holdout.json` | settings -> candles -> features -> windows -> targets -> contract artifact, strict `DataSettings`, same keys as the TF dataset | 82 in `tests/data/` (all data tests, ar and panel included) incl. `test_parity_tf` (bit-exact vs two TF-built fixtures, 4 coins, 1 and 2 timeframes) | **ADOPT-FIX** (I2) |
| C2 | Drive source + INTERVAL re-aggregation | `data/sources.py` (`files` source, Speirsy11 parquet), `sources.resample_ohlcv`, `ar_windows.raw_candles/_drive_frames`, `fetch.py` (Binance FAPI + data.binance.vision) | read the owner's Drive candle files and combine 1h up to 4h/1d with the old project's rules (full bars only) | 12 `test_sources_fetch`, 1 re-aggregation test (the failing float test) | **ADOPT-FIX** (I2; the re-aggregation lives in the data package, `ar_windows` calls it) |
| C3 | SPLIT_AT / TEST_AT calendar split | `data/ar_windows.py:126-210` (`_date_idx`, `split_windows`, `split_panel`) | train/val/test cut at calendar instants shared by all coins | in `test_ar_windows` / `test_panel` | **HARVEST** now, ports with the ar-stream data (I6); our flat path already has `SplitPlan` purge+embargo with `split_dates` |
| C4 | Named datasets and loading | `data/publish.py` 33, `data_io/loading.py` 120, `npy_dir.load(mmap)`, `artifact.load_artifact(mmap)` | `<name>_<stamp>.dataset` + `<name>_latest.dataset`, name/format resolution, mmap, local copy | 7 + parts of notebook tests | **ADOPT-FIX** (I1): D1 must stay mandatory |
| C5 | `workflow/run.py` RunSettings/run (flat + panel + rolling) | `settings.py` 115 (replaces seam), `run.py` 251, `run_common.py` 93, `flat_run.py` 92, `panel_run.py` 99, `rolling.py` +22, `trainer/rolling_fit.py` 53, `core/config.py` `to_dict/updated` | one call from settings to results; reusable steps (load, retarget, split, device, identity, fit, table, report); rolling folds through the same `Trainer` | 10 `test_run`, 4 `test_rolling_fit` | **ADOPT-FIX**, split in three: core helpers into I1, flat single-split steps into I3, rolling + panel into I4 |
| C6 | Grid panel model, panel tasks, 2-D targets | `model/grid_panel.py` 39, `model/panel.py` +18 (`mix`, `heads`), `trainer/panel_tasks.py` 75, `trainer/batching.py` 28, `eval/panel_eval.py` 45, `data/panel.py` 166, selection metric `val_panel_ic_*` | days x slots grids end to end, within-day rank IC as selection metric, trainer takes dict inputs | 5 + 7 + 3 + 7 (data) | **ADOPT-FIX** (I4), after H3/H4 |
| C7 | ar-stream autoregressive model | `model/ar_candle_transformer.py` 94, `model/panel_ar_candle.py` 109, `trainer/ar_stream.py` 335, `trainer/panel_ar_stream.py` 153, `data/ar_windows.py` 350, `eval/ar_stream.py` 277, `workflow/ar_stream.py` 92, 2 notebooks | decoder-only (transformer or GRU trunk) predicting the next candle features + log_sigma + direction; scheduled sampling; czar or huber loss; full-rollout evaluation with shuffled-label control and 2 trivial baselines | 6 + 7 + 12 + 4 + 17 + 7 + 3 + notebooks 6 | **ADOPT-FIX as an isolated experimental package, last (I7)**, only if H0c wants this research line; its ideas are HARVESTED regardless |
| C8 | Evaluation battery | `eval/battery.py` 281, `battery_classics.py` 109, `battery_lib.py` 58, `workflow/ar_stream.run_battery`, `ar_stream_battery.ipynb` | many-angle judgment of a prediction table: sanity IC, confidence buckets, anti-cowardice, trade grids, limit execution, uncertainty filter, shock split, weekly stability, per-coin, flip test, TF-ported classics (integrity, persistence baseline, selective evaluation, market-neutral), verdict | 16 + 2 notebook | **ADOPT-FIX** (I5), the most useful piece for model work |
| C9 | Prediction dumps | `eval/ar_stream.py:257 dump_run` (+ `rollout_predictions`) | write `<run>/predictions/` the battery notebook auto-detects | 2 | **ADOPT-FIX**: replaced by the existing `test_predictions.parquet` of `run.py` + the I5 adapter |
| C10 | Notebooks + headless tests | `notebooks/{data_pipeline,main,cross_asset_pipeline,ar_stream,panel_ar_stream,ar_stream_battery}.ipynb`, `notebooks/README.md`, 5 test files | thin runners over the steps; the tests execute every cell on the synthetic source | 17 | **ADOPT** with their component (each notebook travels with its package); `colab_runner`-style bootstrap unchanged |
| C11 | Pin refreshes + ratchet/tests plumbing | `tests/pins/defaults.json` +423, `tests/ratchets.json`, `tests/scan.py`, `pyproject.toml` (`[data]` extra, `nig-data` script, package data), `colab/bootstrap.py purge_package` | bookkeeping | covered | **ADOPT-FIX**: regenerate per package, never copy; `[data]` extra must be added to CI (this is why #25 is red) |
| C12 | `core.config.to_dict/updated` | `core/config.py` +54 | settings copy with dotted keys, unknown keys raise with did-you-mean | covered by `test_run` and notebooks (no direct unit test) | **ADOPT-FIX** (I1): add direct tests |

**Reasons.**

C1 ADOPT-FIX. The package is the biggest single asset: a strict, frozen `DataSettings` tree replaces TF's global `CONFIG`;
parity with the TF reference is tested bit-for-bit (`assert_array_equal`, dtype equal) on two fixtures written by TF's own pipeline
(`make_tf_parity.py` documents how to regenerate); the artifact writer drops sealed coins and windows past the time seal and
records source commit, coin list, lock hash (D1), the lock is hash-pinned. It makes a whole class of "the TF export lies in a
Drive folder" problems (T2, B-020, HF-15 fixture mismatch) solvable here: a 4h artifact with `split_dates=None` can be built in
a test. Fixes: install the extra in CI; make the one float test robust; cards; keep `pandas_ta_classic` lazy (already
`ta_backend.py`); split the pin per package; confirm the parity fixture still matches TF `main` (the TF repo is frozen, so a
re-run of `make_tf_parity.py` is optional and not done here); decide the CLI `nig-data` is documented (it is in the README).
Not verified: a real Drive build (B-028 on the branch is open too); the fetch paths need the network.

C2 ADOPT-FIX. Reads the owner's files in the TF layout, applies the old project's OHLCV rules (`resample_ohlcv` first/max/min/last/sum, only full bars survive) and has
a test for exactly that; fix the float assertion. This is the "real data loading" the next model work needs, without the TF export
having to be rerun.

C3 HARVEST. Calendar cuts are what the flat path already gets from `SplitPlan.purge_embargo_split` with `split_dates`
(embargo and purity tested); the ar-stream version has no purge, so it comes with the autoregressive data (I6) only.

C4 ADOPT-FIX. Small, pure numpy, tested; `LoaderSettings` + `LoadedData` is the loader the run steps and the job need
(H2 can use it). Required fix: D1 mandatory by default (section 2).

C5 ADOPT-FIX. `run.py` is the first reusable set of run steps (HF-23, B-035): the flat and panel runs and every rolling fold go
through the ONE `Trainer`, the settings are a strict tree, a `retrain` path and a `max_rows` smoke limit exist, and tests pin
the rolling-fold purity. But as is it duplicates `jobs/baseline.py` and the trainer-config builder, lacks the health report, the
D-005 test report, E1/E2, and hard-codes the heads/timestamp column. So port it AFTER H2 and H4 and make `jobs/baseline.py` a
thin entry over it (fixes: one `targets_config` from `HeadSpec`; health report as a step; `TS_COL`; `test_report.json`;
use the H2 strict job config for device/paths; delete the duplicated job helpers).

C6 ADOPT-FIX. The 2-D grid interface keeps predictions as (days, slots, K) with a mask instead of the flat `(day, pos)` index
juggling and a selection metric that is meaningful for a cross-asset model (within-day rank IC, C1-compatible). `PanelModel` changes
are additive. Fixes: register `panel_tasks` through the existing task registry as ported (it does, via the `# noqa: F401` import in
`trainer.py`, which G8 sees as a hidden dependency: import it from `trainer/tasks.py` instead), derive K/targets from `HeadSpec`.
Value for the main line is moderate (the baseline is the flat model); it is needed for rolling-on-panel only.

C7 ADOPT-FIX (conditional) / HARVEST. It is well built and tested (causal-mask test, rollout test, loss tests, panel twin, shuffle control with
the *honest* null, persistence and always-up trivial baselines as C1 requires). But: (a) it is a different model family
(next-candle features in the model's own feature space), not a head of NIG-TimeNet, so it does not fit the current T6-T8 line;
(b) it carries two private training loops (G5); (c) per-call fitted `vol_floor`/`volume median` are not stored with the model (B4);
(d) `ENCODERS` duplicate (G1/G2). Port only if H0c opens the autoregressive line, as an experimental package with the second-loop
debt recorded as an explicit ratchet line ("second optimizer loops", 2) and a path to retire it by giving `Trainer` a step hook. The loss/rollout ideas are kept in
this document regardless (section 5).

C8 ADOPT-FIX. Every section is a pure function over a tidy table and tested with seeded fixtures. It carries precisely the
post-training checks `docs/hardening/AUDIT-2026-11.md` ("For H0c") says are missing, in trader terms: shuffled/flip tests
(leakage canary substitute), persistence baseline (C1), per-coin and weekly stability, shock split, cost-aware trade grids, confidence
buckets and an uncertainty filter. It is bound to the ar-stream table columns, so the work is an adapter from our `PredictionTable`
(`workflow/prediction_table.py`: mu = decoded close-return prediction, sigma from the NIG uncertainty, p_dir = class head),
a `C1` trivial-baseline test per new statistic, cards, and moving `cost=0.0008` and the quantile/percent defaults into a decision.

C9 ADOPT-FIX. Do not port `dump_run`; `run.py` already writes `test_predictions.parquet` and a summary JSON; the battery reads that
(or the in-memory table).

C10 ADOPT. They are thin, have no `def`, and are tested headlessly (the tests run every cell on `SOURCE="synthetic"`). Each travels
with its package so no package ships a notebook whose imports are not yet adopted.

C11 ADOPT-FIX and C12 ADOPT-FIX as above.

## 4. Is the ar-stream model a better route to sound confidence than the NIG heads?

What exists: three heads per position, `mu` (features), `log_sigma` (features), `dir_logit`. `log_sigma` is trained as an
**L1 pull** toward `log1p(|error| / local_vol)` with the error detached (`trainer/ar_stream.py:174-212`, weight `w_sigma=0.1`): a heuristic
scale predictor, not a likelihood (no NLL, no Student-t/NIG evidence, no `lambda` regulariser). It is evaluated in two places: the
battery's `confidence_buckets` (accuracy by `p_dir` bucket) and `uncertainty_filter` (does accuracy/PnL improve when only low-sigma
predictions are kept). There is **no coverage test, no PIT/ECE and no sigma-vs-realised-error calibration curve**, and the sigma of a rolled-out step
is conditioned on the model's own earlier predictions, so it conflates uncertainty with rollout drift. The current NIG heads have a calibration loss
term, floors (B-001), and `eval/target_report.py` + `eval/nig_floor.py` already measure coverage and sharpness. So the ar-stream
model is not a better route to *sound* confidence today; what it contributes is (i) the *evaluation* style (does the confidence
rank the errors: uncertainty filter, buckets, shuffled control with the honest null) which can be applied to NIG output via the battery
adapter, and (ii) the czar/Cauchy idea for the return loss (a shock is bounded, shrinkage to zero stays expensive), a candidate
alternative regression loss for H0c's T6 list. No result has been produced with it on real data (B-028 open on the branch).

## 5. Integration path

Rule for all: port (cherry-pick where it applies cleanly, else copy files from the worktree/branch) into a NEW PR on a branch
cut from `main`; every commit message says `ported from feat/ar-stream <sha> (owner's work)` with the original commit hash(es) from the table below; the
port converts docstrings to cards (once H1 is merged), regenerates its own pins (`python -m tests.defaults_pin --add`), writes
its own `D-nnn` for its new keys, runs `ruff check src tests`, `pytest -q`, updates the CSV registers. One change type
per commit.

Commit map (original sha -> package): `994618c` native data -> I2; `2de2846` loading -> I1; `d812141` panel -> I4;
`aa0f3dd` run -> I3/I4; `13125d9` named output -> I2; `fe55f05` notebooks -> with their package; `4fe8620` pins -> regenerate
(not ported); `904aa91` ar-stream -> I7; `8ba0665` loss kind -> I7; `f7c9af4` drive source -> I2/I7; `7c9736b` dumps +
battery -> I5; `f052cf6` classics + purge -> I5 (purge_package -> I1); `58652c3` INTERVAL -> I2; `351e2ba` notebooks end with battery -> I5/I7; `22d240e` SPLIT_AT -> I7.

**I1. Loading and config foundations** (`feat`, S, risk low). Files: `data_io/loading.py`, `npy_dir.load(mmap)`, `artifact.load_artifact(mmap)`,
`core/config.py` (`to_dict`, `updated`), `colab/bootstrap.py` (`purge_package`), tests `tests/data_io/test_loading.py`,
`tests/colab/test_bootstrap.py` + new direct tests of `to_dict/updated`. Fixes: D1 mandatory (unchecked load needs `allow_unchecked=True`); direct tests; card
docstrings. Decision: new D-nnn for `LoaderSettings` defaults. Place: right after H0c, **before H2** (H2's job reads the artifact through this loader).

**I2. Native data pipeline** (`feat`, L-ish: ~2.9k src + 58 tests; one session because it is a port with tests that already exist, risk medium). Files: all of `data/` except
`ar_windows.py`, `panel.py`; `pyproject.toml` (`[data]` extra, `nig-data` script, package data); `.github/workflows` install `.[dev,data]`; `tests/data/*`
(except ar/panel), `tests/fixtures/tf_parity_*`, `make_tf_parity.py`, `notebooks/data_pipeline.ipynb` + `test_pipeline_notebooks` data part. Fixes: CI extra; float test with `approx`; cards; `scan.py` layer line (`data`)
and the G7 layer table; `ratchets.json` unchanged; split D-006 keys for `data.*`. Acceptance: parity tests green, a 4h `split_dates=None` artifact built in a test and
loaded by the baseline path (this closes HF-15). Place: after H2 and **after H1** (so the 25 modules are carded as they land); H3 does not depend on it.

**I5. Evaluation battery + adapter** (`feat`, M, risk low). Files: `eval/battery.py`, `battery_classics.py`, `battery_lib.py`, `tests/eval/test_battery.py`, new
`eval/battery_adapter.py` (PredictionTable -> battery table), `notebooks/ar_stream_battery.ipynb` reduced to a model-agnostic `battery.ipynb` runner. Fixes: adapter + test on the
fixture NIG table; one `test_trivial_baseline_*` per new statistic (C1); defaults to a D-nnn (cost 0.0008, quantiles 0.3/0.5/0.7, percents 0.05/0.1/0.2); `stats/metrics` for any new metric (C2); cards.
Place: **right after H1, before I2** (needs only `eval/` and `stats/`; it is the first thing T5r/T7r results need).

**H3, H4** as planned (H3 registers the `ENCODERS` list that I7 later reuses; H4 produces the single `targets_config` that I3 consumes).

**I3. Run steps (flat single split)** (`refactor` + `feat`, M, risk medium). Files: `workflow/settings.py` (the new `RunSettings` tree, G4 1 -> 0), `run_common.py`, `flat_run.py`,
`run.py` (single split only), `jobs/baseline.py` rewired to ~30 lines over the steps, tests `tests/workflow/test_run.py` (single-split part) and the existing job tests unchanged
in outcome. Fixes: the trainer config comes from H4's `targets_config`; health report as a step (T4); `test_report.json`; `TS_COL`; `max_rows` smoke wired to the H2 smoke job; heads from `head_spec`.
Proof: `baseline-001` summary on the fixture is byte-identical before/after (refactor). Decision: `RunSettings` defaults. Place: **after H4**, before H5. `notebooks/main.ipynb` with it.

**I4. Rolling + panel on the run steps** (`feat`, M-L, risk medium). Files: `workflow/rolling.py` (+val_span), `trainer/rolling_fit.py`, `workflow/panel_run.py`, `model/grid_panel.py`, `model/panel.py` (`mix/heads`),
`trainer/panel_tasks.py`, `trainer/batching.py` + `Trainer` hunks, `eval/panel_eval.py`, `data/panel.py`, `selection_metric._panel_ic_fn`, tests for each, `notebooks/cross_asset_pipeline.ipynb`. Fixes: import `panel_tasks` from the task registry module (G8-visible),
K from `HeadSpec`, ratchet G8 1 -> 0 here, `PanelModel` unchanged-output test. Place: after I3, only if H0c keeps rolling/cross-asset in the roadmap (T6-T8 do not need it); otherwise park.

**I6 (merged into I7).** The SPLIT_AT/TEST_AT split and `ar_windows` are only used by the ar-stream line, so they travel with it.

**I7. ar-stream experimental package** (`feat`, L, risk medium, OPTIONAL). Files: `model/ar_candle_transformer.py`, `panel_ar_candle.py`, `trainer/ar_stream.py`, `panel_ar_stream.py`, `eval/ar_stream.py`, `workflow/ar_stream.py`,
`data/ar_windows.py`, 2 notebooks and tests. Fixes: register trunks (no second `ENCODERS`), the second training loop declared as ratchet debt with a retire plan (or a `Trainer` step hook),
store `vol_floor`/volume median with the model (B4), drop `dump_run` in favour of the run table, a calibration/coverage check for `log_sigma` (otherwise the head cannot be called uncertainty), cards, D-nnn for the 128 defaults. Place: last, after H6, **and only on H0c's go**.

**Campaign order (changed from H0's `H0b -> H0c -> H2 -> H1 -> H3 -> H4 -> H5 -> H6`)**:
`H0b -> H0c -> I1 -> H2 -> H1 -> I5 -> I2 -> H3 -> H4 -> I3 -> H5 -> H6 -> (I4) -> (I7)`.
What moves: I1 goes in front of H2; I5 and I2 after H1; I3 after H4 (before H5, so H5's ratchet re-base sees the final run steps and the `data` layer once); H5 stays high-overlap-free
because #25 is not merged; H6 refreshes facts after I3. I4 and I7 are optional and follow H0c's decision.

**What each H-package gains or loses** (also recorded in `PLAN.md`): H2 uses `data_io/loading.py` (I1) and its smoke job later becomes the I3 `max_rows` step; H3 gains "ENCODERS list lives in the registry, I7 reuses it"; H4's `targets_config` is the
single builder I3 consumes (the two #25 builders are not ported); H5 re-bases G4/G8 counts only after I3/I4 (they lower the ratchets themselves); H1's card check applies to every I-package;
HF-23, HF-29 and HF-15 are closed by I3, I3/I4 and I2.

## 6. Recommendation for PR #25

Do not merge it as is, and do not close it until I1, I2, I3, I5 are merged.
- Not as is: CI is red (missing `[data]` extra, one long line, one float test), ~6k lines of source and 47 modules would
  land with no cards before H1, the D1 loader bypass and two extra training loops would enter `main`, 422 pins and a 194-line decision file would be
  re-based under every H-package that touches `tests/pins/defaults.json`, `tests/ratchets.json`, `tests/scan.py` or `STATUS.md`, and
  the duplicated builders would make H3/H4 larger. The quality of the code is high and it is mostly additive (9 existing files touched, none of them behaviour-changing for the
  tensor path), so "merge as is, harden afterwards" is feasible if the owner prefers speed: it needs three trivial edits (CI extra, ruff, float test) and the hardening
  packages would then be written against the merged code. It is the riskier order.
- Keep it open as the reference source while the packages are ported; after I1, I2, I3, I5 are merged, close it with a comment listing the
  replacing PRs; keep the branch (or tag it `archive/feat-ar-stream`) for I4/I7. If the owner decides I7 is wanted soon, merging it as the last step (rebased) is also an option.
  The owner's authorship stays visible through the `ported from feat/ar-stream <sha>` lines.

## 7. Not verified

No real-data run of anything on the branch (B-028 there is open); `fetch.py` network paths; the TF parity fixtures were
not regenerated from TF `main`; GPU behaviour; the ten-versus-fourteen sections claim; notebook outputs beyond what the headless tests execute;
timing of the full suite on CI hardware (142 s here). `pandas-ta-classic` was installed into a scratch directory only.
