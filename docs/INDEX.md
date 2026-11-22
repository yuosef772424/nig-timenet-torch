# INDEX of docs/ (generated: `python -m tools.build_index`; never edit by hand)

Where to look. Reading order: `CLAUDE.md`. Freshness: `python -m tools.build_index --check`.

## Standalone docs

Also `CLAUDE.md` (the reading order), `NOTES.md`, `docs/MAP.md` (code map, generated), the registers `docs/bottlenecks.csv`, `docs/friction_log.csv`, `docs/suggestions.csv`, and `docs/cards/`, `docs/hardening/`.

- `docs/BACKTEST.md`: Backtest: a separate historical test for any model and any strategy (owner, 2026-11-17)
- `docs/BUILD_PLAN.md`: BUILD_PLAN — PyTorch re-implementation of NIG-TimeNet v2
- `docs/COLAB_OPS.md`: Colab ops: the steps of every session (2026-11-15)
- `docs/DATA_ARTIFACT_CONTRACT.md`: Data-artifact contract (BUILD_PLAN.md P01)
- `docs/INFLUENCE.md`: INFLUENCE — the effect of each training batch and of each input feature (owner task 2026-11-17)
- `docs/MAP.md`: MAP — where do I edit X?
- `docs/MODEL_DEV_PROTOCOL.md`: MODEL DEVELOPMENT PROTOCOL — binding for every agent that touches the model (owner, 2026-11-07)
- `docs/PHILOSOPHY.md`: How model work is done here: the working method (owner, 2026-11-16)
- `docs/PLANNER_HANDOFF.md`: Planner handoff (rewritten 2026-11-07 by H6, after the hardening campaign; for a fresh planner)
- `docs/ROADMAP.md`: ROADMAP — from a verified port to a finished, honestly evaluated model (owner 2026-11-02; rewritten H0c 2026-11-07)
- `docs/RULES.md`: RULES — how this project is worked on (owner, 2026-11-02)
- `docs/STATUS.md`: STATUS
- `docs/model_plugin_howto.md`: How to add a model
- `docs/reviews.md`: R-REVIEW log (rule F3)
- `docs/roadmap_archive.md`: ROADMAP archive: step specs that are merged / done (moved verbatim from docs/ROADMAP.md, 2026-11-22; append-only, newest at the bottom)
- `docs/status_archive.md`: STATUS archive: rows and paragraphs moved out of docs/STATUS.md (2026-11-22; append-only, newest at the bottom)

## Results notes

36 notes (rule C7: the failure reading is inside each).

- `docs/results/arm-i-001.md`: arm-i-001: arm I of rolling-ind-001 (7 indicator channels, lr 1e-5) read against R (exploratory, rule C5)
- `docs/results/b1-btc-timing-001.md`: b1-btc-timing-001: buy BTC at the times the model is bullish across coins (owner 2026-11-19; exploratory, rule C5)
- `docs/results/baseline-001.md`: T5r — reading the baseline (`baseline-002-s0`, seed 0)
- `docs/results/batch-by-wf-001.md`: batch_by walk-forward on a 20-coin real sample (owner 2026-11-17) — a look, not a result
- `docs/results/batch-by-wf-002.md`: H1 / H2 of the batch_by study on 60 coins (owner 2026-11-18): H1 refuted, H2 supported — a look, not a result
- `docs/results/br1-001.md`: br1-001: BTC-relative inputs AND a BTC-relative close label, GRU, 2 seeds (owner 2026-11-19/20; exploratory, C5)
- `docs/results/btc-hedge-long-001.md`: btc-hedge-long-001: the long book minus BTC (owner 2026-11-19, the free test; exploratory, rule C5)
- `docs/results/btc-hedged-book-001.md`: btc-hedged-book-001: long the top coins against BTC, versus the market-neutral book (owner 2026-11-19)
- `docs/results/btc-relative-001.md`: btc-relative-001: BTC-relative inputs, GRU, 2 seeds (owner 2026-11-19; exploratory, rule C5)
- `docs/results/deep-side-001.md`: Deep side — is "the side where price goes deeper" predicted? (`baseline-002-s0`, seed 0)
- `docs/results/dominance-look-001.md`: dominance-look-001: USDT.D / BTC.D, and BTC + ETH instead of the cross-section as the panel's reference
- `docs/results/dw1-001.md`: dw1-001: day-balanced, failure-weighted loss (owner 2026-11-20; exploratory, C5)
- `docs/results/e1-fixes-001.md`: E1 fixes — where the mu scale and the close signal are lost, and the switches that fix it (FX1)
- `docs/results/gates-fixes-001.md`: Gate fixes — why A5 / A6 / the beta floor fail on E1, the switches, and the new run default (FX2)
- `docs/results/har-range-h3-001.md`: H3: the HAR range terms as a readout input (owner 2026-11-18) — refuted, a partial gain on the low head
- `docs/results/i1-001.md`: i1-001: the indicator channels trained like R (hypothesis I1 of arm-i-001; owner 2026-11-20; exploratory, C5)
- `docs/results/independent-review-001.md`: independent-review-001: an outside audit of the "skill but no profit after cost" reading (owner 2026-11-21; exploratory, C5)
- `docs/results/ir-001.md`: ir-001: arm IR of rolling-ind-001 (indicators + rank term + time blocks, lr 1e-5), the pre-registered candidate (exploratory, C5)
- `docs/results/lr1-cosine-001.md`: lr1-cosine-001: a cosine learning-rate decay in the walk-forward (exploratory, rule C5)
- `docs/results/ls1-side-heads-001.md`: ls1-side-heads-001: a loss per trade side (exploratory, rule C5)
- `docs/results/m3-full-001.md`: m3-full-001: method 3 (market residual) on the full artifact, GRU, 2 seeds (owner 2026-11-19/20; exploratory, C5)
- `docs/results/m3-improve-local-001.md`: m3-improve-local-001: verifying M3 and improving its information, local CPU (owner 2026-11-20; exploratory, C5)
- `docs/results/m3-local-001.md`: m3-local-001: market residual (method 3), a local CPU look (owner 2026-11-19; exploratory, rule C5)
- `docs/results/more-data-flat-001.md`: more-data-flat-001: does more training data help? All train rows against 400k on one flat split (owner 2026-11-19)
- `docs/results/nig-scale-default-001.md`: NIG label-scale default (MG2, B-071)
- `docs/results/no-instance-norm-001.md`: The H3 test again without the normalisation layer (owner 2026-11-18) — refuted, the same partial gain as H
- `docs/results/panel-001.md`: panel-001: the panel study on the full artifact, flat split (exploratory, rule C5)
- `docs/results/panel-rolling-std-001.md`: panel-rolling-std-001: the panel model with std target normalisation, walk-forward (owner task 2026-11-18)
- `docs/results/rolling-cpu-smoke-001.md`: Rolling walk-forward, CPU smoke check on E1 (ROADMAP F2, builder half)
- `docs/results/rolling-memorisation-001.md`: rolling-memorisation-001: does the walk-forward make the model memorise the first fold? + a per-coin cap proposal
- `docs/results/rolling-single-002.md`: rolling-single-002: the pre-registered tests on new seeds (exploratory, rule C5)
- `docs/results/short-side-001.md`: short-side-001: why the "top short" rows lose in every run (exploratory, rule C5)
- `docs/results/snapshot-panel-001.md`: Snapshot panel model ("100 x 3") against today's panel model: walk-forward on the full artifact (2026-11-18)
- `docs/results/staged-cpu-smoke-001.md`: Staged training (ROADMAP ST1), CPU smoke check on E1
- `docs/results/two-frames-001.md`: two-frames-001: the panel model with a 1h window beside the 4h window, walk-forward (owner 2026-11-18)
- `docs/results/v1rep-i2-001.md`: v1rep-i2-001: V1 on two new seeds, and M3 without its residual indicator channels (owner 2026-11-20; exploratory, C5)

## Audits

17 audit folders.

- `docs/audit/2026-11-14-model-first/`: Model-first audit — running notes (2026-11-14) (`NOTES.md`), 12 files
- `docs/audit/2026-11-16-panel/`: Report A: architecture of the panel model (panel audit 2026-11-16) (`A_architecture.md`), 6 files
- `docs/audit/2026-11-17-batch-by/`: PREREGISTERED — batch_by walk-forward on a small real sample (owner 2026-11-17) (`PREREGISTERED.md`), 11 files
- `docs/audit/2026-11-17-failure-charts/`: Audit: the samples the earlier models failed on, read on RAW charts (owner task 2026-11-17) (`REPORT.md`), 16 files
- `docs/audit/2026-11-18-alpha-beta-speed/`: Alpha / beta on their floors, and training speed (owner task 2026-11-18) (`NOTES.md`), 7 files
- `docs/audit/2026-11-18-panel-std/`: Pre-registration: the panel model with std normalisation, walk-forward (owner task 2026-11-18) (`PREREGISTERED.md`), 5 files
- `docs/audit/2026-11-18-snapshot-panel/`: Snapshot panel model: design (2026-11-18) (`DESIGN.md`), 5 files
- `docs/audit/2026-11-19-arch-indicators/`: Pre-registration: architecture and size, indicator channels, rank loss, lr 1e-5 (owner task 2026-11-19) (`PREREGISTERED.md`), 12 files
- `docs/audit/2026-11-19-lessons/`: Lessons from the experiments so far (owner request 2026-11-19) (`LESSONS.md`), 1 files
- `docs/audit/2026-11-19-loss-side-heads/`: LS1 pre-registration: a loss per trade side (written 2026-11-19, BEFORE any number of these arms) (`PREREGISTERED.md`), 12 files
- `docs/audit/2026-11-19-rolling-memorisation/`: , 8 files
- `docs/audit/2026-11-19-short-side/`: Pre-registration: rolling-single-002 (written 2026-11-19, before the job starts) (`PREREGISTERED.md`), 11 files
- `docs/audit/2026-11-20-dw1-day-weights/`: Pre-registration: DW1, day-balanced and failure-weighted loss (owner 2026-11-20) (`PREREGISTERED.md`), 7 files
- `docs/audit/2026-11-20-m3-improve/`: Pre-registration: improving the information of method 3 (M3), a LOCAL CPU study (owner 2026-11-20) (`PREREGISTERED.md`), 2 files
- `docs/audit/2026-11-20-readings/`: Pre-registration I1: the indicator channels trained like R (owner 2026-11-20) (`PREREGISTERED_I1.md`), 5 files
- `docs/audit/2026-11-20-v1rep-i2/`: Pre-registration: V1-rep and I2 (owner 2026-11-20: "run the proposals, in parallel, free RAM under 10 GB") (`PREREGISTERED.md`), 3 files
- `docs/audit/2026-11-21-independent-review/`: , 13 files

## Handoffs

11 handoffs (each carries a `Status:` line: open or closed with a pointer).

- `docs/handoff/2026-11-16-panel-session.md`: Status: closed (PN1 / WF1 #61; `docs/results/panel-001.md`)
- `docs/handoff/2026-11-18-batch-by-60.md`: Status: closed (`docs/results/batch-by-wf-002.md`)
- `docs/handoff/2026-11-18-panel-std-rolling.md`: Status: closed (PS1 #67; `docs/results/panel-rolling-std-001.md`)
- `docs/handoff/2026-11-18-snapshot-panel.md`: Status: closed (`docs/results/snapshot-panel-001.md`)
- `docs/handoff/2026-11-19-arch-indicators.md`: Status: closed (AI1 #71; `docs/results/i1-001.md`, `docs/results/b1-btc-timing-001.md`)
- `docs/handoff/2026-11-19-loss-side-heads.md`: Status: closed (LS1 #70; `docs/results/ls1-side-heads-001.md`)
- `docs/handoff/2026-11-19-rolling-single-002.md`: Status: closed (SS1 #68; `docs/results/rolling-single-002.md`, `docs/results/short-side-001.md`)
- `docs/handoff/2026-11-20-br1-m3-running.md`: Status: closed (AI2 #72; `docs/results/br1-001.md`, `docs/results/m3-full-001.md`)
- `docs/handoff/2026-11-20-dw1-running.md`: Status: closed (DW1 #73; `docs/results/dw1-001.md`)
- `docs/handoff/2026-11-20-start-here.md`: Status: closed (AI1 #71 and AI2 #72 merged in MG4 #74; `docs/results/br1-001.md`, `docs/results/m3-full-001.md`)
- `docs/handoff/2026-11-21-hardening-2026-11b.md`: Status: open (the campaign runs; close this note when HB8 merges)

## Decisions

43 decisions (format: `docs/decisions/README.md`).

- `docs/decisions/D-001-trainer-fit-defaults.md`: D-001 — trainer fit: schedule, early-stopping and fit defaults (ROADMAP T1)
- `docs/decisions/D-002-colab-runner-defaults.md`: D-002 — Colab runner and notebook support: new defaults (ROADMAP T3)
- `docs/decisions/D-003-t4-diagnostics-defaults.md`: D-003 — T4 diagnostics: new defaults (ROADMAP T4)
- `docs/decisions/D-004-baseline-job-defaults.md`: D-004 — T5r baseline job entry: new defaults (ROADMAP T5r)
- `docs/decisions/D-005-baseline-test-evaluation.md`: D-005: test-split evaluation in the baseline job
- `docs/decisions/D-007-loading-defaults.md`: D-007: dataset loading defaults (package I1)
- `docs/decisions/D-008-baseline-config.md`: D-008: baseline job config (package H2)
- `docs/decisions/D-009-battery-defaults.md`: D-009: battery and T-profit defaults (package I5)
- `docs/decisions/D-010-native-data-pipeline-defaults.md`: D-010: native data pipeline defaults (package I2)
- `docs/decisions/D-011-model-contract-module-move.md`: D-011: model contract moves from core/ to model/ (package H3)
- `docs/decisions/D-012-trainer-config-defaults.md`: D-012: trainer config validation, task option defaults and loss defaults move into registries (package H4)
- `docs/decisions/D-013-run-steps-defaults.md`: D-013: run steps (flat single split) and the profit battery in the run (package I3)
- `docs/decisions/D-014-pin-scope.md`: D-014: the defaults pin covers only what the module itself defines (package H5, finding HF-30)
- `docs/decisions/D-015-artifact-timeframes.md`: D-015: the artifact manifest declares its interval and timeframes; a job can refuse another interval (package H5)
- `docs/decisions/D-016-main-notebook-drive-env.md`: D-016: main notebook, Drive sync and the cached environment (ROADMAP T3b)
- `docs/decisions/D-017-t5r-reading-and-logret-codec.md`: D-017: T5r reading tools, the log-close target codec and the decode guard (ROADMAP T5r)
- `docs/decisions/D-018-test-environments.md`: D-018: test environments E1 / E2 / E3 and the E1 generator defaults (ROADMAP G1)
- `docs/decisions/D-019-gates.md`: D-019: gate tooling, gate job and the job-side gate check (ROADMAP G2)
- `docs/decisions/D-020-deep-side-check.md`: D-020: the deep-side check (new derived target, read-only) and two new metrics
- `docs/decisions/D-021-training-switches-taps-versions.md`: D-021: training switches, model taps, versioned Colab results
- `docs/decisions/D-022-e1-fix-switches.md`: D-022: E1 fix switches (FX1): BatchNorm warm-up, detached close mean, NIG head scale, readout dropout, beta-NLL
- `docs/decisions/D-023-gate-fixes-switches.md`: D-023: gate fixes (FX2): loss balancer, Kendall start value, NIG trunk-gradient scale, fitted NIG label scale
- `docs/decisions/D-024-restore-last-weights.md`: D-024: `workflow.run.restore(weights=...)`: read the last finished epoch instead of the best one
- `docs/decisions/D-025-exclude-coins.md`: D-025: `split.exclude_coins`: drop named coins from train, validation and test
- `docs/decisions/D-026-model-first-defaults.md`: D-026: the default model gets MODEL_FIRST_FIXES (model-first audit 2026-11-14)
- `docs/decisions/D-027-steps-per-execution.md`: D-027: `steps_per_execution` in the trainer (Keras option, speed only)
- `docs/decisions/D-028-cross-coin-option.md`: D-028: cross-coin option of the default architecture (`cross_coin_layers`, `SplitSettings.cross_section`)
- `docs/decisions/D-029-audit-new-options.md`: D-029: new options and helper defaults of the model-first audit (PRs #55 and #56), all at the old behaviour
- `docs/decisions/D-030-staged-training.md`: D-030: staged training (ROADMAP ST1, card F): new options, all off or at the old behaviour
- `docs/decisions/D-031-rolling-run.md`: D-031: rolling walk-forward on the run steps (ROADMAP F2, rolling half of I4): new options at the old behaviour
- `docs/decisions/D-032-panel-day-metrics-cross-section-loss.md`: D-032 — Panel (multi-coin) model: audit, day metrics, E4, cross-section loss, Colab jobs (owner 2026-11-16)
- `docs/decisions/D-033-ln1-target-norm-legacy-eval.md`: D-033 — Legacy TF evaluation ported, target normalisation, panel logret_std arm, failure-chart audit (owner 2026-11-17)
- `docs/decisions/D-034-if1-defaults.md`: D-034 — Batch influence during training + feature permutation importance (owner 2026-11-17)
- `docs/decisions/D-035-ab1-defaults.md`: D-035 — Alpha / beta floors of the Drive models, floor-report unit fix, speed arms + lite model, notebook (rolling both models, target_norm, plots, stop-and-keep-weights), time-block batches (owner 2026-11-18)
- `docs/decisions/D-036-ps1-two-timeframes.md`: D-036 — Panel walk-forwards: std target normalisation, LS selection, two timeframes 4h + 1h (owner 2026-11-18)
- `docs/decisions/D-037-ss1-scorecard.md`: D-037 — Short-side reading + day scorecard in the battery (owner 2026-11-19)
- `docs/decisions/D-038-ai1-defaults.md`: D-038 — Indicator channels (RSI, stochastic, volume) + architecture / size look + rank-loss walk-forward at lr 1e-5, long-only reading (owner 2026-11-19)
- `docs/decisions/D-039-dw1-day-weights.md`: D-039 — Day-balanced, failure-weighted loss + walk-forward on Colab (owner 2026-11-20)
- `docs/decisions/D-040-ls1-side-heads.md`: D-040 — Loss per trade side: side-payoff and upper-quantile heads on the close (owner 2026-11-19)
- `docs/decisions/D-041-hb1-correctness.md`: D-041 — Hardening 2026-11b, package HB1: correctness fixes found by the structure audit (owner 2026-11-21)
- `docs/decisions/D-042-hb2-run-identity.md`: D-042 — Hardening 2026-11b, package HB2: run identity and resume safety (owner 2026-11-21)
- `docs/decisions/D-043-hb3-evaluation.md`: D-043 — Hardening 2026-11b, package HB3: evaluation, one implementation per trading rule and helper (owner 2026-11-21)
- `docs/decisions/D-044-hb4-model-trainer.md`: D-044 — Hardening 2026-11b, package HB4a: model and trainer clean-ups (owner 2026-11-21)

## Job files

61 job files in `docs/jobs/` (a job file is queued by the planner, `docs/COLAB_OPS.md` "Queue route"), per entry:

- entry `baseline`: 5
- entry `panel`: 5
- entry `rolling`: 49
- entry `staged`: 2

## One-off scripts under docs/

52 scripts kept under docs/ (one-off analyses next to the note or audit they belong to; not run by CI).

- `docs/audit/2026-11-14-model-first/colab_follow.py`: Follow, resume and read the multi-coin (panel) arms on a Colab runtime, without the bridge.
- `docs/audit/2026-11-14-model-first/quick_arms.py`: Small-sample arms on a contract artifact: encoder (gru / transformer) x learning rate, same rows and seed.
- `docs/audit/2026-11-16-panel/normalisation_look.py`: Report C, model-free: what each normalisation of the close label does on a contract artifact (no network, numpy).
- `docs/audit/2026-11-16-panel/single_coin_balance_look.py`: Report B, owner idea 2026-11-16: does a direction-balance penalty help the SINGLE-COIN model?
- `docs/audit/2026-11-17-batch-by/analyse.py`: The pre-registered reading of the batch_by walk-forward (PREREGISTERED.md, accept rule (a) + (b)), from the runner's
- `docs/audit/2026-11-17-batch-by/failure_reading.py`: The failure reading (PHILOSOPHY section 3) of the batch_by walk-forward, from the saved TEST tables of every fold x
- `docs/audit/2026-11-17-batch-by/h1_check.py`: H1 of PREREGISTERED_2.md: A2 (timestamp batches) vs A0 (rows 256) and vs C34 (rows 34) on the low head's rank IC by
- `docs/audit/2026-11-17-batch-by/h2_bound.py`: H2 of PREREGISTERED_2.md: the model-free bound on the SAME test rows of every walk-forward fold, against the model
- `docs/audit/2026-11-17-batch-by/h3_check.py`: H3 of PREREGISTERED_3.md: A0 vs H (har_range) on one split (train 2023 / test 2024), per quarter of the test year,
- `docs/audit/2026-11-17-batch-by/run_arms.py`: Runner of the batch_by walk-forward (docs/audit/2026-11-17-batch-by/PREREGISTERED.md): one arm per call, so the
- `docs/audit/2026-11-18-alpha-beta-speed/scripts/alpha_beta_read.py`: Run a trained best.pt (from Drive) on a contract artifact and read its NIG alpha / beta against their floors.
- `docs/audit/2026-11-18-alpha-beta-speed/scripts/alpha_trace.py`: Train the default model from scratch and trace alpha / raw-alpha / softplus slope per head every k steps.
- `docs/audit/2026-11-18-alpha-beta-speed/scripts/speed_arms.py`: Speed arms on CPU. Pre-registered (2026-11-18, before any number): a fast arm passes when its wall time per epoch is
- `docs/audit/2026-11-18-panel-std/read_std_rolling.py`: Reading of the panel std walk-forward (owner task 2026-11-18): candidate vs reference, paired by fold and seed,
- `docs/audit/2026-11-18-snapshot-panel/scripts/snapshot_look.py`: CPU look (known answer first, PREREGISTERED.md): base panel model vs the snapshot model (joint / hybrid) on an
- `docs/audit/2026-11-19-arch-indicators/b1_btc_timing.py`: B1 of docs/results/btc-hedge-long-001.md: buy BTC at the timestamps where the model is bullish across coins.
- `docs/audit/2026-11-19-arch-indicators/hold_exit.py`: Amendment 3 of PREREGISTERED_M3.md: enter in the top 10 % of mu, exit only below a lower VAL quantile (hysteresis).
- `docs/audit/2026-11-19-arch-indicators/m3_local.py`: M3 local look (PREREGISTERED_M3.md): the ridge bound and the reading of arms A (raw) and M (market residual).
- `docs/audit/2026-11-19-arch-indicators/m3_readd.py`: M3 amendment 2: the market put back on M's coin selection (V1, V2) against arm A's long book (V0), long only.
- `docs/audit/2026-11-19-arch-indicators/read_arms.py`: Read the arms of the 2026-11-19 architecture / indicator study (PREREGISTERED.md in this folder) and draw charts.
- `docs/audit/2026-11-19-loss-side-heads/e1_known_answer.py`: LS1 known-answer test on E1 (spec: PREREGISTERED.md section 1): arms base / R / S x seeds, 3 epochs, CPU.
- `docs/audit/2026-11-19-loss-side-heads/e1_readout.py`: LS1 E1 diagnostic: can a linear read-out of the trunk h rank the oracle payoff as well as the closed form of mu?
- `docs/audit/2026-11-19-loss-side-heads/read_ls1.py`: LS1 walk-forward reading (spec: PREREGISTERED.md section 2, amendments 1-2): books P1 / P1L / P3 / P4 per arm,
- `docs/audit/2026-11-19-rolling-memorisation/btc_book.py`: BTC-hedged book against the market-neutral book (pre-registered addendum 4, owner 2026-11-19), from the saved TEST
- `docs/audit/2026-11-19-rolling-memorisation/btcref.py`: Model-free look (owner 2026-11-19): BTC alone as the reference, dispersion over all coins except BTC and ETH.
- `docs/audit/2026-11-19-rolling-memorisation/coins.py`: Per-coin concentration of P&L and losses (owner question 2026-11-19: a cap per coin), on the pooled TEST frames.
- `docs/audit/2026-11-19-rolling-memorisation/curves.py`: 
- `docs/audit/2026-11-19-rolling-memorisation/dominance.py`: Model-free look (owner 2026-11-19): USDT.D / BTC.D proxies and a BTC + ETH reference instead of the cross-section
- `docs/audit/2026-11-19-rolling-memorisation/memo.py`: Memorisation in the walk-forward (owner 2026-11-19): per-fold learning curves from the checkpoints, the year
- `docs/audit/2026-11-19-rolling-memorisation/usdt_supply.py`: Model-free look (owner 2026-11-19): does the growth of USDT supply (DefiLlama, daily, free) predict the market's
- `docs/audit/2026-11-19-rolling-memorisation/usdt_supply_after.py`: Model-free look (owner 2026-11-19): does the growth of USDT supply (DefiLlama, daily, free) predict the market's
- `docs/audit/2026-11-19-short-side/crowded_days.py`: Short top 10% /mu/: day P&L by the number of shorts that day (usage: <tables folder>).
- `docs/audit/2026-11-19-short-side/h1_h3_rules.py`: Price-unit short rules R0 / H1a / H1b / H3 / H13 on saved battery tables (spec: PREREGISTERED_H1_H3.md).
- `docs/audit/2026-11-19-short-side/hedge_and_stop.py`: Top 10% /mu/: market-hedged trades and a stop at k x sigma on the realised high (usage: <tables folder>).
- `docs/audit/2026-11-19-short-side/read_short_side.py`: Read the short side of `rolling-single-001` (7 expanding folds x 2 seeds) from the saved battery tables.
- `docs/audit/2026-11-19-short-side/rules_and_buckets.py`: Mean-corrected side rule and sigma / confidence quintiles, pooled over the 7 folds (usage: <tables folder>).
- `docs/audit/2026-11-20-dw1-day-weights/dw1_detail.py`: DW1 failure-reading detail (rule C7): which TRAINING days DW weighted up, and the TEST error map by market day.
- `docs/audit/2026-11-20-dw1-day-weights/dw1_read.py`: Read the DW1 arms (PREREGISTERED.md in this folder): day-loss tail, day-IC tail, long book, paired per-day diffs.
- `docs/audit/2026-11-20-m3-improve/m3_improve.py`: M3 improve (PREREGISTERED.md, 2026-11-20): leakage check, model-free variants (H3 ridge, H4 beta) and the reading.
- `docs/audit/2026-11-20-readings/ir_days.py`: Why does IR's long book pay per DAY but not per TRADE? Day net against the number of trades of the day.
- `docs/audit/2026-11-20-readings/m3_v1_read.py`: Sceptic reading of V1 (amendment 1 of PREREGISTERED_M3_FULL.md): is the market-put-back book coin choice or timing?
- `docs/audit/2026-11-20-readings/read_detail.py`: The failure reading of rule C7 from the saved tables of rolling arms (2026-11-20; arm I, BR1, M3).
- `docs/audit/2026-11-20-v1rep-i2/drop_channels.py`: Write a contract artifact with a SUBSET of the input channels of another one (hypothesis I2 of i1-001: channel
- `docs/audit/2026-11-21-independent-review/a1_reproduce.py`: 
- `docs/audit/2026-11-21-independent-review/a2_trim_fairness.py`: Is 'mean day without the best 5 % of days' a fair test? Apply it to benchmarks and to a placebo (random books).
- `docs/audit/2026-11-21-independent-review/a3_portfolio.py`: Real money view: two 8 h sub-books (phases 0/8/16 and 4/12/20), each holds half the capital, equal weight on the
- `docs/audit/2026-11-21-independent-review/a4_sizing.py`: Fixed-slot sizing (each selected coin gets 1/S of the sub-book; the rest stays in cash), fee on turnover,
- `docs/audit/2026-11-21-independent-review/a5_placebo_fees.py`: 
- `docs/audit/2026-11-21-independent-review/a6_timing.py`: The model's breadth (share of coins with mu >= VAL q90) used as the EXPOSURE of one instrument: BTC, or the
- `docs/audit/2026-11-21-independent-review/load.py`: Shared loader: the 7 TEST folds of one seed of rolling-dw1-r-001 (arm R), each with its VAL 90th pct of mu.
- `docs/cells/backtest_single_coin.py`: ==== nig_backtest on the SINGLE-COIN model (4h bars, horizon 8h), traded on 15m candles ====
- `docs/results/cpu_smoke_st1_f2.py`: CPU smoke runner of ROADMAP ST1 (staged study) and F2 (rolling study) on the synthetic environment E1 (rule C5: a
