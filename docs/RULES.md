# RULES — how this project is worked on (owner, 2026-11-02)

These rules are written so they do not go stale: every rule names the incident that taught it (mostly from
`yuosef772424/crypto-signal-prediction`, "the TF project") and HOW it is enforced. A rule with a check (test, CI
step, script) is kept by the machine; a rule marked `manual` is re-read at every review (R-REVIEW). A rule is never
deleted silently: the review marks it `retired (date, why)`.

The TF project stays readable at any time for reference and follow-up (its failure registry, results, decisions,
holdout lock, data pipeline). This project never edits its recorded results; it only reads it, and exports data
artifacts from it through its own tools.

## A. Sessions and context
| # | Rule | Why (incident) | Enforced by |
|---|---|---|---|
| A1 | One step = one fresh session = one branch = one PR. The session reads `CLAUDE.md`, `docs/STATUS.md`, then ONLY its step in `docs/ROADMAP.md`. | the TF planner conversation grew to 600k tokens | manual |
| A2 | Plans, STATUS and rules are merged to `main` BEFORE any step session starts. | TF S01: the plan existed only on a side branch; the step session could not find it | manual (STATUS says so in line 3) |
| A3 | A step session ends with a short report: what changed, behaviour changes, checks run with results, what was NOT verified. | long reports cost context; unverified claims passed as facts | manual |
| A4 | No subagents in step sessions unless the owner asks. | owner, 2026-11-02 | manual |
| A5 | Reply to the owner in Arabic; code, docs, prompts in English. | owner | manual |

## B. Changes and defaults
| # | Rule | Why | Enforced by |
|---|---|---|---|
| B1 | A new option's default reproduces the old behaviour exactly; a default change needs an owner decision file `docs/decisions/D-nnn-*.md` in the SAME PR, and the PR title/description lists every behaviour change. | TF PR #62 changed 3 defaults under an "add profiler" title | `tests/test_decisions.py` (built in R0): a change to the pinned defaults fingerprint without a new decision file fails |
| B2 | One change type per commit (`feat`/`fix`/`refactor`/`docs`/`test`/`exp`); a refactor proves equal outputs. | TF hardening discipline (kept, it worked) | manual + review |
| B3 | A fix of a CLASS of problem lands with a guard test that scans for the class, not only the instance. | TF S02: first pass fixed 5 of 16 sites; the AST guard caught the rest | manual (review asks "where is the guard?") |
| B4 | Every learned-state transform (normalizer, scaler, target codec, calibrator) is ONE registered component (`fit` on train only -> state, `transform`, `inverse`), stored with the model; consumers call it. Unknown names raise. | TF "one component, one interface" (kept) | `core/registry.py` + tests |
| B5 | Strict settings: unknown keys / unregistered names raise; no silent `.get(k, default)` for config. | TF C2: 19 dead config keys, a false "unknown keys raise" doc claim | `core/config.py` tests |
| B6 | Docs that state a fact about code are either generated or tested. A doc claim found false is fixed in the same PR that finds it. | TF `reference/README` "exactly as main.ipynb" was false (V1); `NOTES.md` here says "no PyTorch code yet" | `tests/test_docs_claims.py` (built in R0, grows per finding) |

## C. Measurement (the model must be judged by numbers that mean something)
| # | Rule | Why | Enforced by |
|---|---|---|---|
| C1 | Every metric used as evidence states what a trivial model scores on it (constant / last value / in-bar position / shuffled score). A metric that a trivial model already maxes out is not evidence. | TF: high/low "hit rate" 99 % for any model; class AUC 0.74-0.78 = range mechanics (F-0085) | a `trivial_baseline` test per metric in `stats/metrics.py` (built in R0) |
| C2 | One implementation per metric (`stats/metrics.py`); no private copies. | TF C7: AUC defined 2x, rank-IC 5x | grep test over `src/` (built in R0) |
| C3 | Best epoch is chosen by a registered selection metric with a fixed definition, independent of loss weights. | TF C7: selection = training objective | `trainer/selection_metric.py` tests |
| C4 | A comparison between arms uses >= 2 seeds per arm on one commit and a paired, dependence-aware test; one seed is "a look", never a result. | TF E-norm-001 seeds on different commits were discarded | manual (card template) |
| C5 | Exploratory results are labelled exploratory. Strict confirmation (holdout, forward data, realistic costs) happens only when the owner asks; until then the goal is to improve the model. | owner, 2026-11-02 | manual |

## D. Data integrity
| # | Rule | Why | Enforced by |
|---|---|---|---|
| D1 | Data enter this project only as a contract artifact (`docs/DATA_ARTIFACT_CONTRACT.md`) whose manifest records: source commit, coin list, sealed coins excluded (TF lock R0), max bar time < the time seal unless a confirmation card says otherwise. | TF: the main model pipeline never called `holdout_guard`; 718 coins in the dataset vs 677 allowed | loader refuses an artifact without these manifest fields (built in T2) |
| D2 | Features at time t use only bars closed at t; split boundaries purge by `label_end` + embargo. | TF leakage controls (kept) | `data_io/split_plan.py` purity tests |

## E. Runs and infrastructure
| # | Rule | Why | Enforced by |
|---|---|---|---|
| E1 | Every Colab job asserts the device it expects and writes `device`, torch/CUDA versions and GPU name into its output; a GPU job on CPU fails. | TF: months of "GPU" runs were on CPU (`tensorflow-cpu` pin) | job entry + output validator (built in T3) |
| E2 | Run outputs are published so the next session can read them without the owner copying anything; a failed publish keeps a retry path (`republish`). Never depend on `/tmp` surviving a long job. | TF profile-005: results stuck on Drive (askpass.sh in /tmp vanished) | runner tests (built in T3) |
| E3 | The fast test suite runs in < 5 minutes on CPU; slower tests are marked `slow` and run in CI only. Step sessions run the tests of what they touched. | TF: full suite > 50 min locally, died on session restart | CI `--durations` + a time budget check (built in R0) |
| E4 | Normal Python imports only (no shared-namespace loaders, no notebook globals). Notebooks are thin runners. | TF W1: implicit namespace edges made changes slow | `tests/test_package_imports.py` |

## F. Self-correction (why these rules do not decay)
| # | Rule | Enforced by |
|---|---|---|
| F1 | Every session appends real obstacles to `docs/friction_log.csv` (what, where, minutes lost, root cause, fix idea). | manual |
| F2 | `docs/bottlenecks.csv` is the one register of what limits the project (model-quality / runtime / development): code location, evidence, cause status (verified / suspected / unverified), difficulty and why, step that addresses it. Rows are never deleted; wrong rows are marked `rejected` with the reason. | manual |
| F3 | R-REVIEW every 5 merged PRs (a short session): read friction + bottlenecks, rank open bottlenecks by gain / difficulty, reorder ROADMAP, add a rule for any repeated friction, retire rules that cost more than they save. Each review appends a dated section to `docs/reviews.md`. | STATUS row counts PRs since the last review |
| F4 | Every rule row above has a "why"; a new rule without an incident or a check is not added. | review |
