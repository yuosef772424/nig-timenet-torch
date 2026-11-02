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

## G. Growth (problems that appear only as the project gets bigger)
Principle: every bad pattern is COUNTED and the count is pinned; CI fails if it grows (a ratchet that only shrinks).
The TF project added this late (`tools/check_patterns.py`, the implicit-deps allowlist) after the debt had piled up;
here the counts start near zero.

| # | Rule | Why (TF incident) | Enforced by |
|---|---|---|---|
| G1 | One source per concept: heads, targets and output names are defined once (`HeadSpec` / schema); every other list is derived from it. | TF C1: "a head" was defined in 4 vocabularies; each addition was an N-file edit | test: no output-name string literal outside the schema module (built in R0) |
| G2 | No branching on a mode name (`if mode == "..."`): behaviour comes from a registered component. | TF C4: 20+ `entry_range` branches | ratchet count of `== "<registered name>"` comparisons (built in R0) |
| G3 | Size budget: a module <= 400 lines, a function <= 80 lines; exceeding needs a one-line reason in the ratchet file. | TF: 700-line modules, a 180-line wiring function | ratchet in CI (built in R0) |
| G4 | No dead setting: every settings field is read by code. | TF C2: 19 of 101 config keys read nowhere | test over `core/config.py` fields (built in R0) |
| G5 | One system per job: one trainer, one evaluation path, one prediction table. New work extends them, never runs beside them. | TF: `cross_asset/` became a second trainer; evaluation had two producers | test: no `optimizer.step()` outside `trainer/` (built in R0) + review |
| G6 | Notebooks and studies are runners: no `def` in notebooks; a study never imports another study; shared code is lifted into the package. | TF: logic piled up in notebooks; research scripts copied package code | tests (built in R0) |
| G7 | Fixed layers: `core` imports nothing from the package; `stats` imports only `core`; `model` and `trainer` never import `workflow`. | TF tier rules (kept) | import-direction test (built in R0) |
| G8 | Adding a component = one new file + one registration line; no orphan file in a registry folder. | TF audit "lens": explicit registration beats auto-discovery | test: every module in a registry folder is registered (built in R0) |

## F. Self-correction (why these rules do not decay)
| # | Rule | Enforced by |
|---|---|---|
| F1 | Every session appends real obstacles to `docs/friction_log.csv` (what, where, minutes lost, root cause, fix idea). | manual |
| F2 | `docs/bottlenecks.csv` is the one register of what limits the project (model-quality / runtime / development): code location, evidence, cause status (verified / suspected / unverified), difficulty and why, step that addresses it. Rows are never deleted; wrong rows are marked `rejected` with the reason. | manual |
| F3 | R-REVIEW every 5 merged PRs (a short session): read friction + bottlenecks + suggestions, rank open bottlenecks by gain / difficulty, reorder ROADMAP, add a rule for any repeated friction, mark each new suggestions.csv row accepted (-> a ROADMAP step or a rule) or declined (with why), retire rules that cost more than they save. Each review appends a dated section to `docs/reviews.md`. | STATUS row counts PRs since the last review |
| F4 | Every rule row above has a "why"; a new rule without an incident or a check is not added. | review |
| F5 | Every step session, before finishing, appends to `docs/suggestions.csv` (and repeats in its final report): (a) at least one concrete suggestion to improve the system (code, rules, tools, workflow) learned while doing the step, and (b) every place where the work was hard because the logic was UNCLEAR (had to guess intent, undocumented behaviour, misleading name) or STALE (doc/comment/test that no longer matches the code, dead option). Each row: where (`path:line`), kind (`suggestion` / `unclear` / `stale`), what, cost (minutes lost), proposed fix. A `stale` finding that is a one-line fix is fixed in the same PR (rule B6). | review counts rows per PR; a PR with none must say "no suggestion" explicitly in its description |
