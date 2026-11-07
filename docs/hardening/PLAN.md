# Hardening plan 2026-11 (H1-H6)

Written by the H0 audit (`docs/hardening/AUDIT-2026-11.md`, findings HF-01..HF-37). Order of work:
H0 (this audit) -> H0b (planner: read PR #25 `feat/ar-stream`, decide what to adopt) -> H0c (planner: the
post-hardening ROADMAP) -> H1 ... H6, one builder session at a time, one PR each, merged before the next starts.
H0b may reorder or drop packages; every package below lists the files PR #25 also changes so H0b can decide.
**Final order (H0c, 2026-11-07): see "Final campaign order" at the end of this file.**

Rules for every package (from `docs/RULES.md`):
- one change type per commit (B2 types: `feat` / `fix` / `refactor` / `docs` / `test` / `exp`); a file move is its own
  move-only `refactor` commit with no content change (TF rule "never move and modify together"); a refactor proves
  equal outputs;
- every behaviour change is listed in the PR description (B1); a moved, renamed, removed or new default needs a
  decision file `docs/decisions/D-nnn-*.md` in the same PR (`tests/test_decisions.py`). PR #25 already uses D-006:
  take the next free number at build time and renumber on a rebase;
- a module the PR touches gets a valid card once H1 is merged (H1 rule);
- run `ruff check src tests` and `pytest -q`; finish with the CLAUDE.md register updates (STATUS row, friction,
  bottlenecks, suggestions).

Ratchet targets of the whole plan (`tests/ratchets.json`, today -> after): G1 22 -> <= 6 (<= 2 with the optional
selection_metric part of H3), G2 11 -> <= 3 (then re-based on the wider H5 count), G3 modules 1 -> 0, G7 1 -> 0,
C2 1 -> 0. G4 (1) and G8 (1) are left to H0b: PR #25 lowers both to 0.

Sizes: S = under ~150 changed lines, M = ~150-500, L = more (none planned).

---

## H1 — Card convention and navigation map (`test` + `docs`, S-M, risk low)

**Goal.** A fresh session finds "where do I edit X?" from one generated page and the card of the module, and cards
spread without a mass edit (owner, 2026-11-07: "the .py files do not carry fixed tags like the old project; do NOT edit
them now — add them when a file is updated").

**(a) Card format** (module docstring, top of the file, fields in this order; existing prose stays below):
```
"""<one-line title, <= 100 characters>

PURPOSE:  <= 240 characters: what the module is for (not how).
TAGS:     >= 3 comma-separated search words: public names, concepts, registry kinds, rule ids, step ids.
PITFALLS: what breaks if edited carelessly (pinned names, parity attribute names, defaults pinned in
          tests/pins/defaults.json, device or dtype traps), or "none known".
SOURCE:   ported modules only: upstream path @ commit (+ BUILD_PLAN step) and what changed. KEEP it when a module
          already has a provenance line ("SOURCE:" or "source: crypto-signal-prediction ..."): convert it into
          this field, never drop it.
"""
```
Field continuation lines are indented (as in `core/registry.py`). Valid = PURPOSE, TAGS, PITFALLS present and
non-empty, PURPOSE <= 240 characters, TAGS >= 3 items, SOURCE present when the base version of the file had a
provenance line. `__init__.py` files are exempt (their docstring is the package index). While touching a module
without `__all__`, add one (HF-10).

**(b) Rule row** to add to `docs/RULES.md` section B:
`| B7 | A module touched by a PR gets (or keeps) a valid card (PURPOSE / TAGS / PITFALLS, SOURCE kept for ported
modules); no mass card edits: untouched modules are carded only when they are next changed. | TF navigation works
through cards and a generated map; the port arrived with 1 card in 61 modules (H0 audit HF-01); owner decision
2026-11-07 | tests/test_cards.py: allowlist tests/cards_allowlist.txt only shrinks; a module changed in the PR may
not be on it; docs/MAP.md is generated and checked |`
Plus one line in `CLAUDE.md`: "Navigate: `docs/MAP.md` -> the module's card -> the symbol (grep, then read by line
range)."

**(c) Machine check** — new `tests/test_cards.py` + `tests/cards_allowlist.txt` (one path per line, relative to
`src/nig_timenet_torch/`, sorted; initial content = every card-less non-`__init__` module = 50 lines today (51 modules, 1 carded)):
1. `test_cards_are_valid`: every module not on the allowlist has a valid card (parser shared with the map builder).
2. `test_allowlist_is_current`: every entry exists and has no card (an entry whose module got a card fails: remove it,
   so the list only shrinks, like `G3_size_exceptions`).
3. `test_allowlist_never_grows`: entries are a subset of the base branch's allowlist (`git show <base>:tests/cards_allowlist.txt`);
   a new module can never be allowlisted.
4. `test_touched_modules_have_cards`: changed modules = `git diff --name-only --diff-filter=AMR <base>...HEAD --
   src/nig_timenet_torch` plus uncommitted changes (`git diff --name-only HEAD`); none may be on the allowlist.
   Base resolution: env `CARD_BASE` (CI) -> `git merge-base HEAD origin/main` -> otherwise `pytest.skip("no base
   branch: run in CI")`. CI sets `CARD_REQUIRE_BASE=1`, which turns that skip into a failure.
5. Fake-package tests of the parser (valid card, missing field, long PURPOSE, SOURCE dropped), as `tests/scan.py`'s
   `test_scan_finds_*` do.
CI (`.github/workflows/ci.yml`): `actions/checkout` with `fetch-depth: 0`; env `CARD_BASE` =
`origin/${{ github.base_ref }}` on `pull_request`, `${{ github.event.before }}` on `push` (skip when it is the zero
sha), `CARD_REQUIRE_BASE: "1"`.

**(d) Navigation map — worth it, small.** TF's generated maps total 6,759 lines because of its research zones; here
one page of ~120 lines covers the package. `tests/build_map.py` (`python -m tests.build_map`, `--check`) writes
`docs/MAP.md`: per subpackage a table `module | title | TAGS` (TAGS empty until the module is carded); a
**Registries** table (kind -> module -> registered names, from the 7 `Registry` objects found by
`tests.scan.import_all`); an **Entry points** list (`jobs/*.py`, `__main__` modules, `notebooks/*.ipynb`). No line
counts, no timestamps, sorted output, so it changes only when a title, card or registration changes. Freshness:
`tests/test_cards.py::test_map_is_fresh` regenerates in memory and fails with the command to run. A separate tags
index file is not worth it at this size (the TAGS column is searchable).

**Scope (files).** `docs/RULES.md`, `CLAUDE.md` (one line), `tests/test_cards.py` (new), `tests/cards_allowlist.txt`
(new), `tests/build_map.py` (new), `docs/MAP.md` (new, generated), `.github/workflows/ci.yml`,
`docs/model_plugin_howto.md` (one line pointing to the registries table), `tests/test_docs_claims.py` only if a new
live-doc path needs it.

**Out of scope.** Writing cards for existing modules (H1 touches no `src/` file); per-zone maps; a tags file.

**Behaviour change.** None (no `src/` change; the pin collector scans `src/` only). No decision file.

**Acceptance.**
- `pytest -q tests/test_cards.py` green locally and in CI (CI log shows the touched-modules test ran, not skipped);
- `python -m tests.build_map --check` exits 0; `wc -l tests/cards_allowlist.txt` = 50;
- manual proof in the PR: a throw-away local edit of a card-less module makes `test_touched_modules_have_cards` fail;
- full `pytest -q` and `ruff check src tests` green; `tests/ratchets.json` unchanged.

**Depends on.** H0 merged. **PR #25 overlap.** No shared file. Consequence: after H1, every module PR #25 adds or
changes needs a card in the PR that adopts it (~40 new `src/` modules). If H0b adopts #25 (or parts) BEFORE H1, those
modules enter the initial allowlist instead; H0b should choose the order with that cost in mind.

---

## H2 — Job runs read the artifact; GPU smoke path (`refactor` + `feat` + `test`, M, risk low-medium)

**Goal.** Close the classes behind the three failed Colab runs (#21, #22, #23; HF-11..HF-17): a job reads every data
fact from the artifact, refuses what it does not understand, and the CPU suite plus a short GPU smoke job catch the
rest before a long run.

**Scope (files).** `src/nig_timenet_torch/jobs/baseline.py`, `src/nig_timenet_torch/data_io/split_plan.py` (TS_COL from
`core.schema`), `src/nig_timenet_torch/data_io/target_codec.py` (LAST_COLUMNS from `core.schema`, kept in `__all__`;
`reg_target_scale` required), `src/nig_timenet_torch/workflow/prediction_table.py` (line 34 only: `TS_COL`),
`docs/jobs/baseline-001-s0.json`, `docs/jobs/baseline-001-s1.json` (+ `test_report.json` in `expected_outputs`),
`docs/jobs/baseline-smoke-001.json` (new), `docs/RULES.md` (rule E5 row), `tests/colab/test_baseline.py`, a variant
artifact helper (e.g. `tests/data_io/variants.py`), `tests/test_hardcoded_guards.py` (new), `tests/ratchets.json`
(one new key), `tests/pins/defaults.json` + `docs/decisions/D-nnn-baseline-config.md`, `notebooks/README.md` (smoke
job paragraph).

**Content.**
1. One schema source (HF-12): `split_plan.TS_COL` and `target_codec.LAST_COLUMNS` become imports from
   `core/schema.py`; `prediction_table.py:34` uses `TS_COL`; the job takes anchors by column name
   (`LAST_COLUMN_INDEX[f"last_{t}"]` or `own_anchor_and_future`), never by position.
2. Artifact-driven job (HF-11, HF-13): `targets = dataset["price_targets"]` (must be a subset of
   `core.schema.TARGET_COLUMNS`, else raise); codec = `TARGET_CODECS[dataset["reg_target_mode"]]` (unknown raises
   UnknownNameError); `ReturnCodec.fit` raises when `reg_target_scale` is missing (contract key,
   `docs/DATA_ARTIFACT_CONTRACT.md:58`).
3. Strict job config (HF-14): `BaselineConfig` frozen dataclass built with `core.config.from_dict`: `device`, `seed`,
   `epochs`, `batch_size`, `run_dir` (required), `artifact=DEFAULT_ARTIFACT`, `embargo_hours=0.0`,
   `health_max_rows=DEFAULT_HEALTH_MAX_ROWS`, `split_dates=None`, `max_rows=None`. Values equal today's `.get`
   defaults. `device_smoke` gets the same treatment (one field).
4. Smoke mode (HF-16): `max_rows` (default None = off) evenly strides train / val / test index arrays down to
   `max_rows` each before the tensors are built; everything else is the same code path (fit with trace, health report
   with probe and nig_floor, test evaluation, summary, manifest). `docs/jobs/baseline-smoke-001.json`: device cuda,
   epochs 1, `max_rows` 20000, its own `run_dir`, `max_minutes` 15.
5. Rule row E5 in `docs/RULES.md`: "A long Colab job on a commit that has not run on the GPU is preceded by the smoke
   job of the same entry on the real artifact" — why: #21, #22, #23 each cost a 170-minute queue round; enforced by:
   `tests/colab/test_baseline.py` checks that every `docs/jobs/*.json` long job (max_minutes > 30) has a smoke job of
   the same entry; the planner queues it.
6. CPU variant artifact (HF-15): a helper copies the fixture artifact and rewrites it TF-shaped: `X_1h` -> `X_4h`,
   `timeframes=['4h']`, `base_tf='4h'`, `split_dates=None` (manifest and export block rewritten with the existing
   `write_manifest` test helper). `test_baseline_on_a_tf_shaped_artifact` runs `baseline.main` (1 epoch, `max_rows`)
   on it. In the PR description: show that re-introducing `"X_1h"` (#21) or the old `split_dates` read (#22) locally
   makes this test fail.
7. Guards (rule B3), `tests/test_hardcoded_guards.py`: (a) no assignment of `TS_COL`, `LAST_COLUMNS`,
   `LAST_COLUMN_INDEX` outside `core/schema.py`; (b) no integer column subscript (`[:, <int>]`) on an expression that
   names `last_candles`; (c) ratchet `G1_price_target_literals_outside_schema` = count of tuple / list / set literals
   whose items are 2+ distinct names of `TARGET_COLUMNS`, outside `core/schema.py` (11 today; 10 after H2 removes the
   job's tuple; H3 lowers it again by using `TARGET_COLUMNS` in model defaults); (d) every `torch.autograd.grad(` / `.backward(` call outside `Trainer.train_step` sits
   inside a `with` whose items include `torch.backends.cudnn.flags(enabled=False)` (the #23 sibling guard).
   Each guard has a fake-source test that it finds the pattern.

**Out of scope.** Run-step extraction from `baseline.main` (H0b decides, HF-23); `data_io/artifact.py` (PR #25 changes
it); `NIG_DRIVE_ROOT` (HF-17, later); multi-timeframe jobs.

**Behaviour change (list in the PR).** (1) the baseline refuses unknown or missing job config keys (before: ignored /
KeyError); (2) it refuses an artifact whose `reg_target_mode` is not registered, whose `price_targets` are not known
targets, or which lacks `reg_target_scale` (before: decoded as "return" with scale 1.0, fixed targets); for the real
artifact: none, if it is "return" with high/low/close — read its mode from the T5r output first; (3) the runner now
requires `test_report.json` for the baseline job files; (4) new pinned defaults (`BaselineConfig` fields, values
equal to today's `.get` defaults): decision file with `<new>` lines.

**Acceptance.**
- `pytest -q tests/colab tests/data_io tests/workflow tests/test_hardcoded_guards.py tests/test_decisions.py
  tests/test_rules_growth.py` green, then the full suite;
- `grep -rn 'TARGET_CODECS\["return"\]' src/nig_timenet_torch/jobs` -> nothing; `grep -rn 'TS_COL = \|LAST_COLUMNS = '
  src` -> `core/schema.py` only;
- `python -m nig_timenet_torch.colab check-job docs/jobs/baseline-smoke-001.json` -> OK;
- the #21 / #22 revert demonstration is in the PR description;
- ratchet file: new key present; G1..C2 unchanged or lower.

**Depends on.** H0b decision (if #25's `workflow/run.py` is adopted first, items 2-4 move into its settings instead of
`BaselineConfig`). **PR #25 overlap.** `workflow/prediction_table.py` (line 34; #25's hunks are elsewhere),
`tests/ratchets.json`, `tests/pins/defaults.json`; semantic: #25's `workflow/flat_run.py` also builds a job path with
its own `TARGETS`.

---

## H3 — Model assembly through registries (`test` + `refactor` + `feat`, M, risk medium)

**Goal.** A new encoder, head kind or input variant is one new class plus one registration line; `model/builder.py`
drops under 400 lines; the health report works for every architecture (HF-08, HF-09, HF-18, HF-19, HF-27, HF-28).

**Scope (files).** `src/nig_timenet_torch/model/builder.py`, `model/encoders.py` (new), `model/heads.py`,
`model/architectures.py`, `model/parts.py` (variant flags), `model/probe.py`, `stats/metrics.py` (+ `r2`),
`core/model_contract.py` -> `model/contract.py` (move), `eval/nig_floor.py`, `jobs/baseline.py` (probe point from
the model, two lines), `docs/model_plugin_howto.md`, tests that import `core.model_contract` (3 files),
`tests/stats/test_metrics_trivial_baselines.py`, `tests/model/test_state_dict_keys.py` (new),
`tests/ratchets.json`, `tests/pins/defaults.json` + decision file. Optional: `trainer/selection_metric.py:33-38`.

**Content.**
1. First commit (`test`): pin the sorted `state_dict` keys and parameter shapes of the built model for transformer,
   GRU, TCN, two timeframes, each `input_skip` variant, binary and multiclass heads (a JSON under `tests/pins/`).
   Every later commit must keep it identical (it guards the Keras attribute names the oracle tests load by).
2. `ENCODERS` registry (`model/encoders.py`): each entry builds its modules through the model's `add(name, module)`
   (same names, so same keys) and owns its forward over the token sequence; `NigTimeNetV2._build_branch` / `branch` /
   `encode_tokens` dispatch through it; `builder.ENCODERS` tuple and the if/elif chains go. Unknown encoder ->
   UnknownNameError (a ValueError subclass, so the existing test still passes).
3. Heads: `HEAD_REGISTRY` meta gains what the builder branches on today (output width: 1 for binary, `n_classes` for
   multiclass); the builder looks up the forward with `HEAD_REGISTRY[kind]` and validates kinds against
   `HEAD_REGISTRY.names()` (drop `HEAD_KINDS`); `heads.nig_outputs(spec, mu, nu, alpha, beta, epistemic, aleatoric,
   confidence)` and `heads.class_outputs(spec, probabilities)` build the output dict from `head_spec(...)` keys and
   are used by NigTimeNetV2 and TinyMLP. Optional (adjacent to PR #25's hunk): `selection_metric` takes its keys from
   `head_spec`.
4. Input variants: `model/parts.py` variant classes carry flags (`adds_modules`, `replaces_patch_embed`) so the builder
   no longer compares `input_skip` with "none" / "no_proj" (the "none" variant must still add no module).
5. Defaults from the schema: model target defaults reference `core.schema.TARGET_COLUMNS` (tuple stays tuple, list
   stays `list(...)`: the pinned values must not change). Builder / architectures cards state which default is THE
   default (registry `default_config`) and that the builder signature mirrors TF (HF-08).
6. `core/model_contract.py` -> `model/contract.py`: a move-only `refactor` commit (file only), then a `refactor`
   commit for the imports (G7 -> 0). Decision file: pinned key `nig_timenet_torch.core.model_contract.contract_of(input_name)`
   removed, `nig_timenet_torch.model.contract.contract_of(input_name)` added, same value.
7. `stats.metrics.r2` (+ `__all__`, + `test_trivial_baseline_r2`), `model/probe.py` uses it (C2 -> 0).
8. Architecture-neutral health report: each architecture implements `head_modules() -> {target: NIGHead}` and a
   `default_probe_points` tuple (v2: `("trunk_norm",)`; TinyMLP: its last trunk layer); `eval/nig_floor.py` and the job
   use them; a test runs `health_report` on a TinyMLP end to end; `docs/model_plugin_howto.md` lists both members.

**Out of scope.** Any change of model values or defaults; `model/panel.py` (PR #25); the TF-signature defaults of
`build_nig_timenet_v2` (they stay).

**Behaviour change.** None in outputs (all oracle parity tests and the state_dict pin unchanged). Unknown-encoder error
message changes (type stays a ValueError). Decision file for the moved `contract_of` key.

**Acceptance.**
- `pytest -q tests/model tests/core tests/eval tests/stats tests/trainer` green with no tolerance or oracle file
  changed (`git diff --stat` shows no `reference/` or tolerance edit), then the full suite;
- `tests/pins/<state_dict keys>.json` identical before/after;
- `wc -l src/nig_timenet_torch/model/builder.py` < 400 and `G3_size_exceptions` emptied;
- ratchets: G1 <= 6 (<= 2 with the selection_metric part), G2 <= 3, G3 modules 0, G7 0, C2 0, and H2's
  `G1_price_target_literals_outside_schema` lower than after H2;
- `python -m tests.defaults_pin` differs from the pin file only by the `contract_of` key (decision file).

**Depends on.** H2 (the job file). **PR #25 overlap.** None of #25's modified files except the optional
`selection_metric.py` part (adjacent hunk); semantic: #25's `model/ar_candle_transformer.py:18` defines its own
`ENCODERS = ("transformer", "gru")` — if adopted, H0b decides whether it registers into the same registry.

---

## H4 — Strict trainer config derived from HeadSpec (`refactor` + `feat`, M, risk medium)

**Goal.** A mistyped trainer option raises; the per-target trainer config is built in one place from HeadSpec
(HF-20, HF-21).

**Scope (files).** `src/nig_timenet_torch/trainer/targets.py` (new), `trainer/trainer.py` (`__init__` and the schedules
line of `fit` only), `trainer/tasks.py`, `jobs/baseline.py` (`trainer_config` becomes a call), `tests/trainer/`,
`tests/test_rules_growth.py` or a new guard test, `tests/ratchets.json`, `tests/pins/defaults.json` + decision file.

**Content.**
1. `trainer/targets.py::validate_trainer_config(cfg)` called first in `Trainer.__init__`: top-level keys within
   `{"targets", "loss", "constraints"}`; `loss` keys within `{"use_uncertainty_weighting",
   "uncertainty_weighting_exclude_task_types", "schedules", "constraints"}`; each target needs `true_key`,
   `task_type`, `output_keys` and may carry `loss_weight`, `flood_level` and the task's registered `config_keys`;
   unknown keys raise with a did-you-mean (the `core.config` message style).
2. Task option defaults move into the TASK_REGISTRY entry (`default_options` meta, e.g. `loss_fn="mse"`,
   `huber_delta=1.0`, `label_smoothing=0.0`, `binary=True`, `use_calibration_loss=False`, `lambda_reg_var`,
   `lambda_calib_var`); validation fills them, task functions read `target_cfg[k]` (no `.get` default left in
   `trainer/`). These are new pinned `registry:task:*` keys: decision file, values equal to today's literals.
3. Schedules parsed once in `__init__` with `from_dict(ParamSchedule, ...)` (suggestions.csv row 12);
   `scheduled_vars` start values come from the parsed objects.
4. `targets_config(head_types, *, options_by_kind)` builds the per-target dict from `head_spec(kind, target)`;
   `jobs/baseline.trainer_config()` calls it and a test asserts its output equals today's dict (a golden copy taken in
   the first commit).
5. Guard: a ratchet on `.get(<str>, <default>)` calls on config-like names in `trainer/` and `jobs/`, expected 0 after.
6. The Trainer card documents that Kendall weights only the non-excluded task types (suggestions.csv row 40).

**Out of scope.** Task weighting as a registry (HF-22, later); trainer behaviour; the RunIdentity format.

**Behaviour change.** Unknown trainer-config keys raise (before: ignored). New pinned defaults (task option table):
decision file. The RunIdentity of a run with the same config is unchanged (assert it in a test: identity
fingerprint of `jobs.baseline.trainer_config()` before == after).

**Acceptance.** `pytest -q tests/trainer tests/colab tests/test_decisions.py` green, then the full suite; the
golden trainer-config test and the identity-fingerprint test pass; the typo command of HF-20 now raises
(`ValueError` naming `label_smoothin`); the new `.get` ratchet = 0.

**Depends on.** H3 (head metadata), H0b. **PR #25 overlap.** `trainer/trainer.py` (different hunks, medium);
semantic: #25 adds `workflow/flat_run.py:41` and `workflow/panel_run.py:46` trainer-config builders and
`trainer/panel_tasks.py` task types: if adopted, they call `targets_config` / declare `config_keys` too.

---

## H5 — Rule and test tooling fixes (`test` + `refactor` + `docs`, M, risk HIGH vs PR #25)

**Goal.** The checks measure what the rules say, the B1 pin cannot be bypassed, tests share fixtures properly
(HF-25, HF-26, HF-30..HF-35) and the R-REVIEW rule proposals become rule rows.

**Scope (files).** `tests/test_rules_growth.py`, `tests/scan.py` (`mode_name_comparisons` only),
`tests/ratchets.json`, `tests/defaults_pin.py`, `tests/pins/defaults.json` + decision file, `tests/test_decisions.py`,
`.github/workflows/ci.yml`, `tests/conftest.py` and/or `tests/helpers.py`, the 11 importing test modules,
`tests/colab/test_baseline.py` -> `tests/jobs/test_baseline.py` (move), `tests/test_docs_claims.py`, `docs/RULES.md`,
`docs/decisions/README.md`.

**Content.**
1. G3: each size exception becomes `{"max_lines": N, "why": "..."}`; the test fails when an excused module exceeds
   `max_lines` (HF-25).
2. G2: also count comparisons against declared mode vocabularies (module-level tuples named `ENCODERS`, `*_MODES`,
   `LEVEL_NORMS`, `*_KINDS`, plus registered names); re-measure and record the new value (HF-26).
3. Pins: collect only constants defined in the module (AST assignment in that file), not re-exports; exempt a
   documented list of injection-point parameter names (`echo`, `run`, `env`, `secrets`, `argv`, `torch_mod`,
   `urlopen`, `version_of`, `accepted`, `ask`, `device_check`, ...); decision file lists every removed key (HF-30).
4. CI step on `pull_request` (`fetch-depth: 0`): every key added, changed or removed in `tests/pins/defaults.json`
   versus the base branch appears in a decision file added by the PR (HF-31; suggestions.csv rows 2, 15).
5. Shared fixtures (`art`, `write_manifest`, `make_trainer`, `identity_for`, `load_fixture`) move to helper modules /
   `conftest.py`; a guard fails on `from tests.<pkg>.test_` imports (HF-32). `tests/jobs/` gets the baseline tests
   (a move-only commit, HF-33).
6. `test_docs_claims.py`: the STATUS state check covers `H\d+[a-z]?` rows (HF-34); optionally fail when "PRs since
   last R-REVIEW" exceeds 7 (suggestions.csv row 10).
7. RULES text from the R-REVIEW (`docs/reviews.md`): G1 names `model/heads.py` (row 6), E1 states what a CPU job
   asserts (row 32), F1 includes fix PRs, B1 says who writes a decision file (row 33, also `docs/decisions/README.md`).

**Out of scope.** New rules beyond the review's proposals; `tests/scan.py` parts PR #25 changes.

**Behaviour change.** None in `src/`. Pin keys removed (re-exports, injection points): decision file.

**Acceptance.** `pytest -q tests/test_rules_growth.py tests/test_decisions.py tests/test_docs_claims.py` green; a
throw-away local edit of `tests/pins/defaults.json` without a decision file fails the new CI check (shown in the PR);
`grep -rn 'from tests\.[a-z_]*\.test_' tests` -> nothing; full suite green and its time within 10 % of today (31.8 s).

**Depends on.** H3 (G2 re-base after the builder refactor), H0b (merge order). **PR #25 overlap.** HIGH:
`tests/scan.py`, `tests/pins/defaults.json`, `tests/ratchets.json`; if #25 (or parts) are adopted, do H5 after that
merge and regenerate pins with `python -m tests.defaults_pin --add` on top.

---

## H6 — Refresh STATUS / PLANNER_HANDOFF / ROADMAP facts after the hardening (`docs`, S, risk low)

**Goal.** The working docs state what is true after H1-H5 (file names, ratchet values, rules, commands). H0c writes
the remaining plan; H6 only corrects facts.

**Scope (files).** `docs/STATUS.md` (hardening rows, ratchet line), `docs/PLANNER_HANDOFF.md` ("State", "How to
queue": smoke job first), `docs/ROADMAP.md` (fact corrections only, e.g. T6 switch 1 now lands in the input-variant
registry), `NOTES.md` (re-triage OPEN 1-10 into RESOLVED / open, HF-36), `docs/DATA_ARTIFACT_CONTRACT.md` header
(suggestions.csv row 20), `docs/bottlenecks.csv` statuses, `docs/reviews.md` (close the 2026-11-07 actions).

**Out of scope.** New steps or reordering (H0c); code.

**Behaviour change.** None.

**Acceptance.** `pytest -q tests/test_docs_claims.py` green; every path quoted in the live docs exists; STATUS < 60
lines; each hardening row has its PR link.

**Depends on.** H1-H5 merged (or dropped by H0b). **PR #25 overlap.** `docs/STATUS.md`, CSVs (append-only).

---

## Integration packages (from H0b)

Written by H0b (`docs/hardening/BRANCH_AR_STREAM.md`, full evaluation of PR #25 / `feat/ar-stream`). Adopting from #25 means
porting into new PRs (commit messages say "ported from feat/ar-stream <sha>"); PR #25 is not merged. Packages: **I1** loading
and config foundations, **I2** native data pipeline, **I3** run steps (flat single split, `jobs/baseline.py` becomes a thin
entry), **I4** rolling + panel on the run steps (optional), **I5** evaluation battery + adapter for our `PredictionTable`,
**I7** ar-stream experimental package (optional, only on H0c's go). Campaign order proposed by H0b (superseded by
"Final campaign order" below):
`H0b -> H0c -> I1 -> H2 -> H1 -> I5 -> I2 -> H3 -> H4 -> I3 -> H5 -> H6 -> (I4) -> (I7)`.

What changes in the H packages:
- **H2**: reads the artifact through I1's `data_io/loading.py` (D1 mandatory there); its `max_rows` smoke path later becomes
  the `max_rows` step of I3's `run.py`; HF-15's variant-artifact test can be built with I2's native pipeline.
- **H3**: the single `ENCODERS` source lives in the registry; I7 must reuse it (no second tuple).
- **H4**: `targets_config` is the ONE trainer-config builder; #25's `flat_trainer_config` / `panel_trainer_config` are not ported.
- **H5**: runs after I3 so the G4 / G8 / layer-table re-base is done once; #25's `scan.py` hunks are re-applied by I2 (`data` layer)
  and I4 (G8), not copied.
- **H1**: every I-package ports its modules with cards; H1 runs before I5 and I2 for that reason.
- **H6**: also refreshes the facts after I3 (HF-23, HF-29 closed there).
- HF-15 closed by I2, HF-23 by I3, HF-29 by I3/I4 (G4 1 -> 0, G8 1 -> 0).

---

## Final campaign order (H0c, 2026-11-07)

`H0 -> H0b -> H0c -> T4b -> I1 -> H2 -> H1 -> I5 -> I2 -> H3 -> H4 -> I3 -> H5 -> H6`, then the model plan in
`docs/ROADMAP.md` (Block 2 onward). One builder session per package, merged before the next starts.

- **T4b first** (a ROADMAP `fix`, S): the T-sharp reference in `eval/target_report.py` is wrong (owner correction,
  2026-11-07: per head, std vs the head's own predicted move from the last close). It must land before any T5r number
  is read, it is independent of the H/I packages, and its one overlap (the job passes `last_close`;
  `jobs/baseline.py:150,164`) is a two-line rebase for H2 (which rewrites those anchor lines for HF-12 anyway).
- **I4 split, not run in the campaign.** Its rolling half (rolling folds over the run steps through the one
  `Trainer`) becomes ROADMAP F2 (walk-forward stability before the final evaluation; owner decision O-2), built after
  T8 when there is a model worth folding. Its panel half (grid panel model, panel tasks, within-day IC selection) is
  parked in ROADMAP "Later" 1: T6-T8 and the memorization study do not need it, it adds a second sample shape to
  every H3/H4 registry, and the TF slot-panel idea must first be checked against failure-registry row F-0035.
- **I7 not opened.** Reasons: (1) the ar-stream `log_sigma` is a heuristic L1 pull toward `log1p(|error| / vol)`, not a
  likelihood, with no coverage test (BRANCH_AR_STREAM section 4): it is not a better route to the confidence and
  uncertainty targets than the NIG heads plus the methods of `docs/MODEL_DEV_PROTOCOL.md` section 4; (2) it has no
  result on real data (B-028); (3) it brings a second training loop (rule G5), 128 new defaults and ~1.4k source lines (L);
  (4) its useful ideas are harvested anyway: the czar / Cauchy-type loss is ROADMAP T6b switch 6, and the battery
  evaluation style arrives with I5. Reopen condition (ROADMAP "Later" 2): the confidence gate fails for every method
  of protocol section 4, or the owner asks. Keep the branch as `archive/feat-ar-stream`.

## Later (does not fit six packages; rows in `docs/bottlenecks.csv` / `docs/suggestions.csv`)
- TASK_WEIGHTINGS registry for Kendall and alternatives (HF-22, B-036).
- Move `identity_for_run` next to the run steps once H0b settles them (HF-24).
- Default artifact path from `NIG_DRIVE_ROOT`; one source for the job's device (`job.device` vs `config.device`)
  (HF-17).
- Train-loop timing separate from validation and trace (HF-37): scheduled in ROADMAP TR1 (log-only switch 6).
