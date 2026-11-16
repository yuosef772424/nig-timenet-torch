# Working instructions (nig-timenet-torch)

Start of every session: read `docs/STATUS.md`, then `docs/RULES.md` (once), then ONLY your step in `docs/ROADMAP.md`.
The port record is `docs/BUILD_PLAN.md` (done; read a section only when a step points to it).
The reference TF project is `yuosef772424/crypto-signal-prediction` (frozen for model work; read-only except what
ROADMAP allows).

- Navigate: `docs/MAP.md` -> the module's card -> the symbol (grep, then read by line range). A module you change gets a card (rule B7).
- Reply to the owner in Arabic; code, docs and prompts in English.
- No subagents unless the owner asks.
- Tests: `pytest -q` (fast suite). Run the tests of what you touched; CI runs everything. Lint: `ruff check src tests tools`.
- `docs/STATUS.md` stays under 60 lines (checked by `tests/test_docs_claims.py`).
- Before finishing: run `python -m tools.close_step --step <your id> [--title ...] [--pr N --state "PR open"]` (it
  regenerates `docs/MAP.md` and stale pins, drafts or updates the decision record, sets your STATUS row and prints the
  PR section "Behaviour changes (rule B1)"), fill every `TODO` it left in the decision, and finish with
  `python -m tools.close_step --check`. Then, in the PR description (`.github/pull_request_template.md`):
  - paste the generated B1 section and add any other behaviour change on its free-text line;
  - write your obstacles under `## Friction` and your suggestions under `## Suggestions` (at least one improvement
    suggestion plus every unclear or stale place, rule F5): do NOT edit `docs/friction_log.csv` or
    `docs/suggestions.csv`, `python -m tools.collect_registers` collects the sections after the merge;
  - update `docs/bottlenecks.csv` (unchanged: it is not collected).
- Final report to the owner (short, in Arabic): what changed, behaviour changes, checks run and their results, what
  was NOT verified, your improvement suggestion(s), and the unclear/stale places you met.
