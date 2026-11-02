# Working instructions (nig-timenet-torch)

Start of every session: read `docs/STATUS.md`, then `docs/RULES.md` (once), then ONLY your step in `docs/ROADMAP.md`.
The port record is `docs/BUILD_PLAN.md` (done; read a section only when a step points to it).
The reference TF project is `yuosef772424/crypto-signal-prediction` (frozen for model work; read-only except what
ROADMAP allows).

- Reply to the owner in Arabic; code, docs and prompts in English.
- No subagents unless the owner asks.
- Tests: `pytest -q` (fast suite). Run the tests of what you touched; CI runs everything. Lint: `ruff check src tests`.
- Before finishing:
  - update your STATUS row;
  - append real obstacles to `docs/friction_log.csv`;
  - update `docs/bottlenecks.csv`;
  - append to `docs/suggestions.csv` at least one improvement suggestion plus every place where the logic was unclear or
    stale (rule F5);
  - list behaviour changes in the PR description (rule B1).
- Final report to the owner (short, in Arabic): what changed, behaviour changes, checks run and their results, what
  was NOT verified, your improvement suggestion(s), and the unclear/stale places you met.
