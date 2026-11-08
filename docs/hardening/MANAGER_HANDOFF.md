# Manager handoff (2026-11-07, end of the hardening campaign)

For the next MANAGER session (the owner clears the conversation). Read this file, then `docs/STATUS.md`,
`docs/PLANNER_HANDOFF.md` (facts, Colab queue), and only then the ROADMAP step you are launching. Reply to the owner in
Arabic.

## Role and working method (owner, 2026-11-07)
- The manager only PLANS and ORDERS. Agents do the work, one at a time: never two agents at once. Each agent commits,
  opens a PR, waits for CI, and merges its own PR (`merge_method="merge"`, `expectedHeadSha`). The owner authorised this.
- Agent model: the hardening used the cheaper model. From ROADMAP block 2 on (model work), use the STRONGER model (`opus`)
  for every agent, for precision.
- No wake-ups without a reason. Each agent works in its own clone `/home/user/work/nig-timenet-torch`, so the primary checkout
  stays clean. It calls `mcp__claude-code-remote__unsubscribe_pr_activity` right after creating its PR. The manager
  unsubscribes from any `subscription.created` notice it still receives, and acts only on agent hand-backs.
- Agents write DETAILED final reports. The owner reads agent work rarely; the manager asks agents for changes through
  SendMessage when needed.
- Rules serve improvement, not restriction. Keep only what protects correctness: tests green, every new or changed default
  pinned with a decision file, recorded TF results never edited. Extra branches or PRs are fine when they help.
- Usage budget: when the limit hits, the agent stops with its work saved locally. Resume the SAME agent with SendMessage
  after the reset; do not start a new one.
- Colab: the manager queues a job only when the owner says so (see `docs/PLANNER_HANDOFF.md`, "How to queue a Colab job").

## Environment notes (cloud container; rebuild if missing)
- CPU torch venv: `python -m venv /home/user/.venv-torch && pip install --index-url https://download.pytorch.org/whl/cpu
  "torch==2.15.0+cpu" && pip install -e ".[dev,data]"`, run from the clone.
- Work clone: `git clone <origin url> /home/user/work/nig-timenet-torch`. Every agent starts with
  `git fetch origin main && git checkout -B <session branch> origin/main`.
- Baseline at handoff: `ruff check src tests` clean; `pytest -q` 749 passed, 1 skipped (~65 s) (839 passed / 2 skipped on 2026-11-08, c411008);
  `python -m tests.build_map --check` clean.

## Agent prompt template (fill the step; keep it self-contained)
1. Where to work: the clone path, `git fetch` and `checkout -B` from `origin/main`.
2. Read: `CLAUDE.md`, `docs/STATUS.md`, `docs/RULES.md` (once), `docs/MAP.md`, and ONLY the step in `docs/ROADMAP.md`. For
   model work also `docs/MODEL_DEV_PROTOCOL.md`: its gates run before any full training (rule C6).
3. Venv path and the baseline test count.
4. Scope: a summary of the step. The ROADMAP or protocol text wins if they differ.
5. Owner decisions that apply (below).
6. Before finishing:
   - update the STATUS row;
   - append to `friction_log.csv`, `bottlenecks.csv`, and `suggestions.csv` (rule F5);
   - add cards on touched modules (rule B7) and run `python -m tests.build_map`;
   - for any new default: a decision file with the next free D number (D-015 was the latest at this handoff, D-018 on 2026-11-08; D-006 is reserved for PR #25)
     plus `python -m tests.defaults_pin --add`.
7. Commit trailer lines, push, draft PR, unsubscribe at once, wait for CI with a background `sleep 120`, mark ready, merge.
8. Final report: detailed (PR, what changed, behaviour changes, checks with output, not verified, unclear/stale, open
   questions).

## Owner decisions in force
- GOAL = PROFIT by any use of the outputs: the T-profit north star, measured by the evaluation battery that every run now
  writes (`battery_report.txt`, `battery_profit.json`). T-skill, T-conf and T-sharp are means, not goals.
- Cost default: 0.0008 per side, i.e. 0.0016 per round trip. The owner may correct it.
- Direction is suspended for the high and low heads (never evidence, never a selection metric). Close direction counts
  only against its trivial baselines.
- O-1 is closed: close-head T-sharp is informational only. Open decisions O-2..O-7 are listed in `docs/STATUS.md`.
- PR #25 (`feat/ar-stream`) stays open as a reference.
  - Ported: I1, I2, I3, I5.
  - I4 split: its rolling half becomes ROADMAP F2; its panel half is parked.
  - I7 is not opened. Its useful ideas were harvested: the czar loss became switch T6b-6, and the battery came in I5.

## Where things stand
- Campaign: H0-H6 and I1/I2/I3/I5 merged, PR #26 to #40 (table in `docs/hardening/PLAN.md`, "Outcome").
- GPU: `baseline-001-s0-r2` (@44f6e71) is the reference run (numbers in `docs/STATUS.md`).
  - Validation is best at epoch 1 and worsens after it, while train loss keeps falling (memorization evidence).
  - Confidence works (decile Spearman ≤ -0.81 on all heads).
  - High/low T-sharp share is ≥ 0.80, but intervals are too wide (coverage 0.89 to 0.93). [2026-11-08: the share was a B-051
    misdecode; decoded right it is 0.012 / 0.272 on `baseline-002-s0`.]
  - Close rank IC is 0.057.
  - Profit is not measured yet: that run predates the battery. [2026-11-08: measured on `baseline-002-s0` by T5r: no use of
    the outputs is profitable after cost on VAL and TEST, `docs/results/baseline-001.md`.]

## Next (in this order)
[2026-11-08, fourth R-REVIEW: items 1 and 2 are done (smoke-002 and baseline-002-s0 ran @8451a62; T5r #44 / #45),
T3b #43 and G1 #46 merged. The order after G2 and the owner questions: `docs/reviews.md`, section 2026-11-08.]
1. If the owner agrees, queue `baseline-smoke-002` with `repo_ref` = current main, then `baseline-002-s0`
   (both job files exist in `docs/jobs/`; `baseline-smoke-001` cannot be reused: its results and run_dir exist).
   - `baseline-002-s0` (`docs/jobs/baseline-002-s0.json`): config of `baseline-001-s0` with a NEW run_dir, 5 epochs,
     about 16 minutes on a T4.
   - This gives the first real profit numbers.
2. Launch ROADMAP T5r (reading, stronger model) on those outputs.
   - Compute the volatility baseline for the high/low heads (rule D).
   - Read the battery.
   - Write `docs/results/baseline-001.md`.
3. Then the ROADMAP block 2 order: T3b, G1, G2, TR1, G3r, M1, M2r, TR2, T5r-s1, T6a-c, T7r, T8, D1, D2, F1-F5. One agent at
   a time, stronger model.
