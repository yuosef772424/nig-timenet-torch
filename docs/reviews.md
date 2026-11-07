# R-REVIEW log (rule F3)

Each review appends a dated section: ranked open bottlenecks, repeated frictions turned into rule proposals,
decisions on new `docs/suggestions.csv` rows, rules to retire. Rows of the CSV registers are never deleted; their
`status` column carries the decision.

## 2026-11-07 — first review (run inside step H0, the hardening audit)

Overdue: rule F3 asks for a review every 5 merged PRs; this is the first, after 10 (R0 #14, T1 #16, T2 #17, T3 #18,
T4 #19, T5r #20, fixes #21-#23, #24). Inputs: `docs/friction_log.csv` (61 rows), `docs/bottlenecks.csv` (27 rows),
`docs/suggestions.csv` (43 rows), the H0 audit (`docs/hardening/AUDIT-2026-11.md`).

### Open bottlenecks ranked by gain / difficulty

Gain: what it unlocks for the model goal (confidence meaningful, uncertainty sharp, skill kept). Difficulty: S / M / L.

| rank | id | what | gain | difficulty | addressed by |
|---|---|---|---|---|---|
| 1 | B-024, B-027, B-033 | the GPU path past the first epoch (health report, probe, test evaluation) has never run; no short GPU job exists, so each GPU-only bug costs a 170-minute queue slot | high: every model result waits on it | S | H2 (smoke mode + rule E5), then the T5r result |
| 2 | B-034 | test / health reports decode prices with the "return" codec whatever the artifact's mode | high: T-sharp and coverage could be wrong | S | H2; read the real artifact's `reg_target_mode` from the first T5r result |
| 3 | B-001 | NIG variance parameters on their floor (TF: up to 96 % on the high head) | high (T-sharp) | M | ROADMAP T6 switch 4 |
| 4 | B-005 | confidence trained on batch-normalised error with a diversity penalty | high (T-conf) | M | T6 switch 3 |
| 5 | B-003 | best epoch chosen by the training objective | medium | S | T6 switch 2 |
| 6 | B-008 | GPU step time unmeasured (and the current measure includes validation and trace, HF-37) | medium (sets epochs / batch size) | S | T5r reading; H0c |
| 7 | B-028, B-029, B-030 | hard-coded artifact facts in the job; mode branching in the model builder; untyped trainer config | medium (every T6/T7 change touches them) | M each | H2, H3, H4 |
| 8 | B-020 | `last_candles` timestamp = bar OPEN time still unproven | medium (label timing, leakage) | S | T5r reading / H0c |
| 9 | B-018 | no grad clipping / weight-decay exclusions / EMA (TF used clip 1.0) | medium | S-M | H0c (trainer improvements) |
| 10 | B-007 | torch default init vs TF glorot / orthogonal | medium-low | S | after T5r, as a switch |
| 11 | B-002 | first learned projection may discard information | unknown until measured | M | T6 switch 1 (`direct_wide`) + the probe |
| 12 | B-010..B-012, B-015, B-017, B-032 | ratchets G1, G2, G3, G7, C2 and the blind spots of their scans | low directly, high for development speed | M | H3, H5 |
| 13 | B-013, B-016, B-035 | dead seams (RunSettings.rolling, model/panel.py) and the missing run-step layer | low now | - | H0b (PR #25 touches all of them) |
| 14 | B-021 | the npy_dir format has three copies (TF storage, TF export, torch) | low (format is stable) | M | later; H0b (PR #25 adds a native writer) |
| 15 | B-025, B-026 | iid bootstrap CI for T-conf; the 5 % floor rule | medium for reading, not for code | S | T5r reading |
| - | B-004 | high/low hit rate and class AUC are range mechanics | settled as a reading rule (C1 baselines in reports) | - | stays open as a reminder |
| - | B-022 | TF results are not a clean holdout | information only | - | none |

Status changes recorded in `docs/bottlenecks.csv` today: B-023 fixed (the real `.pkl.gz` export ran on Colab
2026-11-03, fingerprint `c572b1b36960`); B-024 partly closed (setup, export, device and publish ran on a T4; the
baseline job has not finished on the GPU). New rows B-028..B-036 from the audit.

### Repeated frictions -> rule proposals

| friction (line numbers in `docs/friction_log.csv`) | count | proposal | where it lands |
|---|---|---|---|
| CPU-only tests passed, the GPU / real-artifact run failed: P45 Kendall on CPU (row 38), T5r data not on the device (row 62), and PRs #21 (1h input), #22 (`split_dates` None), #23 (cuDNN eval backward), which logged no friction row at all | 5 | **E5 (new):** a long Colab job on a commit that has not yet run on the GPU is preceded by the smoke job of the same entry on the real artifact. **B3 extension:** a job reads every data fact (timeframe, targets, codec mode, split dates) from the artifact, and the CPU suite runs the entry on a TF-shaped variant artifact | H2 builds both (rule row, smoke job, variant test, guards) |
| fix PRs skip the friction log (#21-#23) | 3 | **F1 amendment:** "every session, including a fix PR opened by the planner, appends its real obstacles" | H5 (rule text); backfilled today as one friction row |
| sandbox without torch / pytest / ruff (rows 39, 48, 53) | 3 | not a rule: an environment setup script for the cloud environment (planner / owner, with the environment tools); this campaign already runs on a prepared venv | owner / planner (suggestions.csv row 3) |
| the B1 pin counts injection points and re-exports as defaults (rows 49, 55; suggestions row 28; audit HF-30) | 3 | tool fix, not a new rule: pin only module-defined names and exempt a documented list of injection-point parameters | H5 |
| plan or rule text did not state the operational meaning (rows 11, 32, 40; R0 suggestions rows 5-7) | many (port era) | each rule row names what is counted (already the R0 practice); keep it for new rows (H1's B7 and H2's E5 are written that way) | H1, H2 |
| two repos in one step (rows 52, 58) | 2 | none now: TF is frozen; PR #25 brings a native data pipeline (H0b) | - |

### Rules whose enforcement is weaker than the table claims (fix, do not retire)
- G2 counts only registered names (11 of 47 string comparisons; HF-26) -> H5.
- G3 does not cap an excused module (HF-25) -> H5.
- B5 / `core/config.py` docstring: the trainer config and the job config are dicts read with `.get` defaults
  (HF-14, HF-20) -> H2, H4.
- F3 counter is hand-kept and ran 5 PRs late -> H5 (STATUS check).
No rule is retired: none costs more than it saves yet.

### Suggestions decided
Every `open` row of `docs/suggestions.csv` (rows 2-44 of the file) now carries `accepted -> <package or step>` or
`declined (<why>)` in its `status` column, prefixed "2026-11-07 review:". 37 rows were open. Accepted into H1-H6:
13; into H0b / H0c / T5r reading / owner / later: 10; already done by later steps: 3; declined: 11 (mostly ROADMAP
wording of merged steps and edits to the frozen TF repo). Row numbers in this file and in `docs/hardening/` are line
numbers of the CSV (line 1 = header).

### Actions
- Next review: after 5 more merged PRs (counter reset in `docs/STATUS.md`).
- H0b decides PR #25 before any H package starts; H0c writes the post-hardening ROADMAP.

## 2026-11-07 — second review (run inside step H1; covers #27-#31: H0b, H0c, T4b, I1, H2)

New rows since #26: friction 67-78 (12), suggestions 64-93 (30), bottlenecks B-037..B-041 (+ B-033/B-034 updates).
Every open suggestion row now carries `2026-11-07 review (second): accepted -> <step>`, `done by <step>` or
`declined (<why>)` in its status column. Totals: accepted 19 (H5 7, I3 3, H3/H4 3, G1/G2 2, I5 1, H6 1, T5r reading 2),
done by a merged step 5, declined 2 (rows of the unmerged `feat/ar-stream`), owner decision 2.

### Repeated frictions -> rule proposals
| friction | count | proposal |
|---|---|---|
| test-support helpers hard-code a fixture fact (1h bar in `export_block`; `signal_sd=0` NaN; tests importing a helper from another test module) (rows 73, 74, 77) | 3 | not a rule: `tests/artifact_helpers.py` and base-aware helpers (H5 test tooling, suggestions 83, 91) |
| a count or plan fact was stale when the step started (RULES wording in T4b, allowlist 50 vs 37, B-041 id twice, E501 vs card lines) (rows 71, 72 and H1) | 4 | none new: B6 already covers it; H5 adds a register check (unique bottleneck ids) and H6 refreshes the audit/plan counts |
| the lint/length conflict of card fields (rows 72, I1 suggestion 85) | 2 | solved by B7 (wrapped fields, no E501 ignore) |
No new rule beyond B7 and E5 (both landed with H1 / H2). No rule retired.

### Rules whose enforcement is weaker than the table claims
Unchanged from the first review (G2, G3, B5, F3 counter): F3 ran on time this round (5 PRs, counter reset).

### Actions
- Next review: after 5 more merged PRs (counter reset in `docs/STATUS.md`).

## 2026-11-07 — third review (run inside step H6; covers #33-#39: I5, I2, H3, H4 + #37, I3, H5)

Seven PRs merged since the second review (rule F3 says 5; the campaign merged faster than the hand counter, suggestion
row 131 proposes deriving it). New rows since #32: friction 79-98, suggestions 94-130 (37), bottlenecks B-042..B-046.
Every open suggestions row now carries `2026-11-07 review (third): accepted -> <step>`, `done by <step>` or
`declined (<why>)`. Totals of the 37 + 3 older open rows: done 15, accepted 19 (T3b 4, F1 4, TR1 2, T5r 1, T6b 1, D1 1,
B-046 2, later 5), declined 6.

### Repeated frictions -> rule proposals
| friction | count | proposal |
|---|---|---|
| a plan or count fact was stale when the step started (G2 "<= 3", RunSettings sections, "fills defaults", allowlist) (rows 95, 119, 124, 128, 129) | 5 | no new rule (B6 covers it); H6 appended "Outcome" / as-built sections; future plans carry an as-built table (row 132) |
| the pandas-vs-settings-field G4 blind spot and pins by parameter NAME (rows 104, 120, 130) | 3 | none: counted in ratchets, reviewed here |
| the F3 counter ran 2 PRs over (rows 131 + friction) | 2 | derive it from PR numbers (row 131), not a rule |
No new rule. No rule retired: B7, E5 and the pin diff each caught or prevented something in this window.

### Rules whose enforcement is weaker than the table claims
G2 now counts declared vocabularies (honest 5, B-046); G3 caps exceptions (H5); B5/B-030 closed by H4; F3 counter still hand-kept.
C4 (>= 2 seeds) and C6 (gates) remain `manual` until ROADMAP G2 builds the job-side refusal.

### Actions
- T5r reading with a fresh `baseline-002-s0` (smoke first); next review after 5 more merged PRs (counter in `docs/STATUS.md`).
