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
