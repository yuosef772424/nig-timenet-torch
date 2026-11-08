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

## 2026-11-08 — fourth review (own docs PR; covers #41-#46: manager handoff + float32 fix, job files, T3b, T5r phase 1 / 2, G1)

Six PRs merged since the third review, and the STATUS counter still said 1 (hand-kept; third failure, H7 derives it).
Inputs: friction rows 99-112 (+ 113-116 written by this review), suggestions rows 130-161, bottlenecks B-047..B-056
(+ B-057..B-059 written by this review from the T5r error map), `docs/results/baseline-001.md` (T5r verdict: no T-profit
signal; high ~ HAR, low no skill beyond volatility). Row numbers are CSV line numbers (line 1 = header).

### Open bottlenecks ranked by gain / difficulty (goal: T-profit)

Gain = what it unlocks for profit after cost from any use of the outputs (directly, or by making a profit reading or a
gate verdict trustworthy). Difficulty S / M / L.

| rank | id | what | gain | difficulty | addressed by |
|---|---|---|---|---|---|
| 1 | B-040, B-056 | no cheap gate before a full run; Gate C's fixed band is unreachable for a symmetric high / low head (the TRUE mean / sd covers 0.88) | high: every later switch is judged by the gates | M | G2 (now; E1 high / low coverage relative to the oracle, manager decision) |
| 2 | B-052 (+ B-001, B-026) | NIG beta on its floor in ~98 % of high / low rows: std set by alpha alone, intervals too wide in calm rows | high: the uncertainty as a trade filter / size is one of the T-profit uses, and it shapes every NLL gradient | S-M | owner picks the route (below); G3r gives the evidence (E1 at scale 1.0 vs 100) |
| 3 | B-053 | battery LONG rows: side from p_dir, size from a mu of the opposite sign; limit-at-mu credits a falling long | high for MEASUREMENT (every LONG profit row is wrong), none for the model | S | a small fix PR with the martingale guard over every battery section (suggestions 151, 154); timing = owner question |
| 4 | B-057 (new) | high / low tie HAR; IC 0.11-0.20 inside the middle 60 % of volatility; rule D's last-range baseline is weaker than HAR | high for the limit-order uses at the predicted high / low | M | T6c switch (HAR inputs or HAR-residual target), card first; G2 shows HAR next to rule D (informational) |
| 5 | B-038 | memorization: val worse after epoch 1 on every head | high: picks the lever (capacity / data / regularisation) for every switch | M | M1 / M2r (unchanged) |
| 6 | B-059 (new) | market-wide shock days dominate the worst errors and every positive-looking strategy row | medium-high | M (artifact rebuild) | next artifact + T6c card |
| 7 | B-058 (new) | mu shrinks to the mean (slope 1.4-1.8) | medium: predicted extremes too close to the price | M | T6b (switch 6 / mu calibration) |
| 8 | B-043 | selection by loss, not by profit | medium | M | T6a switch 2 |
| 9 | B-018, B-041, B-007 | no clip / wd groups / EMA; wd a no-op; torch init | medium (wd is a memorization lever) | S-M | TR1 |
| 10 | B-025 | T-conf gap CI is an iid row bootstrap | medium: Gate B verdicts would be too lenient | S | G2: gate verdicts with the day-block CI (`stats.metrics.block_bootstrap_ci`) |
| 11 | B-005 | confidence loss with a diversity penalty | medium-low now (T-conf decile curve already monotone; Spearman(std, \|err\|) 0.35-0.42) | M | T6a switch 3 |
| 12 | B-048, B-024 | Drive sync / env cache never ran on Colab | medium runtime risk (one GPU session) | S | the owner's next Colab session of `main.ipynb` |
| 13 | B-054, B-055 | label clip at \|log\| 1; no volume in the artifact | low-medium | S each | next artifact export |
| 14 | B-002 | first projection may drop information | unknown | M | T6a switch 1 |
| 15 | B-008 | step time | low now (174 s / epoch measured, 16-minute runs) | S | TR2 (proposed later, below) |
| 16 | B-047 | job_id reuse across commits | low (unique-file test since #42) | S | H7 (check-job vs colab-results) |
| 17 | B-049, B-045, B-046, B-036, B-021 | development debt | low | S-M | when the module is next touched |
| - | B-020, B-022 | bar open time (settled internally), TF results not a clean holdout | information | - | none |

Status changes recorded today: B-019, B-027, B-034 closed (GPU path and health report ran; B-034 superseded by B-051);
B-047 mitigated (with the correction of friction row 102, see below); B-001, B-008, B-040 updated; B-057..B-059 new.

### Repeated frictions -> rule proposals

| friction (rows) | count | proposal | where |
|---|---|---|---|
| a corrected result leaves stale copies of the number in live docs: T-sharp 0.84 / 0.89 in STATUS (sugg. 153), ROADMAP T5r, PLANNER_HANDOFF, MANAGER_HANDOFF (row 116); "HF-13 settled" (sugg. 152) | 5 | **B6 extension (proposal):** a measured number has one home (`docs/results/*.md` or the run folder); live docs link it; a correction lists the superseded values in `tests/test_docs_claims.py`, which fails on a live-doc line quoting one without its B-id | H7 (check); rule text by the manager |
| the hand-kept R-REVIEW counter (rows 100, 115; sugg. 131) | 3 | no new rule: F3 enforcement derived from the PR numbers | H7 |
| fix commits inside other PRs skip F1 and the PR body: #21-#23 (first review), #41 (row 114: "docs only" with a test tolerance change) | 2 | **B1 wording (proposal):** the PR body lists every commit subject; a test-tolerance change is a behaviour change of the suite | rule text by the manager; manual |
| float32 parity drifts across runner CPUs (rows 93, 114) | 2 | no new rule (B3 class fix): one cross-machine float32 tolerance and a scan | H7 |
| a clean-up deleted another component's files (row 106, B-050), and a record inferred, not read, a fact about it (row 102 / 113) | 2 | no new rule: an `rmtree` ratchet in `colab/` (sugg. 145) | H7 |
| label / codec semantics had to be guessed from `price_norm_mode` (rows 107, 110; sugg. 155, 158) | 3 | no new rule: the contract now states it (`docs/DATA_ARTIFACT_CONTRACT.md`); the next export declares a label-codec key | next artifact |
| venv activation per shell call (row 112) | 1 | none (agent prompt carries it) | - |

### Rules: retire / amend
- Retired: none. E5 cost one job file per commit (B-047) and caught nothing this window, but its cost is a few minutes
  and the three failed rounds it targets cost hours. B7, G3, the pin diff: low cost, kept.
- Amend (proposal; CLAUDE.md is touched too, so the manager / owner applies it): **A1** says one step = one fresh
  session, but the owner's method (MANAGER_HANDOFF 2026-11-07) reuses and resumes one agent across consecutive steps;
  **A3** says a short report, the method asks for detailed agent reports to the manager. Proposed text: one step = one
  PR on a branch fresh from `main`; an agent may be resumed for the next step and then reads only that step; agent
  reports to the manager are detailed, the owner gets a short Arabic summary.
- Weaker than claimed: C6 stays manual until G2 builds the job-side refusal; F3's counter until H7.

### Suggestions decided (rows 130-161)
Every open row now carries `2026-11-08 review (fourth): ...`. 22 rows: accepted 13 (H7 6, G2 3, the B-053 fix PR 2,
F1 1, next artifact 1), declined 4 (132, 137, 144, 156, with the reason in the row), owner decision 2 (138, 150), done by
this review 3 (130: the injection-point list was reviewed, every exempt default in `src/` is a callable; 146; 155). The
rows marked `fixed` by their own PR (133, 134, 136, 142, 143, 147, 149, 152, 153, 161) are unchanged.

### ROADMAP after G2 in the light of T5r
Applied in `docs/ROADMAP.md` (no owner decision needed): H7 added after G2 (tooling only); G3r also runs E1 at
`reg_target_scale` 100 (D-018 parameter, CPU) as the evidence for the B-052 route; T6b lists B-052 / B-058 / B-056 as
inputs; T6c lists HAR inputs or a HAR-residual target (B-057) and cross-coin day context (B-059). Resulting order:
`G2 -> H7 -> TR1 -> G3r -> M1 -> M2r -> TR2 -> T5r-s1 -> T6a -> T6b -> T6c -> T7r -> T8 -> D1 -> D2 -> F1-F5`.

Proposed, NOT applied (the manager decides; no owner decision named):
1. Fold T5r-s1 into T7r: the T7r baseline arm runs seeds 0-2 on the T7r commit. Seed 1 on another commit cannot be
   paired with the T7r arms (C4: one commit), and a second seed of a no-signal baseline buys little now.
2. Move TR2 after T8: runs take ~16 minutes and B-008 is measured; the equal-output pipeline matters when runs get long.
3. Run the B-053 fix (with suggestions 151 / 154) before the next T-profit reading; it changes battery outputs only.

Needs an owner decision (listed, not reordered):
- **B-052 route:** (a) re-export the artifact at `reg_target_scale` 100 (the native default D-010; new fingerprint, so
  the baseline is re-run; can bundle B-054 unclipped prices, B-055 volume, a label-codec key (sugg. 158) and B-059 context
  features into ONE "next artifact" step), or (b) a model option (a beta floor scaled by the label variance, T6b switch 4)
  on today's artifact. G3r's E1 arms at scale 1.0 vs 100 show what each route changes.
- **Re-read the old runs after B-051?** Only `baseline-002-s0` was re-read with the right decode (T5r); the recorded
  high / low decodes of r2, s0, trial-002 stay wrong. If yes: a `jobs/reread.py` entry (S, sugg. 150).
- **B-053 timing:** before the next T-profit reading (proposal 3) or in T6a as planned.
- **B-056 for E3 and the real data:** a band derived from a symmetric oracle, a skewed / quantile interval (T6b), or a
  redefined band for high / low; until then the high / low coverage verdict there is informational.
- **Rule D's volatility baseline (B-057):** the last bar's range (protocol) vs HAR mean6 (stronger in T5r).
- **Market context (B-059)** needs a new artifact: part of the "next artifact" step or later.
- Sugg. 138: a write token to push from Colab. The A1 / A3 wording above.
- Still open from STATUS: O-2 (F2), O-3 (doctor before T7r; T5r found B-052 through a health-report floor share, which
  argues for doctor checks early), O-4, O-5, O-6, O-7 (T5r section 8 has the r2 / baseline-002-s0 curves).

### Stale facts fixed in this PR (rule B6)
- `docs/ROADMAP.md` T5r, `docs/PLANNER_HANDOFF.md` State, `docs/hardening/MANAGER_HANDOFF.md`: the misdecoded T-sharp
  share, "profit not measured", B-020 "unproven", D-015 / 749 tests as the latest, the done "Next" items: dated notes.
- `docs/PLANNER_HANDOFF.md` "next free: baseline-smoke-002": both smoke ids are used; the next commit needs a new file.
- `docs/MODEL_DEV_PROTOCOL.md` Gate C E3: "T-sharp share >= 0.80 for every head" contradicted O-1 (close informational).
- `docs/DATA_ARTIFACT_CONTRACT.md`: B-020 note; a section on what the TF `price_norm_mode` means for X and the labels
  (checked on the real artifact: X std 0.026-0.030, labels log vs last close clipped at 1.0).
- friction row 102 / B-047 ("holds the 44f6e71 checkpoint"): corrected by friction row 113 and the B-047 status.
- STATUS: G1 merged, counter reset, H7 row.

### Actions
- Next review after 5 more merged PRs (counter in `docs/STATUS.md`, derived once H7 lands).
