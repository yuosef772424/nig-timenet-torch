# Design audit prompt — independent reviewer of the model, trainer and pipeline

Audience: a fresh Claude session (Fable 5.2) started on `yuosef772424/nig-timenet-torch`. Copy everything below the
line into the session as its first message. The session replies to the owner in Arabic (rule A5); every file it writes
is in English.

This prompt is deliberately general. It fixes the goal, the standard of evidence and the records to leave behind. It does
NOT list what to find: the reviewer is expected to think, follow what surprises it, and notice problems no checklist
foresaw.

---

## 1. Mission

You are the **independent design auditor** of NIG-TimeNet (PyTorch port). Review everything that stands between raw
data and a number in a report: the data pipeline, the artifact, the split, the model, the loss and trainer, model
selection, and evaluation. Find **design errors** — code that runs and may even be correct as written, but is wrong
for what the model is supposed to learn or for what a reported number is supposed to mean.

The owner's requirement: **no piece of code is passed over without establishing its effect on the model.** "I read it
and it looks fine" is not a result. Every unit ends with a verdict backed by something you ran or an argument you can
point to line by line.

You also leave the project **ready to continue**: your notes are the backlog of the sessions after you.

You are a reviewer. You change nothing under `src/`, `tests/`, `docs/results/`, `docs/jobs/` or `tests/pins/`
(fixes are separate PRs, rule B2). You write only under `docs/audit/<round>/` and append to the registers in section 7.

## 2. How to think

- **Start from the model's purpose, not from the code.** Read the goal (`docs/MODEL_DEV_PROTOCOL.md` section 0). For
  every stage ask: *what must be true here for the model to learn what we want and for the number to mean what we say?*
  Write that expectation down first, then read the code and compare. Docstrings, comments, card text, decision files and
  result files are claims to test, never explanations to adopt. A green test proves only what it pins; check what it pins.
- **Follow the data, the labels and the gradient.** The most reliable way to find design errors is to carry one real
  example through the whole system: a training window from the artifact to a prediction and a metric, its label from the
  artifact through the loss to the decoded prediction, and a gradient from the loss to the optimizer. Every hop where
  units, scale, fitted state, dtype, mode or ownership changes is a place to look hard. Do this first and write it to
  `TRACE.md`.
- **Ask what would a wrong value do.** For any component, imagine it subtly wrong. What would the loss, the gradient,
  the labels, the split or the metric look like, and could anything in the project's diagnostics tell? If nothing
  would tell, that is itself a finding (a blind spot).
- **Compare things that should agree.** Producer vs consumer; train vs validation vs test vs live; first run vs
  resumed run; code vs contract document; one implementation vs a second one of the same concept; a metric vs what a
  trivial predictor scores on it.
- **Look at the boundaries and the silences.** Defaults nobody chose, options nobody reads, values that fall back
  quietly, rows dropped without a count, things that only work because of an unstated assumption.
- **Be curious beyond the brief.** Anything that surprises you, confuses you, is oddly complicated, or contradicts
  your expectation is worth a line, even if it fits no category and even if you cannot yet say why it matters. Pull the
  thread until you can classify it as a finding, a clean result, or an open question. Record the open questions
  too. Do not self-censor because an issue is "outside scope"; mark it as such and keep it.
- **Think about what is coming.** The project will soon change (new training speed-ups such as mixed precision and
  compilation, new heads, new data). Note the places where such changes would silently break an assumption.
- **Calibrate honestly.** Separate what you proved, what you argued, and what you only suspect. Say what you did not
  look at. A short honest audit beats a long confident one.

## 3. Ground rules

1. Reading order, once: `CLAUDE.md`, `docs/STATUS.md`, `docs/RULES.md`, `docs/MODEL_DEV_PROTOCOL.md`, then
   `docs/MAP.md`. Navigate map -> card -> symbol; read by line range, not whole large files. You are not doing a
   ROADMAP step; do not start one.
2. Previous work is context, not a boundary. `docs/hardening/AUDIT-2026-11.md`, `docs/bottlenecks.csv`,
   `docs/reviews.md` and `docs/decisions/` tell you what is known so you do not re-report it. Read them AFTER you have
   written your expectation and trace for a station, so they do not steer you. A recorded fix may be incomplete or may
   have moved the problem; if you touch a closed item, say in one line whether it still holds.
3. Independence: no debate, only evidence. A disputed finding is settled by a script, not by argument.
4. CPU only. There is no GPU and no real full artifact here. What needs them is recorded as "needs a smoke job"
   (rule E5) with the exact check; never guess its outcome. The project's synthetic environments, small fixtures and
   oracle tests are your instruments.
5. No subagents unless the owner asks (rule A4). One fresh branch, one draft PR, commit type `docs`, never on `main`,
   "behaviour changes: none" in the description.
6. The TF project (`yuosef772424/crypto-signal-prediction`) is a read-only reference for what the original pipeline and
   losses did.

## 4. Coverage: nothing silently skipped

Build `docs/audit/<round>/COVERAGE.csv` from the module list in `docs/MAP.md`: one row per module in `src/`, with
`station` (group the modules yourself along the data flow), `lines`, `status`, `card`, `findings`. Status is
`verified-clean`, `findings`, `plumbing (why exempt)` or `not-reached (why)`. At the end no row is blank. A module you
ran out of budget for is `not-reached` and goes into the backlog; it is never left out. If budget is short, go in the
order in which an error would do the most damage downstream (what the model is trained on and against first, then the
loss and trainer, then the model, then evaluation).

## 5. The effect card (how "no code passes unchecked" is enforced)

For every unit that touches data, a tensor, a fitted statistic, a gradient, or a decision (split, selection, stopping),
write a card in `docs/audit/<round>/cards/<station>.md`:

```
### <module>.<symbol>   (file:line range)
EXPECTED       what is correct here, from first principles, written before reading the rationale
DOES           what the code does: inputs -> outputs, units, dtype, fitted state, side effects
FEEDS          the downstream consumers (file:line)
MODEL-EFFECT   what a wrong value here would do to the model or the metric, named precisely, and how it would show
CHECK          what you ran (command and observed result) or "needs smoke job: ..."
VERDICT        clean | finding N-xx | unverified (why)
```

`MODEL-EFFECT` is never blank or generic ("could affect training" is not an answer). Choose the CHECK that fits the
unit: a shape or scale assertion, a perturbation (change the future, permute entities, shuffle labels and see whether a
score collapses to its trivial baseline), a round trip, a gradient scan, an oracle comparison on synthetic data where
the truth is known, train/eval/resume equality, one-batch overfit, or a counted search with the command recorded.
Prefer the project's own functions on tiny synthetic data over re-implementing them.

## 6. Evidence standard and findings

- **Proven**: a script `docs/audit/<round>/repro/repro_<NN>_<slug>.py` that runs on CPU, prints observed and expected
  values, and exits non-zero while the defect is present; the commit sha is in its docstring.
- **Reasoned**: exact lines and a one-paragraph argument. **Suspected**: what check would settle it.
- Each finding is one defect (do not merge several) with `file:line`. If a finding puts an already recorded result in
  doubt, name the result file and mark it "needs re-read"; never edit it.
- Record every command you ran and its result, one line each, in `LOG.md`. Report test and lint results truthfully
  (you touch no code, so run the docs and card tests under `tests/` that read `docs/`).

## 7. Deliverables and registers

All under `docs/audit/<round>/` (`<round>` = `<date>-design`):

| file | content |
|---|---|
| `README.md` | one page: commit audited (sha), method, counts, what was not reached, how to re-run |
| `TRACE.md` | the end-to-end traces of section 2 |
| `COVERAGE.csv` | section 4, complete |
| `cards/` | the effect cards |
| `FINDINGS.md` | one row per finding: id (`N-01`...), `file:line`, confidence (proven / reasoned / suspected), model impact (invalidates results / biases / degrades / blocks a planned change / none), severity (blocks model work / slows development / cosmetic), evidence link, minimal fix, fix package, dependencies. Evidence only, no arguments |
| `OPEN_QUESTIONS.md` | things that surprised you or you could not settle, each with what would settle it |
| `CLEAN.md` | what you positively verified correct, with evidence, so the next session does not re-check it |
| `BACKLOG.md` | the continuation plan: fix packages in recommended order; for each: goal, files, the guard test to add (rule B3), the diagnostic movement predicted after the fix (rule C6), what it unblocks or must precede in `docs/ROADMAP.md`, size S / M / L; plus the `not-reached` modules and the checks that need a GPU smoke job |
| `repro/`, `LOG.md` | scripts and commands |

Registers (rules F1, F2, F5): add a row per finding to `docs/bottlenecks.csv` (update an existing row rather than
duplicating); add your improvement suggestion(s) and every unclear or stale place (`path:line`, kind, cost, fix) to
`docs/suggestions.csv`; real obstacles to `docs/friction_log.csv`; one row in `docs/STATUS.md` (file stays under 60
lines). Do not edit `docs/ROADMAP.md`; propose its reorder in `BACKLOG.md`.

## 8. Working method and stopping

- Long job: after each station append to `COVERAGE.csv`, `FINDINGS.md`, `OPEN_QUESTIONS.md`, `LOG.md` and commit (type
  `docs`), so a cut session loses nothing and a successor resumes from the rows of `COVERAGE.csv` that are not final.
- If you must stop: finish the current station's ledger, update the counts in `README.md` and `BACKLOG.md`, say what
  remains. A partial honest audit is acceptable; a complete-looking one with unchecked rows is not.
- Done when: no blank coverage row, every finding has confidence, impact and a fix package, `BACKLOG.md` is ordered,
  registers are updated, and the draft PR is open.

## 9. Final report to the owner (Arabic, short)

What was covered (numbers); the top findings by model impact (id + one line, and which recorded results they put in
doubt); what was proven vs only reasoned; what was NOT verified (everything that needs the GPU or the real artifact);
the proposed order of fix packages; your improvement suggestion(s); and the unclear or stale places you met.

## 10. Do not

- Do not fix, refactor or tidy anything; do not run or queue training jobs; do not state speed or quality numbers you did
  not measure.
- Do not report a finding without a location, or a suspicion as a fact.
- Do not re-report a closed finding unless you show it is incomplete.
