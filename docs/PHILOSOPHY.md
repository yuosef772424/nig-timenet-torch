# How model work is done here: the working method (owner, 2026-11-16)

Owner observation, 2026-11-16: the sessions that moved the model forward were led by the strongest model the owner has
(Fable 5.2 at that date) and worked the way this file describes; sessions on other models that only walked a step
checklist did not. This file writes that way of working down so that every session follows it, whatever model runs it.
`docs/RULES.md` says what is forbidden and how it is checked; `docs/MODEL_DEV_PROTOCOL.md` defines the gates and the
metrics; this file says how to think, and in what order to spend effort. Where they disagree, RULES wins and the
disagreement is reported under `## Suggestions` of the PR.

## 1. Who does what

- **The lead session** decides what to try, reads the results, writes the conclusions and reviews everything before it
  is pushed. The owner opens model work (anything that changes or trains the model, or reads a result to decide the
  next experiment) on the strongest model. Builder work (tools, tests, docs, a job the owner named exactly) can run
  on any model.
- **Subagents** take the reading and the mechanical work on a cheaper model, so the lead's context stays free for
  decisions (owner, 2026-11-16). What they take and what they never do (decide what to try, judge a result, design a
  model or trainer change, push) is rule A4 in `docs/RULES.md`, the one place the rule is written.
- **A brief for a subagent** states the goal and what it is for, what "done" looks like, the exact paths and ids, what
  to return (short, verified facts apart from guesses) and what it must not touch. The agent works in its own
  `git worktree`, commits locally and never pushes. The lead reads the diff and spot-checks the numbers against
  their source before using or pushing them: delegated output is evidence, not truth.

## 2. The loop of every model question (cheap before expensive)

| # | Step | Where | Why |
|---|---|---|---|
| 1 | Write the question, the metric, the baselines it must beat and the accept rule BEFORE any number is read | a `PREREGISTERED.md` in the audit folder, or the job note | a rule written after the number is fitted to it |
| 2 | Known answer first: the synthetic environments (E1 single coin, E4 panel) and shuffled labels (E2) | CPU, minutes (`nig-envs`) | separates "the model cannot read the signal" from "the data hold none"; the arm is read against the ORACLE's score, not against 100 % |
| 3 | A small real look: a slice of the TRAINING era (E3, E3p) | CPU, minutes | the mechanism on real data; "a look, not a result" (rule C5) |
| 4 | The smoke job of the same entry on the real artifact, on the commit that will run | GPU, minutes (rule E5) | three 170-minute runs died on facts only the GPU or the artifact shows |
| 5 | The comparison: named arms (the baseline arm always among them) x >= 2 seeds, on ONE commit, in one job | GPU | a paired comparison finished in one sitting; one seed is a look (rule C4) |
| 6 | Walk-forward (entry `rolling`): the same arms over expanding folds, the test days of all folds pooled per seed | GPU | one split is one market regime; ~1260 pooled days give a CI that means something |
| 7 | Read: every number next to its trivial baselines, with a day-block CI, per seed, and the count of variants tried | `panel_report.md`, battery | rule C1; T-profit after costs is the goal, everything else explains it |
| 8 | An arm that fails its accept rule: the failure reading of section 3, BEFORE the next arm | the results note | a failed experiment that is not read teaches nothing; "the data hold no edge" is not a reading |
| 9 | Decide and record: results note, decision file for a default, STATUS row, bottleneck rows; a failed idea is recorded too | `docs/results/`, `docs/decisions/` | the next session must not repeat it |

A step is skipped only with a written reason. The gates of the protocol (rule C6) sit between steps 3 and 4.

## 3. When no arm wins: the reading that must follow (owner, 2026-11-16; this section has priority)

We are in the BUILD phase. An experiment is run to learn what the next update must change; a negative P&L ends an
arm, it does not end the question. The failure this section exists to stop (owner's observation of other sessions):
an experiment fails, the session moves to the next idea without reading what was gained, why it failed and which
samples it failed on, and closes with "the data hold no edge".

**"The data hold no edge" is not an accepted conclusion.** It may be written only when BOTH are shown on the same
rows: (a) a model-free bound (the oracle on the known-answer set, a ridge or a ranking rule on the real rows) finds
nothing either, and (b) the error map below shows no group of rows where the model is right beyond its trivial
baseline. Until then the finding is "this arm, read this way, on all rows, does not pay", and the work is to find out
what the model did understand and how to help it.

**The model will not trade every row.** In real trading only some samples are taken. So an arm is never judged on
all rows alone: it is also read on the rows a rule would select, and the rules themselves are part of the model.

Every arm that fails its accept rule gets a **failure reading** before the next arm starts. It is a part of the
results note (headings below, checked by `tests/test_docs_claims.py`), written from the saved prediction tables, not
from the summary numbers:

| # | Heading in the results note | What it answers | How |
|---|---|---|---|
| 1 | `## What the model learned` | which numbers ARE above their trivial baseline (rank IC per head, the sd ranking the error, a fold, a coin group), on every seed | the report tables next to the baselines; a known-answer set says how much of the planted signal was read |
| 2 | `## Where it is right and where it fails` | the error map: the same score per group of rows | groups: deciles of confidence, of predicted sd and of the size of the predicted move; long side against short side; coin groups (history length, volatility); market regime at the timestamp (up / down / flat, calm / volatile); fold and year; event class (market shock, specific jump, calm: card C-001). Per group: rows, days, hit rate, mean gross and net, CI, and the best trivial baseline on the SAME rows |
| 3 | `## The samples it failed on` | what the worst rows have in common | the rows with the largest confident wrong-sign errors and the worst days: coin, date, what the market did, what the model said; a table of concrete rows, not an average |
| 4 | `## Bias` | does it lean, where, and which way would be right | share predicted up against realised, per coin and per day; which side carries the loss; what a correction fitted on VAL (re-centring, a threshold per side) does on TEST |
| 5 | `## Selective use` | what a trading rule would keep | rules written BEFORE reading TEST, fitted on VAL, applied to TEST: top confidence, low sd, move over sd, agreement between the heads, a coin or regime filter. Report the share of rows and days kept against the net per trade (the coverage against profit curve) and count the rules tried (multiple testing). A rule found by looking at TEST is a hypothesis for the next walk-forward, never a result |
| 6 | `## Why it failed` | ONE named cause with its evidence | no signal (model-free bound) / signal smaller than the cost (IC and gross above 0, net below: a horizon and turnover question) / signal there but not read (the oracle or a ridge beats the model) / read but biased or badly calibrated / hidden by the selection metric or the evaluation / memorised (train against val gap) |
| 7 | `## Next hypotheses` | what to try and what would refute it | at least two hypotheses that follow from 1-6, each with the measurement that would prove it wrong and its cheapest test; they go into ROADMAP "Now" or a card under `docs/cards/` |

What is kept from a failed arm is written down too: a switch that moved a diagnostic the right way stays available
(off by default) even when the P&L did not move, and the next arm starts from the reading, not from a new idea.

## 4. Principles

1. **Arms, not a chain of single runs.** One change per arm, every arm a registered switch that is off by default
   (rule B1), all arms on one commit and one data fingerprint. A result that cannot name its baseline arm and its
   seeds is not reported as a result.
2. **Fill the GPU.** Independent jobs (arms, seeds, the baseline walk-forward next to the candidate's) run as separate
   processes at the same time when memory allows (`colab_ops rt` shows it); seeds are the OUTER loop, so a cut job
   still holds a complete seed of every arm.
3. **Every long job is resumable and keeps its outputs on Drive.** Checkpoint per arm / fold x seed under `run_dir`;
   a job that reaches its time limit exits cleanly and the SAME job is queued again. Nothing that matters lives only
   on the runtime or in `/tmp` (rule E2).
4. **Localise before fixing.** A symptom (a gap, a bias, a dead head) gets a measurement that discriminates between
   causes (a tap, the doctor, a model-free look) before any fix is proposed; the fix is then one switch, verified on
   the environment where the answer is known.
5. **Model-free first.** Before asking a network, ask a ridge / a ranking / a trivial rule on the same rows: it bounds
   what the data can give and exposes a metric a trivial model already wins.
6. **Read your own result as a sceptic.** Look for the cheapest explanation first: leakage, a baseline that was not
   run on the same rows, a selection metric that peeked, a direction bias (the model predicts "up" for 60 % of rows),
   a gain inside the seed spread.
7. **Say what was not verified.** The final report separates measured facts from expectations, and names every check
   that was not run.

## 5. Keeping the work safe

- Commit and push after every finished unit (a feature with its test, a report, a job file), not at the end. A
  session can die at any moment; 2026-11-16 the panel session ended in a failed turn with hours of work unpushed and
  an agent had to rebuild it from the transcript.
- A session with more than one part, or one that leaves a job running, writes `docs/handoff/<date>-<topic>.md` and
  pushes it BEFORE the long wait: what was asked, what is running where (job id, `run_dir`, how to poll it), what is
  decided, what is next. Never a secret (token, URL of a tunnel) in a file of the repository.
- Colab steps go through `python -m nig_timenet_torch.colab_ops` (`docs/COLAB_OPS.md`); a step done twice by hand
  becomes a recipe there.
