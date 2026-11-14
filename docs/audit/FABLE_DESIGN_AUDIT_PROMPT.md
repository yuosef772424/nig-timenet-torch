# Prompt — find what stops the model from learning, fix it, and build the instruments to keep checking

Audience: a fresh Claude session (Fable 5.2) started on `yuosef772424/nig-timenet-torch`. Copy everything below the
line into the session as its first message. The session replies to the owner in Arabic; files and code it writes are in
English.

Owner decision (this version): the session may **edit anything** — code, tests, docs, pins, registers — and the
project's working rules (`docs/RULES.md`, the model protocol, card / ratchet / decision-file requirements) are
**guidance, not constraints**: ignore any of them that slows the work. The rule is simple: when you notice a problem,
either fix the code or write the documentation, whichever is faster and clearer, and move on.

---

## 1. Mission and goal

The goal is a model that is **ready for trading** (profit after costs, the owner's goal in
`docs/MODEL_DEV_PROTOCOL.md` section 0). If it is not ready, we must be able to answer *why*, quickly, from code, not
from guessing. So your job has two parts that feed each other:

1. **Find and fix** what in the model, the training, the data and anything else that touches the model keeps it from
   learning or keeps its numbers from meaning what we think.
2. **Leave instruments behind**: runnable diagnostic code plus a short **card** for each (what question it answers, how to
   run it, how to read the result), so that when a model fails in the future we can see *exactly where information was
   lost or gained*, whether *the trainer is guiding the model to learn*, and whether *the data is sound* — then fix that
   one thing and run everything again. The loop we want to make cheap:

   `run the checks -> the report says which stage is the problem -> fix it -> run the checks again -> compare`

You are free to change defaults, structure and tests when that is the right fix. Keep the old behaviour reachable only
if it is cheap (an option) — it is not required.

## 2. Order of work

Start with **the model and the training** (they decide whether anything is learned), then **the data**, then
evaluation and everything else that influences the model (selection, splits, targets and their decoding, configuration
defaults, precision, resume). Go deep where you find something; do not sweep everything shallowly first.

You may read the other project, `yuosef772424/crypto-signal-prediction` (the TF original: data pipeline, loss
definitions, its failure registry and results), whenever it helps — to compare behaviour, to see what was already tried
and why it failed, or to take a fix. Edit it only if the root of a problem is there; say so when you do.

## 3. How to think

- **Start from what must be true for the model to learn.** For each stage write down what should hold (an input carries
  information about the target; a layer keeps it; the loss rewards the right thing; the update actually moves the
  weights in a useful direction; the validation metric picks a model that is better for the goal). Then check, with a
  run, whether it holds. Docs, comments and "it passes tests" are claims, not proof.
- **Follow one example end to end**: a training window from the artifact to a prediction and a metric; its label through
  the codec, the loss and back to a decoded prediction; a gradient from the loss to every parameter. Wherever units,
  scale, fitted state, dtype, mode or ownership change, look hard.
- **Ask what a wrong value would do** and whether any existing diagnostic would show it. If nothing would, build the
  diagnostic.
- **Compare what should agree**: producer vs consumer, train vs val vs test vs live, first run vs resumed, code vs
  contract, two implementations of one idea, a metric vs what a trivial predictor scores.
- **Look at silences**: defaults nobody chose, options nobody reads, quiet fallbacks, rows dropped without a count.
- **Be curious beyond the brief.** Anything surprising, over-complicated or contradicting your expectation deserves a
  look. Pull the thread; classify it as fixed, documented, or an open question. Do not drop something because it is
  "out of scope".
- **Think about what comes next**: mixed precision, `torch.compile`, new heads, new data. Note where such changes would
  silently break an assumption (e.g. places that must stay float32).
- **Be honest about confidence.** Say what you proved by running, what you only argued, what you did not look at. No
  GPU is available here: say what a GPU run would have to confirm instead of guessing it.

## 4. Working loop (light, no ceremony)

For every problem you notice:

1. **Verify quickly** with the smallest runnable check (a few lines on synthetic or fixture data are enough; keep it as a
   test or a diagnostic if it will be useful again).
2. **Fix the code**, or, if the fix is large, uncertain or belongs to a later step, **write it up** in `NOTES.md`.
3. **Record one line** in `docs/audit/<round>/NOTES.md`: what, where (`file:line`), evidence (command + result), action
   (fixed in commit X / documented / open question), and which recorded results it puts in doubt.
4. Move on.

Small honest guard-rails that serve the goal (not project rules): work on a branch, never directly on `main`; commit in
small coherent commits with clear messages; run the tests of what you touched and fix or update tests and pins that
only pinned the old behaviour; do not overwrite a recorded measured result (add a corrected note next to it instead);
list every intentional behaviour change in `NOTES.md` with the reason, so old and new numbers can be compared.
Subagents are allowed for self-contained heavy sub-tasks; keep their prompts short and specific.

## 5. The instruments to build (the cards)

Each instrument = code that runs from one command + a card in `docs/audit/<round>/cards/` with: **question**, **how to
run** (command, inputs, CPU/GPU, expected duration), **how to read it** (a small table: observation -> conclusion ->
what to change), and **baseline**: what a trivial or random model scores, so a number is never read alone. Build on what
the project already has (`model/probe.py`, `workflow/health_report.py`, `trainer/grad_report.py`, the synthetic
environments E1-E3, the oracle tests, the battery) rather than starting over; extend or replace them where they fall
short. Make one entry point that runs all cards and prints a readiness table (CPU quick mode; heavier checks flagged for
a GPU smoke job).

**Card A — Where is information lost or gained in the model?** Walk the model stage by stage (input, embeddings /
projections, each block, the pooled representation, each head) and measure, at every stage, how much of what predicts
the target is still there. For example: a probe from the stage's activations to the target (rank IC / R² against a
strong baseline on the same inputs), variance retained and effective rank, saturation or dead units, input-ablation or
shuffle effect per stage, how much each stage changes the information relative to the stage before. The output is a
table that shows the stage where information drops (or where it is finally created), compared against the best simple
baseline from the same window. Verify the instrument itself on a synthetic case where you built the answer in (a stage
that deliberately destroys the signal must be pointed out).

**Card B — Does the trainer guide the model to learn?** Answer from measurements, not from the loss curve alone: can the
model overfit a single small batch (if not, something is broken); is each parameter group receiving a gradient of the
right size, and does the update-to-weight ratio make sense (is weight decay, clipping, EMA actually doing anything);
are the loss terms balanced or does one dominate; does a learning-rate range test show a sane region; does the loss
fall on synthetic data with a known truth (E1) and reach the oracle; does it collapse to the trivial score when the
labels are shuffled; does the epoch chosen by the selection metric actually have the better target metric; does
resume equal an uninterrupted run; do train and eval modes agree. Output: "learns / does not learn / learns the wrong
thing" with the failing check named.

**Card C — Is the data sound?** Leakage by perturbation (change data after time t, outputs at t must not move); label
and unit consistency across every consumer (codec round trips); split and purge correctness, and coin/time composition
of train/val/test and of batches; distribution shift between splits; how much information the inputs hold about the
target (an upper-ish bound from a strong simple model on the same inputs, such as the volatility baseline) so we know
what a good model could reach; near-duplicate windows and memorisation risk; coverage, flat bars, clipping and
missing-data counts.

**Card D — Do the numbers mean what we say?** Every reported metric next to its trivial baseline; the decode path from
output to price; profit after cost with its uncertainty; the selection metric vs the trading goal.

**Card E — The readiness report.** One command that runs A-D and prints: stage, check, result, baseline, verdict, and
for each failure the card section that says what to change. When the model is not ready for trading, this report must
tell us which stage to work on.

If you find while building these that a card's question cannot be answered with the current code, that is a finding:
add the missing hook (a tap, a return value, a switch) and say so.

## 6. Records (keep them short)

Under `docs/audit/<round>/` (`<round>` = `<date>-model-first`):

- `NOTES.md` — the running list from the working loop (one line per problem; fixed / documented / open).
- `cards/` — the cards of section 5.
- `NEXT.md` — what remains: open questions with what would settle them, things that need a GPU smoke job, findings you
  documented but did not fix (ordered by expected effect on the model), and the exact commands to rerun the readiness
  report after each future fix.

The project's registers (`bottlenecks.csv`, `suggestions.csv`, `friction_log.csv`, `STATUS.md`) are optional: update
them only if it takes a minute and helps the next reader; otherwise point to `NOTES.md` from one STATUS line.

Commit as you go (a cut session must lose nothing) and keep one draft PR open for the branch.

## 7. Report to the owner (Arabic, short)

What was found and fixed (with evidence), what was documented but not fixed, the instruments built and how to run
them, what the readiness report says about the current model (if you were able to run it), what was NOT verified
(everything that needs the GPU or the full real artifact), behaviour changes you made on purpose, and what to do next.
