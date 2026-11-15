# Card F — Staged training: grow the data and lower the learning rate at each sign of memorisation

Status: SPEC for a future session (owner request 2026-11-15). Nothing here is built yet. The session that takes it
reads `CLAUDE.md`, `docs/STATUS.md`, `docs/RULES.md`, then this card, then
`docs/audit/2026-11-14-model-first/NOTES.md` rows N-10, N-13, N-14, N-20 and `TRANSFORMER_HEAD_RESULTS.md`.

## 1. The idea (owner) and the question

Train on a PART of the training data until the model shows a sign of memorisation; then add new data, lower the
learning rate, and continue until the sign shows again; repeat until all the data is in and the training ends.

Question the experiment must answer with numbers: does this schedule end with a better model (validation loss, rank
IC, calibration, and the T-profit rows) than ordinary training on all the data for the same number of optimizer steps?
And as a by-product: how many steps does the model need to start memorising N rows (the curve "steps to the sign"
against data size)?

Why it could help here: full-data runs overfit after epoch 1 (baseline r2), 60,000 rows hold only ~7,200 independent
labels (N-10), and the configured lr is far below the lr range test's best (N-14). Why it could do nothing: the
outputs barely use the encoder (N-20), so there is little capacity that memorises; the baselines of section 5 decide.

## 2. Definitions (fix these before any run; they are the experiment)

| term | definition | why this and not the obvious one |
|---|---|---|
| sign of memorisation | at an evaluation point: the 3-point mean of `val_raw_loss` on the FIXED validation set has not improved by `min_delta` (1e-3) for `k` (3) evaluations, WHILE the eval-mode loss on a fixed train probe set (2,000 rows of the current stage) fell by more than `min_delta` over the same evaluations | the train loss of the log is train-mode (dropout, noise) and sits ABOVE the val loss in every run so far (-6.3 vs -6.8), so "train below val" never fires; a val plateau with a flat train probe is "finished learning", not memorisation, and must not trigger a stage |
| evaluation point | every `eval_every` optimizer steps (default: 1/4 of the stage's epoch, at least 100 steps), not every epoch | the full data memorises inside epoch 2; an epoch-level check sees it too late |
| stage | one data subset + one learning rate; ends at the sign | |
| fixed validation set | the whole validation split (strided to at most 60,000 rows), the SAME rows in every stage and in every baseline | a per-stage validation set makes stages incomparable |
| train probe set | 2,000 rows drawn once per stage from that stage's training rows, evaluated in eval mode | measures fit without dropout |

## 3. The loop

```
fit the scalers once on the FULL training split (train rows only; units must not move between stages)
rows_0 = first subset; lr_0 = base lr
for stage s = 0, 1, ...:
    train on rows_s at lr_s, evaluating every eval_every steps, keeping the best weights (by the fixed val set)
    at the sign of memorisation (or at the stage's step cap):
        restore the best weights of this stage            # improvement 1
        if rows_s is the whole training split: stop
        rows_{s+1} = rows_s + new rows                    # union, never only the new rows (improvement 2)
        lr_{s+1} = max(lr_s * gamma, lr_floor)            # gamma 0.5, lr_floor 1e-6
        reset Adam's moment estimates; warm up lr over 100 steps   # improvement 3
final: the best weights over ALL stages (the tracker spans the whole run, raw value: see N-13)
```

Growth schedule (the main design choice; run arm G1 first, the others only if G1 shows an effect):

| arm | how rows are added | note |
|---|---|---|
| G1 stride | start with every 32nd row of the training split (all coins, all dates), halve the stride each stage: 1/32, 1/16, ..., 1/1 | rows overlap 0.97 with their neighbour (N-10): a halved stride adds rows that are new windows but NOT new independent labels. This is the owner's idea in its simplest form and the cheapest to build |
| G2 coins | start with 1/8 of the coins, add coins each stage | adds genuinely new series; watch the validation loss per coin group |
| G3 time | start with the OLDEST 1/8 of the dates, extend forward | the closest to live use; the newest data gets the smallest lr, which may be wrong (regime change, N-11): report it |

## 4. Improvements added to the owner's idea (each is a switch, default as listed; report its effect if you turn it off)

1. Restore the best weights of the stage before adding data: do not carry memorised weights forward.
2. Union, not replacement: each stage trains on everything seen so far; optional `new_weight` (default 1.0) oversamples
   the new rows.
3. Optimizer reset + short warm-up at each stage boundary (Adam's second moments are tuned to the old data).
4. Memorisation is measured in eval mode on a train probe set (section 2), with the gap `val - train_probe` logged at
   every evaluation: its growth is the quantity of interest, the trigger is only a threshold on it.
5. A step cap per stage (default 20 x the stage's epoch) so a stage that never shows the sign cannot run forever; a
   stage ended by the cap is recorded as "no sign" (that is a result: at this size the model does not memorise).
6. Equal-budget baselines (section 5): without them a gain cannot be told apart from "more steps" or "lr decay".
7. Record per stage: rows, independent labels (card C `overlap` estimate), lr, steps to the sign, best val, gap at the
   sign, rank IC on the fixed val set. The table "steps to the sign vs rows" answers the memorisation question (B-038)
   whatever the final score is.
8. BatchNorm: keep `bn_warmup` on; after rows are added run 50 forward-only steps before the first update so the
   moving statistics see the new rows (the level path's BatchNorm, N-04).
9. Small model first (owner): `encoder: transformer`, `ffn: none`, d_model 64 (79,918 weights), and
   `steps_per_execution` 32 on the GPU. The validation set is evaluated at evaluation points only, strided to 60,000
   rows: a 200,000-row validation pass per epoch tripled the cost of the 2026-11-15 Colab run.

## 5. Baselines and the accept rule (rules C1, C4)

Same model, same seeds (>= 2 per arm), same fixed validation set, the SAME total number of optimizer steps as the
staged run of that seed:

| baseline | what it isolates |
|---|---|
| B1 all rows from step 0, constant lr_0, early stop off | the plain run |
| B2 all rows from step 0, lr lowered by gamma at the SAME steps as the staged run | the lr decay alone (without it a gain of the staged run is credited to the data schedule by mistake) |
| B3 stride-1/32 rows only, constant lr_0 | the small-data floor |

Accept "staged training helps" only if, on every seed, the staged run's best `val_raw_loss` beats B1 AND B2 by more
than the seed spread of B1, and the TEST rank IC (high, low) is not lower. Then run the battery (T-profit rows) on the
winner and on B1. Anything else is reported as "no gain" with the per-stage table.

## 6. Build plan (one PR; rule G5: no second trainer)

- `trainer/staged.py` (new, card in the docstring): `StageSettings` (growth arm name, start share, gamma, lr_floor,
  eval_every, k, min_delta, step cap, warm-up steps, new_weight) and `staged_fit(trainer, data, settings, ...)` that
  drives the EXISTING `Trainer` (its train / eval step and checkpoint functions). No `.backward()` / `.step()`
  outside `trainer/` (G5 ratchet). If `Trainer.fit` cannot be reused per stage because it is epoch-based, add a
  step-level loop inside `trainer/` that shares `train_step` / `eval_step`, not a copy of them.
- Growth arms as a small registry `GROWTH` (`stride`, `coins`, `time`): `rows(stage) -> index array`; a new arm is
  one function + one registration line (G8).
- `workflow/run.py`: a `train_staged` step next to `train` (same Run object, same summary file plus `stages`: the
  per-stage table of improvement 7); `TrainSettings.staged: StageSettings | None = None` (None = today's training,
  byte for byte: pin test).
- `docs/audit/2026-11-14-model-first/quick_arms.py`: `--staged JSON` so the arms and baselines run from one command.
- Tests (CPU, synthetic E1 fixture): staged with one stage and gamma 1 equals plain training bit for bit; the row
  sets are nested and end at the full split; the sign fires on a planted case (a model that memorises 64 rows with
  shuffled labels) and does NOT fire when the train probe is flat; resume in the middle of a stage equals an
  uninterrupted run; the fixed validation rows are identical across stages.
- Docs: this card gets a "How to run / how to read" section with the real table; NOTES rows; a decision file only if
  a default changes (none planned).

## 7. Runs (after the tests are green)

1. CPU look, real slice (`data_io/vision_slice.py`, 60 coins), G1, 2 seeds, plus B1-B3: about 1 h on 4 cores.
2. Colab (Drive artifact, T4 or L4), G1 with the small model, 2 seeds, plus B1 and B2:
   `python -m nig_timenet_torch.colab_ops setup BRANCH`, then `colab_ops start "... quick_arms.py /content/art1 OUT
   --arms transformer:1e-4 --model '{\"ffn\": \"none\"}' --staged '{...}' --spe 32 --device cuda" NAME`.
3. Only if G1 beats B1 and B2: G2 and G3, then the doctor on the winner
   (`python -m nig_timenet_torch.doctor ARTIFACT --run-dir RUN --trained --include-test`).

## 8. What the report must contain

The per-stage table (improvement 7) for every seed; the final comparison against B1 / B2 / B3 with seed spread; the
curve steps-to-sign against rows; which improvements were on; what was NOT run; and the verdict by the accept rule of
section 5 in one sentence.

## 9. Known traps

- A sign that never fires is a finding, not a bug (cap, improvement 5). Check the planted-case test before believing
  either outcome.
- With G1 the added rows are near-duplicates in time: expect the sign to arrive after about the same number of
  PASSES per stage, i.e. more steps per stage. If steps-to-sign scales with rows, memorisation is of windows, not of
  independent labels.
- Lowering the lr at every stage can stall learning before the data is all in (lr_0 3e-4, 6 stages, gamma 0.5 ->
  4.7e-6). Start from the lr range test's region (N-14) and keep `lr_floor`.
- The selection must use raw validation values (N-13: the smoothed tracker keeps the wrong epoch's weights).
- Do not compare a staged run with a baseline that had fewer steps, another validation set, or another seed.
