# Card E — Readiness report (the one command)

Question: the model is not ready for trading; which stage is the problem (data, trainer, model internals, or the metrics), so that ONE thing is fixed and the checks run again?

## Run

```
python -m nig_timenet_torch.doctor e1 --out doctor_out                                  # everything, synthetic data
python -m nig_timenet_torch.doctor /path/to/artifact --out doctor_out [--max-rows 60000] [--epochs 3]
python -m nig_timenet_torch.doctor ARTIFACT --out doctor_out --cards A                  # letters A,B,C,D, comma separated
python -m nig_timenet_torch.doctor ARTIFACT --out doctor_out --run-dir RUN_DIR --trained  # a finished run's best weights, no short training
python -m nig_timenet_torch.doctor ARTIFACT --out doctor_out --model '{"dropout": 0.1}'   # another model configuration
python -m nig_timenet_torch.doctor ARTIFACT --out doctor_out --train '{"lr": 0.003}'      # another training configuration
python -m nig_timenet_torch.doctor e1 --out doctor_pre --model '{"bn_warmup": false, "order_detach_close": false, "nig_scale": null, "input_scaler": "none"}'   # the model before the audit
python -m nig_timenet_torch.data_io.vision_slice OUT_DIR --raw RAW_DIR                  # real-data artifact without Drive (public Binance 4h klines, sealed coins refused)
```

Other flags: `--full` (more rows and steps, minutes on a CPU, meant for a GPU smoke job or a longer CPU run), `--include-test` (card D scores test), `--seed`, `--batch-size`, `--architecture`, `--no-preset`.

- Quick mode (default) trains the configuration for 3 epochs on up to 60000 strided rows per split on CPU. The whole report on small E1 takes about 80-90 s on a 4-core CPU with no GPU; per-card seconds are in the `readiness.md` header (small E1: C 1.4, B 55-56, A 5.6-5.7, D 0.3-0.4, the rest is building and training the context).
- Cards run in the order C, B, A, D (data first; D reads C's ceiling).
- Outputs in `--out`: `readiness.md` (diagnosis, verdict per card, failing and warning checks, all checks, the stage table), `readiness.json` (every check with its detail numbers), `info_flow.csv` (card A's stage table).

## What it measures

The report table columns: card, stage, check, value, baseline, verdict (pass / info / warn / fail), action. Cards: A `model: information flow`, B `trainer`, C `data`, D `metrics and profit`.

Diagnosis (function `diagnosis` in doctor/readiness.py): the first line that applies is the stage to fix first. Read top down.

| Order | Line printed | Fires when | Card to open |
|---|---|---|---|
| 1 | `DATA is not sound` | a FAIL at stage `data/alignment`, `data/split` or `data/coverage` | C |
| 2 | `UNITS` | a FAIL at `data/units` (label sd in head units, InstanceNorm eps) | C |
| 3 | `TRAINER: DOES NOT LEARN` / `LEARNS THE WRONG THING` / `learns, with failing health checks` | a B FAIL; the names decide (see card B) | B |
| 4 | `MODEL loses information` | any A FAIL (trunk vs baseline, head vs trunk, self-check) | A |
| 5 | `INPUTS hold no readable signal for: ...` | C `data/information` WARN | C |
| 6 | `GOAL (profit after cost): MET / NOT met` | the D `GOAL` row | D |
| 7 | `METRICS below their trivial baseline` | D FAIL that is not a profit row | D |
| none | `No failing check.` | nothing failed | none |

Example, pre-audit default model on small E1 (`/home/user/work/doc_e1_default/readiness.md`): line 1 UNITS (4 scale checks fail), line 2 TRAINER DOES NOT LEARN (`overfit one batch` 0.267), line 3 MODEL loses information (trunk `high` 0.1698 vs 0.2758), line 4 GOAL NOT met. Verdict per card: A 5 pass / 8 warn / 4 fail, B 10 / 2 / 16, C 12 / 1 / 4, D 4 / 3 / 17.

Same data, fixed model (now the default; `/home/user/work/doc_e1_fix/readiness.md`): line 1 TRAINER LEARNS THE WRONG THING (`train vs eval mode, close` 0.72 / 0.74; `E1 oracle, low` 0.95 / 1.77; `E1 oracle, close` 0.43 / 0.81), line 2 GOAL NOT met. Verdict per card: A 12 / 3 / 0, B 21 / 4 / 3, C 16 / 1 / 0, D 16 / 3 / 5. The units, overfit, BatchNorm and information-flow failures are gone; what remains points at the trainer (a short run) and at the signal in E1 `close`.

## How to read it

| Observation | Conclusion | What to change |
|---|---|---|
| line 1 or 2 (DATA, UNITS) | every later number was measured on bad data | card C; fix, run again before reading B, A, D |
| line 3 TRAINER | the optimiser or train / eval behaviour is the problem | card B row named in the line |
| line 4 MODEL | the model throws away information the simple baseline reads | card A: `info_flow.csv`, the stage with the most negative `d_` |
| line 5 INPUTS | no model change fixes it | new inputs or another target |
| GOAL met, no other line | ready on this split | confirm on test (`--include-test`) and on a finished run (`--run-dir --trained`) before any trading step |
| GOAL not met, lines 1-5 absent | model is sound but there is no edge in this data after cost | research, not debugging |
| a WARN only | read it, but it does not by itself decide a line | the action text of the row |

Compare two runs: keep two `--out` folders (for example `doctor_pre`, `doctor_fix`). Compare the `## Verdict per card` tables (pass / warn / fail counts per card) first, then `readiness.json` (match rows by `card`, `stage`, `check`; compare `value` and `verdict`; `detail` holds the extra numbers) and the two `info_flow.csv` files. A change counts only if the predicted check moved by more than its noise (card A: 0.03 on probe scores) and no other card got worse. The same seed and the same `--max-rows` are required, otherwise rows differ.

## Rerun after each fix

1. Run the command; open `readiness.md`; read `## Diagnosis` from line 1.
2. Take the FIRST line only; open the card it names; read its "How to read it" table.
3. Change ONE thing (one model override `--model`, one training setting `--train`, or one file named in the action text). Never two stages at once.
4. Write down the check you expect to move and by how much, before running.
5. Run again into a NEW `--out` folder with the same seed and `--max-rows`.
6. Compare the two reports (above): the named check moved, the other cards did not get worse. If not, undo the change.
7. When lines 1-5 are gone, run `--full` (GPU smoke job or longer CPU run) and then `--run-dir RUN_DIR --trained --include-test` on the finished run.

## How we know the instrument works

- Each card has its own proof: A the planted destroyed stage `branch_out`, B the E1 oracle and E2 permuted labels, C labels rebuilt from later windows, D trivial baselines (see the cards).
- The same command on the pre-audit and the fixed model produced very different reports (above), and the fixes were the ones the action texts named.
- tests/doctor/test_readiness.py: the diagnosis order and the report layout; tests/doctor/test_common.py: the Check row, verdict helpers, probes.

## Limits

- Quick mode describes the METHOD on a short run (3 epochs, strided rows), not a finished long run; some B and D rows (E1 oracle slope, profit) can still fail simply because training was short.
- The diagnosis shows the first stage to look at, not a proof of cause; a FAIL names a place and an action to try.
- No GPU run and no full real artifact were used for the numbers above; `--full` and the real artifact need a GPU smoke job or a longer CPU run.
- Real labels cannot be checked against an oracle: E1 rows exist only on synthetic data.
