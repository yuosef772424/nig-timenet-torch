# Hardening audit 2026-11b (structure and clarity), 2026-11-21

Owner message 2026-11-21: launch a hardening campaign; the goal is better structure and more clarity. Five read-only
audits (one per area, a cheaper model, rule A4), 177 findings with file:line evidence, proposed fix, effort and
whether behaviour changes. The lead checked the findings that would change behaviour or claim a bug before the plan
used them; the plan is `docs/hardening/PLAN-2026-11b.md`.

| file | area | findings |
|---|---|---|
| `A_model_trainer_stats_core.md` | `model/`, `trainer/`, `stats/`, `core/` | A-01..A-35 |
| `B_eval_doctor.md` | `eval/`, `doctor/` | B-01..B-35 |
| `C_workflow_jobs_colab.md` | `workflow/`, `jobs/`, `colab/`, `colab_ops/` | C-01..C-35 |
| `D_data_data_io.md` | `data/`, `data_io/` | D-01..D-35 |
| `E_docs_tools_tests_backtest.md` | session docs, `tools/`, tests infrastructure, `src/nig_backtest/` | E-01..E-37 |

## Lead verification (read against the code, 2026-11-21)

| id | verdict | what the code shows |
|---|---|---|
| A-01 | **rejected as a bug**, kept as duplication | `nig_scale_fit="train_std"` (`model/label_scale.py`, buffers named `label_scale`) and `nig_scale="train_std"` (`model/scalers.py` FittedScale, buffer `value`) are two exclusive components with different buffers; `workflow/run.py` build calls both fit functions, each fits only its own module, and the run default uses only `nig_scale_fit` (D-023 / D-026, B-071 closed). The finding stands as A-18: two components for one concept (package HB4, owner decision on which stays). |
| C-01 | **confirmed, fixed in HB1** | `flat_run.predict_stage` cut whole timestamps into chunks of `evaluation.chunk` (20,000 rows), then `predict_table` re-sliced each chunk into fixed 512-row batches: with ~100 coins a timestamp is cut at about one batch border in five, and a cross-section model (relative inputs, cross-coin layers, cross_center) computed its timestamp statistics on part of the coins for those rows. `Trainer.evaluate` (selection metric) was already group-aligned. |
| C-02 | **confirmed, fixed in HB1** | `health_step` strided ROWS of a cross-section run's VAL split. |
| C-08 | **confirmed, fixed in HB1** | `train.cross_section_loss` without whole-timestamp batches failed only at the first training step. |
| A-30 | **confirmed in part, fixed in HB1** | most combinations were already refused by `RunSettings`; the one left: a grouping `batch_by` with a timestamp `batch_order` (`Trainer._epoch` takes `batch_groups` first and ignores the order). |
| C-13 | **confirmed, fixed in HB1** | "batches per epoch" and `step_seconds_per_batch` used ceil(rows / batch_size) for grouped batches too. |
| C-03 | confirmed (= suggestion row 190, fifth review) | `trainer/identity.py::_payload` hashes fit / model / trainer / optimizer settings, none of `split.*`; a resumed run folder accepts another coin set or split. HB2. |

Every other finding is taken as written until the package that handles it reads the code again (a package session
re-verifies its rows before changing anything).
