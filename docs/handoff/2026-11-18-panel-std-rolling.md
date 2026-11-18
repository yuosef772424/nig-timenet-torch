# Handoff: panel model with std normalisation, walk-forward (owner task 2026-11-18)

Asked (Arabic, owner 2026-11-18): test the panel model with std normalisation of the batch data, with the rolling
(walk-forward) training. Read as the panel arm `relative_rank_loss_logret_std` (D-033): train-standardised inputs
(already the default) plus per-head train-standardised labels (`target_norm: logret_std`), relative inputs and the
rank term. Pre-registration: `docs/audit/2026-11-18-panel-std/PREREGISTERED.md`.

Branch `ccr-acb890db-b2ekab`, commit `6c928bb` on Colab (`/content/nig-timenet-torch`, L4 runtime started 2026-11-18).

| job | entry | status (18:02 UTC) | outputs (Drive) |
|---|---|---|---|
| `panel-rolling-std-smoke-001` | rolling | done, exit 0, 62 s; checkpoint holds the fitted `target_norm` buffers | `MyDrive/nig_timenet_runs/panel-rolling-std-smoke-001/out` |
| `panel-rolling-std-001` (candidate) | rolling | running since 18:01 | `MyDrive/nig_timenet_runs/panel-rolling-std-001/{out,train}` |
| `panel-rolling-ref-002` (reference, same commit) | rolling | running since 18:01 | `MyDrive/nig_timenet_runs/panel-rolling-ref-002/{out,train}` |
| `panel-rolling-ls-smoke-001` | rolling | done, exit 0, 51 s | `MyDrive/nig_timenet_runs/panel-rolling-ls-smoke-001/out` |
| `panel-rolling-ls-001` (arm `relative_rank_loss_ls`, owner request 18:3x) | rolling | running since 18:36 | `MyDrive/nig_timenet_runs/panel-rolling-ls-001/{out,train}` |
| `panel-std-e4-look` (known answer, E4, 2 arms x 2 seeds) | panel | done 18:18 (close relative rank IC 0.125 for both arms vs oracle 0.15; high / low IC 0.23 with std vs 0.27 without) | `MyDrive/nig_timenet_runs/panel-std-e4-look/out` |

Poll: `colab_ops out <pid>` (pids change with the runtime; the outputs above are what counts). A cut job: start the
same command again (resumes from `train/`). Config of each: `docs/jobs/<job>.json` with `artifact` = `/content/art1`;
the E4 look: `/content/e4_long` from `write_panel_synthetic(..., PanelSyntheticSettings(n_bars=7200))`, arms
`relative_rank_loss` / `relative_rank_loss_logret_std`, seeds 0 and 1, 12 epochs, patience 4, battery off.

Next: read the pooled days and `rolling_folds.csv` of both jobs paired by fold and seed against the accept rules of
the pre-registration; write `docs/results/panel-rolling-std-001.md` (with the failure reading of rule C7 if the
candidate fails); close the step (`python -m tools.close_step`), draft PR.

Owner 2026-11-18: when training is finished, end the Colab session (after the reading is saved to Drive and pulled):
`runtime.unassign()` executed in the notebook kernel through `jupyter_client` (the bridge is a separate process, the
kernel is idle), which also ends the bridge.
