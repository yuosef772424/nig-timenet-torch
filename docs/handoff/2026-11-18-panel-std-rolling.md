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

## State at 21:30 UTC (after the session's worker restart)
- `panel-rolling-std-001` and `panel-rolling-ref-002`: finished 19:56 (7 folds x 2 seeds each; pooled reports on Drive).
- `panel-rolling-ls-001`: cut at about 11 of 14 folds when the runtime / tunnel died (between 19:56 and 21:29); its
  checkpoints only resume on commit `6c928bb` (run identity): the chain below does that from a second checkout.
- The reading script (`read_std_rolling.py`) has NOT run on the finished jobs (no runtime): the chain runs it first.

## The two-timeframe chain (owner 2026-11-18): the run command
A NEW Colab GPU runtime, one cell (the token is read from Drive, never typed):

```python
from google.colab import drive; drive.mount('/content/drive')
!T=$(tr -d "\r\n " < /content/drive/MyDrive/crypto/secrets/github_token.txt); B=$(printf "x-access-token:%s" "$T" | base64 -w0); rm -rf /content/nig-timenet-torch; GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.https://github.com/.extraheader GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $B" git clone -q --branch ccr-acb890db-b2ekab https://github.com/yuosef772424/nig-timenet-torch.git /content/nig-timenet-torch && echo cloned
!mkdir -p /content/drive/MyDrive/nig_timenet_runs && SHUTDOWN=1 nohup bash /content/nig-timenet-torch/docs/audit/2026-11-18-panel-std/two_frames_chain.sh > /content/drive/MyDrive/nig_timenet_runs/two_frames_chain.log 2>&1 &
```

Follow: `!tail -5 /content/drive/MyDrive/nig_timenet_runs/two_frames_chain.log`. Without `SHUTDOWN=1` the runtime stays.
What it does: `docs/audit/2026-11-18-panel-std/two_frames_chain.sh` (header). Expected time: artifact build
(535 coins of 1h candles, first time only) then about 2 h of jobs. Outputs: `MyDrive/nig_timenet_runs/<job>/out`,
readings in `<job>/reading_vs_<ref>/reading.json`, the artifact in
`MyDrive/crypto_model/torch_artifacts/preprocessing_output_1_plus_1h` (with `derivation.json`).
Not verified: the chain end to end on Colab, the artifact build on real data (only the BTCUSDT channel check and the
CPU tests), the runtime release (`runtime.unassign()` through the kernel).

## State at 22:10 UTC on 2026-11-18 (runtime `68b28de0b4262a26`, L4)
- The owner started the chain (no `SHUTDOWN`) about 21:3x: artifact `preprocessing_output_1_plus_1h` built (fingerprint
  `02f375cb3286`, on Drive), smoke passed, `panel-rolling-2tf-001` running (commit `03bafba`), `panel-rolling-ls-001`
  resuming from `/content/nig_ls` (`6c928bb`). A second chain started by the session waited behind it and was killed.
- `panel-rolling-ref-003` failed at start (its config did not name a frame; fixed in `ed2d3b4`) and was restarted at
  22:06 from `/content/nig_03b` (`03bafba`, the commit of the 2tf job, rule C4).
- `label-order-e4-look` (E4, arms relative_rank_loss / _logret_std / _logret_std_label_order, seeds 0 and 1) from
  `/content/nig_new` (`ed2d3b4`): the known-answer test of `order_in_label_units` (B-079).
- `finish_and_release.sh` waits for every job, writes the readings (2tf vs ref-003, ls vs ref-002, ref-003 vs ref-002)
  to `<job>/reading_vs_<ref>/`, flushes Drive and releases the runtime. Log: `nig_timenet_runs/finish_and_release.log`.

## State at 22:55 UTC on 2026-11-18
- Finished: `label-order-e4-look` (order on prices restores high / low IC on E4: 0.263 / 0.252 vs reference 0.263 /
  0.253, std 0.232 / 0.222), `panel-rolling-ls-001` (14 / 14 folds), `btc-hedged-book-001` (results note).
- Running (all on Drive, resumable): `panel-rolling-2tf-001` (`03bafba`, main checkout started it: resume ONLY from
  `/content/nig_03b`), `panel-rolling-ref-003` (`/content/nig_03b`), `panel-rolling-usdt-001` and
  `panel-rolling-std-order-001` (`/content/nig_u`, `df4580b`; artifact `preprocessing_output_1_plus_usdt`, fingerprint
  `8dbebef936cd`, on Drive).
- `finish_and_release2.sh` (= `docs/audit/2026-11-18-panel-std/finish_and_release.sh`) waits for all of them, writes the
  readings (2tf vs ref-003, ls / usdt / std-order vs ref-002, ref-003 vs ref-002) and the BTC-hedged book of the later
  jobs to Drive, flushes Drive and releases the runtime.

## The runtime stopped at about 22:58 UTC (not the release waiter: its log is empty)
Cut, resumable from Drive: `panel-rolling-2tf-001` (fold 4 of 7), `-ref-003` (fold 3), `-usdt-001` and
`-std-order-001` (fold 0). Not started: the flat-split pair `panel-std-order-full-001` / `-400k-001` (owner
2026-11-19: does more data help) and its smoke. `docs/audit/2026-11-18-panel-std/resume_all.sh` brings all of it back
on a new runtime (pinned checkouts `03bafba` / `df4580b` for the resumes, branch head for the new pair, then readings
and the release). One Colab cell (the token is read from Drive, never typed):

```python
from google.colab import drive; drive.mount('/content/drive')
!T=$(tr -d "\r\n " < /content/drive/MyDrive/crypto/secrets/github_token.txt); B=$(printf "x-access-token:%s" "$T" | base64 -w0); rm -rf /content/nig-timenet-torch; GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.https://github.com/.extraheader GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $B" git clone -q --branch ccr-acb890db-b2ekab https://github.com/yuosef772424/nig-timenet-torch.git /content/nig-timenet-torch && echo cloned
!nohup bash /content/nig-timenet-torch/docs/audit/2026-11-18-panel-std/resume_all.sh > /content/drive/MyDrive/nig_timenet_runs/resume_all.log 2>&1 &
```
Follow: `!tail -5 /content/drive/MyDrive/nig_timenet_runs/resume_all.log`. `RELEASE=0` before `nohup` keeps the runtime.

## State at 23:30 UTC on 2026-11-18 (new runtime `052940f2bbbcfc2d`, L4)
`resume_all.sh` started 23:22 (log `nig_timenet_runs/resume_all.log`): the four walk-forwards resumed on their commits
(`/content/nig_03b`, `/content/nig_df4`), the smoke `panel-std-order-smoke-001` passed, `panel-std-order-full-001` and
`-400k-001` running from the branch head (`c0f1219`); the release waiter runs at its end (readings, BTC book, flush
Drive, release).

## State at 02:10 UTC on 2026-11-19 (runtime `052940f2bbbcfc2d` stopped at about 01:03 UTC, NOT by the waiter)
- Finished and read: `panel-rolling-2tf-001`, `-ref-003` (results note two-frames-001), `panel-std-order-full-001` /
  `-400k-001` (results note more-data-flat-001).
- Cut, resumable: `panel-rolling-usdt-001` and `-std-order-001` (25 of their 28 fold x seed runs written; about 30 min
  of work left). The runtime is released (the bridge answers 502), so nothing bills. To finish: a new GPU runtime and
  the same `resume_all.sh` cell (it re-reads every finished job quickly and resumes the two on `df4580b`), then the
  waiter writes the readings and releases the runtime.
