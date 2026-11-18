# Handoff: snapshot panel model (the owner's "100 x 3" design) on Colab, 2026-11-18

## What was asked (owner, Arabic, 2026-11-18)
Build a model in which the whole moment (about 100 coins x 3 channels x 32 bars) enters as one input and the model
outputs NIG for every coin, designed for accuracy (not simplified for speed), test it on a small sample for errors, then
run it on Colab on the full data with rolling training. Then: "run the experiment on Colab directly; no need to test
the performance locally".

## What exists (branch `claude/alpha-beta-models-training-bxm3on`, PR #66)
- `src/nig_timenet_torch/model/snapshot.py`: architecture `snapshot_panel` (joint moment path: per bar the vector of
  all coin slots [channels, presence] -> Linear -> LayerNorm -> GRU 256 x 2 -> moment state, per-coin output weights
  `slot_out` (zero at start); optional shared per-coin path (relative inputs -> GRU 64 x 2); coin embedding; readout
  -> NIG / confidence / class heads). Coin slots fitted on the train rows (`CoinSlots`, 128 slots).
- `SplitSettings.coin_channel` (asset index as an input channel; needs `cross_section`), `RollingConfig.architecture`.
- Pre-registration: `docs/audit/2026-11-18-snapshot-panel/PREREGISTERED.md` (arms, metrics, accept rule).
- Tests: `tests/model/test_snapshot.py` (keys, row order, no leak between moments, joint dependence, unknown coins,
  run steps on a tiny E4). CPU rolling smoke on E4 passed; GPU smoke `snapshot-rolling-smoke-001` passed (91 s) after
  one fix (CoinSlots device, commit a2096fe).

## What runs where (Colab L4, runtime id 422b5e1019887999, started 2026-11-18 ~03:39 UTC)
| job | bridge process | config | outputs (Drive) |
|---|---|---|---|
| `snapshot-rolling-base-001` (baseline arm) | p-033930-ab04b8 | `docs/jobs/snapshot-rolling-base-001.json` | `MyDrive/nig_timenet_runs/snapshot-rolling-base-001/{out,train}` |
| `snapshot-rolling-joint-001` | p-033931-b42dcd | `docs/jobs/snapshot-rolling-joint-001.json` | `.../snapshot-rolling-joint-001/{out,train}` |
| `snapshot-rolling-hybrid-001` | p-033931-ab2807 | `docs/jobs/snapshot-rolling-hybrid-001.json` | `.../snapshot-rolling-hybrid-001/{out,train}` |
| E4 known-answer look (6 runs: base / joint / hybrid x seeds 0, 1) | p-034010-f94210 | `docs/audit/2026-11-18-snapshot-panel/scripts/snapshot_look.py` from the second checkout `/content/nig2` | `MyDrive/nig_timenet_runs/snapshot-e4-look-001/snap_look.jsonl` |

The three rolling jobs import `/content/nig-timenet-torch` at commit a2096fe: do not update that checkout while they
run (use `/content/nig2`, env file `/content/nig_env2.sh`). Poll: `python -m nig_timenet_torch.colab_ops out <pid>`
(env CB_URL / COLAB_BRIDGE_TOKEN from the owner's bridge cell; never write them to a file). The rolling entry does not
read `--max-minutes` and resumes from `run_dir`: after a cut, start the SAME command again.

## What is next
1. When the jobs end: read `out/rolling_summary.md` and `out/pooled_days_seed_<s>/panel_report.md` of each arm; apply the
   accept rule of PREREGISTERED.md (close relative rank IC pooled over the test days, both seeds; high / low not lower
   by > 0.01; net day P&L); read E4 against its oracle (close rank IC by timestamp 0.13, high / low 0.28).
2. Write `docs/results/snapshot-panel-001.md` with the seven headings of rule C7 for every arm that does not pass.

## Outcome (2026-11-18 ~04:50 UTC)
All three walk-forwards and the E4 look finished; the Colab tunnel then went down (HTTP 530). Read in
`docs/results/snapshot-panel-001.md`: base 0.087 close relative rank IC (7 folds) against joint 0.013 and hybrid 0.058;
neither snapshot arm passes the accept rule. Still to do: the error map on the fold tables (H4) and the arms H1-H3.
