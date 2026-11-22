# Handoff 2026-11-19: architecture / size, indicator channels, rank loss, lr 1e-5 (owner task)

Status: closed (AI1 #71; `docs/results/i1-001.md`, `docs/results/b1-btc-timing-001.md`)

## What was asked (owner, 2026-11-19)
Choose the best architecture and number of units; train with several losses (the rank loss and the others); add RSI
and the stochastic (and volume, owner follow-up "reload the data and add volume if useful"); learning rate 1e-5;
launch training. One model will be tested in live trading, LONG ONLY. Charts of a small sample on Drive. Owner
follow-up: every test separates buys from sells (e.g. the battery on long trades only).

Branch `claude/sweet-meitner-vqs1ig`. Pre-registration: `docs/audit/2026-11-19-arch-indicators/PREREGISTERED.md`.

## What was built
- `data_io/indicator_channels.py` (+ tests): RSI 14, stochastic 14-3-3 (%K, %D), relative quote volume over 30 bars,
  from the coin's own Binance futures 4h candles with 100 bars of warm-up, appended to every bar of the windows.
- Artifact on Colab `/content/art_ind`, copy on Drive `crypto_model/torch_artifacts/preprocessing_output_1_plus_ind`
  (fingerprint `468bfd1a5666...`; 2,520,913 rows; 0.64 % rows without values (listing warm-up), 66 close mismatches).
  Candle cache: Drive `crypto_model/candle_cache_4h_hlcqv/`.
- `docs/audit/2026-11-19-arch-indicators/read_arms.py`: long book (VAL 90th percentile of mu) and short book
  (VAL 10th percentile) read APART, hedged versions, always-long, per fold; PNG charts.

## What runs where (Colab L4, my checkout `/content/nig_ind`, env `/content/nig_ind_env.sh`)
Another session (branch `claude/loss-side-heads`) runs 5 rolling jobs on the same runtime (`rolling-lr1-cos-001`,
`rolling-ls1-*`) from `/content/nig_lr1` / `/content/ls1`: do not touch them.

| process | what | outputs (Drive `MyDrive/nig_timenet_runs/`) |
|---|---|---|
| `look-{g64,g128,g128l3,tcn64,tr64,ct64}` | stage 1, `docs/jobs/arch-ind-look-001.json` per arm (configs `/content/cfg_ind/look_<arm>.json`) | `arch-ind-look-001/out_<arm>`, ckpts `arch-ind-look-001/train_<arm>` |
| `wf-R-s{0,1}` | stage 2 arm R (gru 64, art1, lr 3e-4) | `rolling-ind-001/out_R_s<seed>` |
| `wf-I-s{0,1}`, `wf-IR-s{0,1}` | stage 2 arms I / IR (tcn 64, art_ind, lr 1e-5; IR + rank term + time blocks), started 20:02 UTC | `rolling-ind-001/out_<arm>_s<seed>` |
| `br-chain` (`/content/br_chain.sh`) | BTC-relative artifact `/content/art_btcrel` (`data_io/btc_relative.py`, 0.64 % rows without values), smoke `rolling-btcrel-smoke-001` (OK), then `rolling-btcrel-001` seeds 0 / 1 in parallel (logs `/content/br_main_s<seed>.log`), checkout `/content/nig_br` pinned at 34d574a, started 21:05 UTC | `rolling-btcrel-001/out_s<seed>` |

All Colab runs of this study (smoke, stage 1, stage 2) run on commit **458bcdb** (`code_sha` in the checkpoints'
run identity). `/content/nig_ind` was later pulled for the reading script: a cut stage-2 job must be resumed from a
worktree at 458bcdb (`git -C /content/nig_ind worktree add -f /content/nig_ind_458 458bcdb`, PYTHONPATH to its
`src`, NIG_REPO_DIR to it; `docs/COLAB_OPS.md` traps). Smoke `rolling-ind-smoke-001` ran OK (rule E5). Stage 2 (`rolling-ind-001`, arms R / I / IR x seeds
0, 1) starts after the stage-1 choice. A cut job resumes when the SAME command is started again (same commit).

## What is decided
- Reference reading (rolling-single-002 through read_arms.py): the long top 10 % book pays (+0.16-0.22 % per day,
  CI above 0) but loses after removing the market move of the same timestamp: the model times the market.
- Stage 1 read on VAL (`analysis-ind-001/look_val/`): mean VAL long net per trade over 2 folds g64 +0.028 %, tcn64
  +0.023 %, g128l3 +0.004 %, ct64 -0.041 %, g128 -0.056 %, tr64 -0.181 %; VAL close rank IC tcn64 0.065 (highest).
  Choice by the rule: tcn64 (d_model 64, 2 layers, 120k parameters). Every arm's best epoch was the last of fold 0
  (lr 1e-5 still learning at 10 epochs).
- instance_norm (owner question): RevIN per window and channel, on by default; levels return through the level
  passthrough and the stats path; `use_instance_norm=False` tested 2026-11-18 (`docs/results/no-instance-norm-001.md`).

- Arm R finished (`analysis-ind-001/stage2_R/`): long book mean day +0.173 % [+0.034, +0.310] (seed 0) and +0.215 %
  [+0.060, +0.362] (seed 1) at 98.3 %; without the best 5 % of days -0.072 / -0.022 %; hedged by the market -0.159 /
  -0.110 % (CI below 0); fold 3 (2024-06 rally) carries it (+0.57 / +0.76 % per trade), fold 6 loses. The same shape
  as rolling-single-002 (replicated on seeds 0 / 1 on this commit).
- Stage 2 timing: arm I best epoch = 16 of 16 in fold 0 (lr 1e-5 still learning); about 22 min per fold for I, about
  35 min for IR (time-block batches): I ~2.5 h, IR ~4 h from 20:02 UTC.

- Free tests on the saved tables: `docs/results/btc-hedge-long-001.md` (no edge beyond BTC, beta 1.35) and
  `docs/results/b1-btc-timing-001.md` (BTC at the bullish times beats BTC always, nets about 0: cost wall).
- BR (owner): BTC-relative inputs, R configuration otherwise; pre-registered in `PREREGISTERED_BR.md`.

## What is next
1. (done) stage 1 choice.
2. Launch stage 2 (6 processes), read it, write `docs/results/arch-indicators-001.md` (failure reading if IR fails).

## Stopped by the owner (2026-11-19 ~22:20 UTC: "close the Colab session")
Every process of this session was killed; nothing runs on the runtime. State on Drive (`MyDrive/nig_timenet_runs/`):
- `rolling-ind-001` arm I: finished, both seeds (`out_I_s{0,1}/rolling_summary.json`); NOT read yet.
- arm IR: stopped in its last fold (6 of 7 folds written per seed); resumes from its checkpoints with the SAME
  command on a worktree at 458bcdb (configs `/content/cfg_ind/wf_IR_s{0,1}.json` are lost with the runtime: rebuild
  them from `docs/jobs/rolling-ind-001.json`, model_overrides tcn 64 x 2, run_dir `train_IR_s<seed>`).
- BR1 (`rolling-btcrel-lbl-001`, commit 02dd45f): the label artifact was built and the smoke ran on the runtime's
  local disk; the two seeds were killed seconds after start. Re-run from the start: `art_btcrel` (btc_relative on
  art1) then `--label-only`, smoke, seeds 0 and 1 (`docs/jobs/rolling-btcrel-lbl-001.json`).
- Artifacts on the runtime's local disk (`/content/art_ind`, `/content/art_btcrel`, `/content/art_btcrel_lbl`) are
  lost with it; `art_ind` has a Drive copy (`crypto_model/torch_artifacts/preprocessing_output_1_plus_ind`); the two
  BTC artifacts rebuild in about 1 minute from the Drive candle cache (`crypto_model/candle_cache_4h_hlcqv`).
