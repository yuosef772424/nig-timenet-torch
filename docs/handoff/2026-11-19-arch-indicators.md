# Handoff 2026-11-19: architecture / size, indicator channels, rank loss, lr 1e-5 (owner task)

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

Smoke `rolling-ind-smoke-001` ran OK on commit eb6f1eb+ (rule E5). Stage 2 (`rolling-ind-001`, arms R / I / IR x seeds
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

## What is next
1. (done) stage 1 choice.
2. Launch stage 2 (6 processes), read it, write `docs/results/arch-indicators-001.md` (failure reading if IR fails).
