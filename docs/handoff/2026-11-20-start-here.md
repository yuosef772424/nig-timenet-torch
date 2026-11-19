# START HERE (2026-11-20): run method 1 (BR1) and method 3 (M3) on Colab, read them, then I / IR

For a NEW chat. Branch `claude/sweet-meitner-vqs1ig`, PR #71 (draft). Read `CLAUDE.md`, `docs/STATUS.md` (row AI1),
then this file; the day before is in `docs/handoff/2026-11-19-arch-indicators.md` and the result notes listed below.
The owner gives a fresh Colab bridge (URL + token in the chat; never write them to a file). All work is exploratory
(rule C5); every run below is pre-registered (files in `docs/audit/2026-11-19-arch-indicators/`).

## What the owner asked for tomorrow
1. **Method 1 = BR1**: BTC-relative inputs (coin minus BTC per bar, RSI / stochastic of coin / BTC, BTC features)
   AND the close label relative to BTC. Pre-registration `PREREGISTERED_BR1.md`, jobs `rolling-btcrel-lbl-001` /
   `rolling-btcrel-lbl-smoke-001`.
2. **Method 3 = M3 on the full artifact**: inputs and close label minus beta x the equal-weight market (beta per
   coin from its own past). Pre-registration `PREREGISTERED_M3_FULL.md`, jobs `rolling-mres-001` /
   `rolling-mres-smoke-001`. Local look on 60 coins: `docs/results/m3-local-001.md` (feasible, does not pay at 8 h,
   market put back does not help).

## Steps on Colab (about 1.5 h with both methods in parallel)
```bash
export CB_URL=...  COLAB_BRIDGE_TOKEN=...          # from the owner's Colab cell, shell only
pip install -e <a local clone of yuosef772424/colab-mcp-bridge>   # add_repo first if it is not in the session
PYTHONPATH=src python -m nig_timenet_torch.colab_ops rt
PYTHONPATH=src python -m nig_timenet_torch.colab_ops setup claude/sweet-meitner-vqs1ig   # env + /content/art1
```
Then pin one checkout at the commit to run (`git rev-parse HEAD` of the branch) and start the chain:
```bash
colab_ops exec 'git -C /content/nig-timenet-torch worktree add -f /content/nig_t <SHA>'
colab_ops start 'bash /content/nig_t/docs/audit/2026-11-19-arch-indicators/colab_chain.sh /content/nig_t both' chain
colab_ops wait <pid> --limit 400        # repeat; logs /content/rolling-*_s{0,1}.log
```
`colab_chain.sh` builds `/content/art_btcrel`, `/content/art_btcrel_lbl` (about 1 min) and `/content/art_mres` (reads
the Drive candle cache `crypto_model/candle_cache_4h_hlcqv`, all 535 coins cached), runs each smoke (rule E5), then
seeds 0 and 1 of each method in parallel (4 processes). Outputs: Drive `nig_timenet_runs/rolling-btcrel-lbl-001/
out_s{0,1}` and `rolling-mres-001/out_s{0,1}`. Started again on the SAME commit it skips what is done and resumes.

## Reading (run from a separate checkout, e.g. `/content/nig_read`, never the pinned one)
- BR1: `read_arms.py --level 0.975 --arm BR1=.../rolling-btcrel-lbl-001/out_s0 --arm BR1=..._s1 --arm
  BR=.../rolling-btcrel-001/out_s0 --arm BR=..._s1 --out .../analysis-ind-001/br1 --charts`; the rule is in
  `PREREGISTERED_BR1.md` (y_ret is relative to BTC there: primary = `long` - 0.16 %).
- M3: the three commands of `PREREGISTERED_M3_FULL.md` (reference A = `rolling-ind-001/out_R_s{0,1}`).
- Write `docs/results/br1-001.md` and `docs/results/m3-full-001.md` with the seven headings of rule C7.

## Also waiting on Drive (no new training needed)
- `rolling-ind-001` arm I (tcn 64, 7 indicator channels, lr 1e-5): finished, not read. Read with `read_arms.py`
  against R; also BTC-hedged.
- arm IR: stopped at 6 / 7 folds; resume only on a worktree at 458bcdb (details in the 2026-11-19 handoff).

## Known traps for tomorrow
- `market_residual` reads candles one coin at a time; on Colab they come from the Drive cache (fast). Cached
  candles end one bar after a coin's last row, so the last rows of each coin keep their own close label and zero
  channels (counted in `derivation_market_residual.json`: `rows_with_channels`).
- Cost per round trip is 0.16 % (0.08 % per side): every reading so far fails on it (signal about 0.04-0.10 % per
  trade over the market). The ideas that attack the cost are in the list below.

## Ideas recorded (not yet tested), from the 2026-11-19 readings
| id | idea | from |
|---|---|---|
| B1a | hold BTC while the bullish-timing signal stays on (one entry / exit per run, not a round trip per 8 h) | b1-btc-timing-001 |
| B1c | market regime filter (BTC above its 7-day mean, fitted on VAL) for the long book: the last year (folds 5-6) loses | b1-btc-timing-001, btc-hedge-long-001 |
| B2 | long book of low-beta coins only (beta < 1 on VAL) | btc-hedge-long-001 |
| BR2 | rank term on BTC-relative inputs | btc-relative-001 |
| M3c | residual model as a FILTER for the long book (skip its lowest 3 deciles) | m3-local-001 |
| M3e | target = mean simple residual return (or an upper-tail quantile head): the P&L sits in the right tail | m3-local-001 |
| C1 | maker orders (about 0.02 % per side): at 0.04 % per round trip M's +0.04 % per trade breaks even | m3 note, owner question |
| C2 | hold while a coin stays in the top decile (fewer round trips) | m3 note |
| LR | lr 1e-5 still improves at the last epoch: a warm-up + cosine schedule or more epochs before judging it | stage-1 look |
