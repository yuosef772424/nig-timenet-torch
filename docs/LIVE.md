# Live prediction and the latest-samples report (owner, 2026-11-23)

What it does: fetch the latest CLOSED candles from Binance, build the model windows with the SAME code as a dataset
build, load a model from a folder, predict the last N bars of every coin (the older ones have a finished horizon and are
judged, the newest ones are live), and print / chart them in original prices. Nothing places an order. Exploratory
(rule C5).

## The two commands

```
# 1. once per trained model: write live_model.json beside best.pt (a run trained after LV1 can call export_live_model(run))
python -m nig_timenet_torch.workflow.live_model RUN_FOLDER --artifact ARTIFACT_DIR [--overrides '{"d_model": 64}']

# 2. every time: fetch -> prepare -> predict -> table + selection + charts
python -m nig_timenet_torch.workflow.live_run --model RUN_FOLDER [--coins BTCUSDT,ETHUSDT] [--last 6] [--out DIR]
```

Without ``--coins`` the training universe is read and shown. ``--last 6`` predicts the 6 newest bars per coin. A row whose
horizon is not over has no truth (status ``live``): its true columns print ``-``.

## What a row shows

| column | meaning |
|---|---|
| `signal` | LONG / SHORT: the close class head (p > 0.5), the battery's one rule |
| `confidence_pct` | probability the class head gives the signalled side (50-100) |
| `uncertainty_pct` | predictive sd of the close return (NIG), percent of the entry |
| `move_pct` / `true_move_pct` / `error_pct` | expected move, true move (live: `-`), expected minus true |
| `edge_ratio`, `agrees` | `abs(move) / uncertainty`; whether the sign of mu and the class head point the same way |
| `entry`, `target`, `target_lo/hi`, `true_target` | entry = close of the last input bar; target = entry x (1 + move); band = +-1 sd |
| `target_high/low`, `true_high/low` | the high / low heads as price levels (never a direction), when the model has them |
| `ok` | signal side x true move > 0 (live: `-`) |

`eval/trade_selection.py` keeps the rows with `edge_ratio > k` (default 1) and ranks them by confidence, move, edge or a
combined score; `after_cost_pct` is the move minus the round trip. The chart (`eval/latest_plots.py`) draws the candles,
each row's entry -> expected target with the +-sd band, the true target, and below it expected vs true move with the error.
The same report over a finished run: `python -m nig_timenet_torch.workflow.latest_report RUN_FOLDER --stage val`
(`test` is the judged stage; tuning the selection while reading it is selection bias).

## What is guaranteed and what is not

Tested (CPU, fake API, `tests/data/test_build_live.py`, `tests/data/test_live_fetch.py`, `tests/workflow/test_live_run.py`):
- the live windows equal the dataset windows at the same bars, bit for bit, on the synthetic market (1 and 2 frames);
- a live row at a bar of the run's grid has the same `mu`, `sigma`, `p_dir` as the run's own battery table;
- the open bar is never used; a coin with a gap in its newest window, a stale newest bar (versus the clock) or no
  candles is listed in `skipped` and gets NO row (never a prediction on an older bar);
- the fetch pages past a 1000-row cap (found on the real spot mirror: a short page is not the end).

NOT verified:
- a real trained model on real data: no TF artifact or Drive run was available. The real-data checks made: the pipeline
  built live windows from 6 real coins (spot mirror) with all features finite; the fetch read 3100 contiguous real bars.
- the futures API itself: `fapi.binance.com` answers HTTP 451 from some countries (it did from the build machine). Run
  from a machine where it works, or pass `--klines-url https://data-api.binance.vision/api/v3/klines` (SPOT prices, close
  to but not equal to the futures prices the model trained on; judge a model on it only knowing that).
- models trained on a TF-exported artifact: the json then has no `data_settings` (the TF export does not carry them), so
  pass `--data-settings FILE` (the DataSettings the artifact was built with) and check parity first: compare the live
  windows of old bars with the artifact rows of the same bars.
- a coin-channel (snapshot panel) model: refused (its coin ids must follow the training coin order).

## Cautions

- History: `--bars` (default 1000) must cover the window plus the indicators' warm-up. Long-memory features (EMA 200...)
  would differ from the training values; the 1h feature set of the tests gives the same window from 200 bars on (atol 1e-3).
- Cross-asset columns (breadth, momentum rank, orthogonal momentum) were computed over the TRAINING universe: the run
  reads that universe, not only the coins shown. A coin that is not in it is predicted with those columns from other coins.
- Sealed coins are dropped as in a dataset build (rule D1). Bars after the time seal (2026-10-01) are live data: use them
  to look, never to train or to choose settings; a forward test is a separate decision (rule C5).
- A live row is a prediction about the next `horizon` bars from a model whose edge is not established; the TF project and
  `docs/results/` say what it measured so far.
