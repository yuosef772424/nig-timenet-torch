# Backtest: a separate historical test for any model and any strategy (owner, 2026-11-17)

Code: `src/nig_timenet_torch/backtest/`, run as `nig-backtest` (or `python -m nig_timenet_torch.backtest`).

The backtest imports nothing from the model, the trainer or the run steps (rule G7, `tests/backtest`). A model, a
hand-written rule, or the training labels themselves all reach it the same way: through ONE signal table.

## 1. The input (the only way in)

`nig-backtest contract` prints the contract, which `backtest/contract.py` defines:

- **Required columns.** `coin` and `time` identify the signal. `time` is the moment the signal is known, and entry
  happens after it. `score` is the signal: its sign is the direction and its size ranks the rows.
- **Optional features.** `confidence`, `sigma`, `p_up`, `pred_high` / `pred_low`, `y_true` and `stage`. A feature
  can be absent, or present and switched off (`SignalSet(use=(...))`, CLI `--use`). Either way it is dropped before
  any rule sees it. A rule that needs it raises instead of guessing.
- **Free columns.** Any column named `x_*` is carried into the trades and can be read by your own rules.
- **Candles.** RAW candles come from a `PriceSource`, one coin at a time and cached:
  - `FilePrices` reads a folder (the TF raw layout `{coin}.csv.gz`, or parquet);
  - `ArchivePrices` reads the public Binance futures archive, with a disk cache;
  - `FramePrices` uses frames already in memory.
- **From this project.** `nig-backtest from-battery RUN_FOLDER signals.parquet` turns a run's battery tables into a
  signal table: score = mu, sigma, p_up = p_dir, y_true = the label, stage = val / test.

## 2. A strategy is data (`backtest/rules.py`, unknown keys raise)

```json
{"name": "fade_tp",
 "entry": {"top_share": 0.1, "score_over_sigma": 1.0, "stages": ["test"]},
 "exit":  {"max_bars": 3, "take_profit": 1.5, "take_profit_unit": "atr", "stop_loss": 2.0, "stop_loss_unit": "atr"},
 "costs": {"fee_per_side": 0.0008},
 "book":  {"risk_per_trade": 0.02, "max_open": 20}}
```

- **Exit after N candles:** `exit.max_bars`.
- **Exit at a target or a stop:** `take_profit` and `stop_loss`, each with its own unit:
  - `atr`: a multiple of the ATR at entry;
  - `pct`: a fraction of the entry price;
  - `sigma`: a multiple of the row's predicted sd;
  - `pred`: a multiple of the predicted extreme.
- **Entry price:** `next_open` (the default) or `signal_close`.
- **A candle that touches both target and stop:** `both_hit` decides which one counts.
- **Your own entry rule:** write a function `fn(rows, params) -> bool array`, register it with
  `ENTRY_FILTERS.register("name", fn, version=1, needs=(...))`, and list it in `entry.filters`. Every strategy and
  every test can then use it. The built-in `min_column` is an example.

## 3. Run, look, compare

```
nig-backtest run     --signals s.parquet --prices archive:4h:candles --strategy fade_tp.json --out out/
nig-backtest compare --signals s.parquet --prices files:/data/raw --strategies a.json b.json --out cmp/
nig-backtest ladder  --signals s.parquet --prices archive:4h:candles --strategy base.json --steps steps.json --out lad/
```

**`run` writes:**
- `trades.parquet`: every trade with its entry and exit, the exit reason, gross and net, MFE / MAE and the signal's
  features;
- `summary.md` / `.json`: the rule trace (how many rows each decision kept), the tables by exit reason, side, stage,
  month and coin, and the per-trade tearsheet;
- `equity.png`: equity and drawdown;
- `worst/` and `best/`: a chart of each trade on raw candles, with entry, exit, target and stop.

**`ladder` adds one decision at a time.** `steps.json` is `[["tp 1.5 atr", {"exit.take_profit": 1.5}], ...]`.
`leave_one_out` (Python) removes one decision from the full strategy. Both print the effect on the portfolio and the
number of variants tried. A best variant picked from many is a hypothesis for unseen data, not a result (rule C5).

**In Python, `TradeBook(run.trades)` filters the trades:**
- by `coins`;
- by `outcome` (`"win"` or `"loss"`);
- by `direction_right` (gross > 0);
- by `label_right` (the model's label went the traded way);
- by `exit_reason`, `side`, `stage` or a time range;
- with a pandas `query`.

`.by("coin" | "month" | ...)` groups the trades, `.plot(i, prices, dir)` draws one trade and `.gallery(...)` draws
many.

## 4. Testing the training data and large tables

- **Training data.** Pass the training rows with `stage = "train"`. The labels themselves (score = y_true) are the
  oracle of that data: they show what an exit rule can at best keep of a perfect direction.
- **Memory.** Selection runs on the whole signal table, so cross-section rules such as `top_share` and
  `per_time_top_k` see every coin. The candles are loaded one coin at a time, in batches of `batch_coins`, and cached
  (`cache_coins`). A parquet signal table is read with filters (`SignalSet.read(coins)` / `.batches(size)`).
