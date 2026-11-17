# Backtest: a separate historical test for any model and any strategy (owner, 2026-11-17)

Code: `src/nig_backtest/` (a package of its own next to `nig_timenet_torch`), command `nig-backtest`.

## 0. Separate by construction

- **No project imports.** No module of `nig_backtest` imports `nig_timenet_torch`, except the bridges in
  `nig_backtest/bridges/`, which import it lazily inside their functions. `nig_timenet_torch` never imports
  `nig_backtest`. `tests/nig_backtest` checks both directions.
- **Portable.** The package needs numpy and pandas (matplotlib for charts). To use it in another project, copy the
  folder and write one bridge.
- **It never reads the model's features.** It reads the model's OUTPUTS with the time of each output, and candles
  (from the artifact on Drive through the bridge, from files, or from the public archive).

## 1. The input (the only way in)

`nig-backtest contract` prints the contract, which `nig_backtest/contract.py` and `kinds.py` define:

- **Required columns.** `coin` and `time` (when the output is known; entry happens after it) and `score`.
- **Optional features.** `confidence`, `sigma`, `p_up`, `pred_high` / `pred_low`, `y_true`, `stage` and
  `ref_price`. Each can be absent, or present and switched off (`SignalSet(use=...)`, CLI `--use`). A rule that needs
  a missing feature raises.
- **Free columns.** Any column named `x_*`.
- **What `score` means: the output kind** (`SignalSet(kind=...)`, CLI `--kind`):
  - **Absolute kinds** decode to a price without losing information, so they give a predicted move and allow exits
    in `move` units:
    - `return`;
    - `log_return`;
    - `price` (needs `ref_price`).
  - **Lossy kinds** are used only as what their target represents: a direction and a ranking. An exit in `move`
    units raises for them.
    - `score`: any directional number (the default);
    - `rank`: read inside each time; direction = above or below that time's median;
    - `probability`: p(up);
    - `class`: -1 / 0 / +1.
- **Horizon.** `SignalSet(horizon="8h")` is the model's horizon. An exit with `hold: "horizon"` uses it.

## 2. A strategy is data (`nig_backtest/rules.py`, unknown keys raise)

```json
{"name": "model_tp",
 "entry": {"top_share": 0.1, "score_over_sigma": 1.0, "stages": ["test"]},
 "exit":  {"hold": "horizon", "take_profit": 1.0, "take_profit_unit": "move",
           "stop_loss": 2.0, "stop_loss_unit": "atr", "atr_timeframe": "4h"},
 "costs": {"fee_per_side": 0.0008},
 "book":  {"risk_per_trade": 0.02, "max_open": 20}}
```

**When a trade exits:**
- after N candles (`max_bars`), or after a duration (`hold`: `"8h"`, or `"horizon"`);
- at a target (`take_profit`) or a stop (`stop_loss`). Their units are:
  - `atr`: a multiple of the ATR at entry;
  - `pct`: a fraction of the entry price;
  - `sigma`: a multiple of the predicted sd;
  - `pred`: a multiple of the predicted extreme;
  - `move`: a multiple of the decoded predicted move (absolute kinds only).

**Candles finer than the model's frame.** A 4h model can run on 1h candles:
- `hold` counts time, not candles;
- `atr_timeframe: "4h"` computes the ATR on the model's frame;
- the 1h path decides which of the target and the stop came first inside a 4h bar.

**Your own rule.** Register a function `fn(rows, params) -> bool array` with
`ENTRY_FILTERS.register("name", fn, version=1, needs=(...))` and list it in `entry.filters`. Every strategy and every
test can then use it.

## 3. Run, look, compare

```
nig-backtest run     --signals s.parquet --kind return --horizon 8h --prices archive:1h:candles --strategy s.json --out out/
nig-backtest compare --signals s.parquet --prices files:/data/raw --strategies a.json b.json --out cmp/
nig-backtest ladder  --signals s.parquet --prices archive:4h:candles --strategy base.json --steps steps.json --out lad/
nig-backtest from-battery RUN_FOLDER signals.parquet        # a finished run's outputs (bridge)
```

**`run` writes:**
- the trades (`trades.parquet`);
- the summary, with the rule trace and the tables by exit reason, side, stage, month and coin;
- the equity and drawdown chart;
- galleries of the worst and best trades on raw candles.

**`ladder` and `leave_one_out`** show what each decision does to the portfolio, and the number of variants tried.

**`TradeBook(run.trades).where(...)` filters by:**
- coin;
- win / loss;
- `direction_right`;
- `label_right`;
- exit reason, side, stage or time;
- a pandas `query`.

`.plot` and `.gallery` draw the trades.

## 4. The bridge to this project (`nig_backtest/bridges/nig_timenet_torch.py`)

```python
from nig_backtest.bridges import nig_timenet_torch as bridge
signals, meta, run = bridge.run_on_coins(settings, ["BTCUSDT"])      # restore a trained run, run the model on one coin
prices = bridge.artifact_prices(run.dataset, coins=["BTCUSDT"])      # the model's own candles from the artifact
res = bt.run_backtest(bridge.signal_set(signals, meta), prices, strategy)
```

- `signals_from_run(run, stage, coins)` gives the close head's decoded return (kind `return`), its confidence head,
  sigma, p_dir, the label, and the prediction time.
- `signals_from_folder(folder)` reads a finished run's battery tables, for example a run folder on Drive.
- Artifact candles are one per window: with a stride larger than 1 they are that many bars apart. Full or finer
  candles come from the archive.

## 5. Training data and large tables

- **Training data.** Pass the training rows (`stage = "train"`). With score = y_true (kind `return`) the label is the
  oracle of that data.
- **Memory.** Selection reads the whole signal table, so cross-section rules see every coin. The candles load one coin
  at a time, in batches, and are cached. A parquet signal table is read with filters.
