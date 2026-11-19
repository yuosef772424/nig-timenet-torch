"""PURPOSE: A web page over the backtest (Gradio): browse every signal with filters and its chart on raw candles, set
    the entry / exit / cost / book rules with controls, run, read each period apart, browse and chart the trades.
TAGS: web app, gradio, signal browser, strategy controls, interactive backtest, trade browser, charts, colab
PITFALLS: gradio is optional (extra ``app``) and imported inside ``build_app`` only; every number comes from the same
    engine as the command line (``run_backtest`` / ``by_period``). TEST is hidden by default (summary, curve and trades
    of the test period): tuning rules while looking at TEST is the selection bias the protocol exists to stop; the
    switch shows it. A share link of gradio (``launch(share=True)``) is public to whoever holds the link.
"""

import json

import numpy as np
import pandas as pd

from nig_backtest.contract import SignalSet
from nig_backtest.engine import run_backtest
from nig_backtest.explorer import TradeBook, draw_candles
from nig_backtest.portfolio import by_period, equity
from nig_backtest.prices import PriceSource, bar_interval
from nig_backtest.rules import ADMISSION, COST_MODES, ENTRY_FILTERS, SIDE_RULES, UNITS, Strategy, strategy_from_dict

__all__ = ["filter_signals", "signal_figure", "controls_to_strategy", "run_view", "filter_trades", "equity_figure",
           "build_app", "TABLE_ROWS", "DEFAULT_CONTROLS"]

TABLE_ROWS = 500
#: the controls of the strategy tab and their starting values (a control at 0 / "" means "off")
DEFAULT_CONTROLS = {"side": "score", "quantile_q": 0.2, "top_share": 0.1, "min_strength": 0.0, "score_over_sigma": 0.0,
                    "min_confidence": 0.0, "hold": "horizon", "max_bars": 2, "take_profit": 0.0,
                    "take_profit_unit": "atr", "stop_loss": 0.0, "stop_loss_unit": "atr", "atr_timeframe": "",
                    "both_hit": "stop_first", "cost_mode": "fixed", "fee_per_side": 0.0008, "move_share": 0.0,
                    "per_coin": 1, "max_open": 0, "admit": "first", "filters": [], "filter_params": "{}",
                    "risk_per_trade": 0.02}


def filter_signals(rows: pd.DataFrame, coins=None, periods=None, side: str = "all", min_strength: float = 0.0,
                   start=None, end=None) -> pd.DataFrame:
    """The decoded signal rows (``SignalSet.read``) that meet the filters, strongest first."""
    m = np.ones(len(rows), dtype=bool)
    if coins:
        m &= rows["coin"].isin(list(coins)).to_numpy()
    if periods:
        m &= rows["period"].isin(list(periods)).to_numpy()
    if side in ("long", "short"):
        m &= rows["dir"].to_numpy() == (1 if side == "long" else -1)
    if min_strength:
        m &= rows["strength"].to_numpy() >= float(min_strength)
    if start:
        m &= (pd.to_datetime(rows["time"]) >= pd.Timestamp(start)).to_numpy()
    if end:
        m &= (pd.to_datetime(rows["time"]) < pd.Timestamp(end)).to_numpy()
    return rows[m].sort_values("strength", ascending=False, kind="stable")


def signal_figure(row: pd.Series, prices: PriceSource, hold: str | None, context_bars: int = 64):
    """One signal on its raw candles: the time it is known, the holding window, its direction and the realised move
    over the window (close at the end of the window against the close at the signal)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = prices.bars(str(row["coin"]))
    dt = bar_interval(bars)
    t = pd.Timestamp(row["time"])
    span = pd.Timedelta(hold) if hold else 2 * dt
    w = bars[(bars.index >= t - context_bars * dt) & (bars.index < t + span + 8 * dt)]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    draw_candles(ax, w, width=dt / pd.Timedelta(days=1) * 0.7)
    ax.axvline(t, color="#264653", lw=1)
    ax.axvspan(t, t + span, color="#ffd166", alpha=0.25)
    before = bars[bars.index < t]
    after = bars[bars.index < t + span]
    move = float(after["close"].iloc[-1] / before["close"].iloc[-1] - 1.0) if len(before) and len(after) else np.nan
    d = int(row["dir"]) if "dir" in row else int(np.sign(row["score"]))
    ax.set_title(f"{row['coin']}  {t:%Y-%m-%d %H:%M}  {'UP' if d > 0 else 'DOWN' if d < 0 else 'FLAT'}  score "
                 f"{row['score']:+.4g}  period {row.get('period', '-')}  realised over {span}: {100 * move:+.2f}%  "
                 f"(raw prices)", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _off(v):
    return v is None or v == "" or (isinstance(v, (int, float)) and v <= 0)


def controls_to_strategy(c: dict, name: str = "web") -> Strategy:
    """The strict Strategy of the controls (``DEFAULT_CONTROLS`` keys; 0 / "" = the rule is off)."""
    c = {**DEFAULT_CONTROLS, **c}
    params = json.loads(c["filter_params"] or "{}")
    entry = {"side": c["side"], "min_strength": float(c["min_strength"] or 0.0),
             "top_share": None if _off(c["top_share"]) or float(c["top_share"]) >= 1 else float(c["top_share"]),
             "score_over_sigma": None if _off(c["score_over_sigma"]) else float(c["score_over_sigma"]),
             "min_confidence": None if _off(c["min_confidence"]) else float(c["min_confidence"]),
             "filters": list(c["filters"] or []), "filter_params": {k: v for k, v in params.items() if k in
                                                                     (c["filters"] or [])}}
    if c["side"] == "quantile":
        entry["side_params"] = {"q": float(c["quantile_q"])}
    exit_ = {"max_bars": int(c["max_bars"] or 2), "hold": None if _off(c["hold"]) else str(c["hold"]),
             "take_profit": None if _off(c["take_profit"]) else float(c["take_profit"]),
             "take_profit_unit": c["take_profit_unit"],
             "stop_loss": None if _off(c["stop_loss"]) else float(c["stop_loss"]),
             "stop_loss_unit": c["stop_loss_unit"],
             "atr_timeframe": None if _off(c["atr_timeframe"]) else str(c["atr_timeframe"]), "both_hit": c["both_hit"]}
    return strategy_from_dict({
        "name": name, "entry": entry, "exit": exit_,
        "costs": {"mode": c["cost_mode"], "fee_per_side": float(c["fee_per_side"]),
                  "move_share": float(c["move_share"])},
        "book": {"per_coin": None if _off(c["per_coin"]) else int(c["per_coin"]),
                 "max_open": None if _off(c["max_open"]) else int(c["max_open"]), "admit": c["admit"],
                 "risk_per_trade": float(c["risk_per_trade"])}})


def equity_figure(trades: pd.DataFrame, risk: float, show_test: bool):
    """Equity and drawdown of each period (test only when ``show_test``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for p in pd.unique(trades["period"]) if len(trades) else []:
        if p == "test" and not show_test:
            continue
        c = equity(trades[trades["period"] == p], risk)
        if len(c):
            ax[0].plot(pd.to_datetime(c["exit_time"]), c["equity"], lw=1.2, label=str(p))
            ax[1].plot(pd.to_datetime(c["exit_time"]), 100 * c["drawdown"], lw=1.0)
    ax[0].axhline(1.0, color="#6c757d", ls="--", lw=0.8)
    ax[0].legend(fontsize=8)
    ax[0].set_title("equity per period (each period is its own book)", fontsize=10)
    ax[1].set_ylabel("drawdown %")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def run_view(sig: SignalSet, prices: PriceSource, controls: dict, show_test: bool = False,
             merge_val_test: bool = False) -> dict:
    """Run the strategy of the controls: {summary (one row per period), trades, message, equity figure}; the test
    period is hidden unless ``show_test``."""
    s = controls_to_strategy(controls)
    res = run_backtest(sig, prices, s)
    trades = res.trades
    rows = []
    for p, sm in by_period(trades, s.book.risk_per_trade, merge_val_test).items():
        hidden = p in ("test", "val+test") and not show_test
        pt = sm["per_trade"]
        rows.append({"period": p, "trades": "hidden" if hidden else sm["trades"],
                     **{k: "hidden" if hidden else (round(v, 5) if isinstance(v, float) else v) for k, v in (
                         ("final_equity", sm["final_equity"]), ("max_drawdown", sm["max_drawdown"]),
                         ("win_rate", pt.get("win_rate")), ("mean_net", pt.get("mean")),
                         ("profit_factor", pt.get("profit_factor")), ("positive_days", sm["positive_days"]))}})
    msg = ["**Rule trace:** " + " -> ".join(f"{k} {v}" for k, v in res.rule_trace),
           f"**Skipped:** {({k: v for k, v in res.skipped.items() if k != 'coins_without_prices'})}"]
    if res.skipped.get("no_prices"):
        msg.append(f"**Warning:** {res.skipped['no_prices']} signals without candles "
                   f"({', '.join(res.skipped['coins_without_prices'][:10])}).")
    if not show_test:
        msg.append("TEST is hidden: tune on VAL, then switch it on once.")
    visible = trades if show_test else trades[trades["period"] != "test"]
    return {"summary": pd.DataFrame(rows), "trades": visible.reset_index(drop=True), "message": "\n\n".join(msg),
            "figure": equity_figure(trades, s.book.risk_per_trade, show_test)}


def filter_trades(trades: pd.DataFrame, coins=None, outcome: str = "all", exit_reason=None, period=None,
                  side: str = "all", order: str = "worst") -> pd.DataFrame:
    """The trades that meet the filters (``TradeBook.where``), worst / best / earliest first."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    book = TradeBook(trades).where(coins=list(coins) if coins else None,
                                   outcome=None if outcome == "all" else outcome,
                                   exit_reason=list(exit_reason) if exit_reason else None,
                                   period=list(period) if period else None,
                                   side=None if side == "all" else side)
    t = book.trades
    if order == "worst":
        return t.sort_values("net", kind="stable")
    if order == "best":
        return t.sort_values("net", ascending=False, kind="stable")
    return t.sort_values("entry_time", kind="stable")


_SIGNAL_COLS = ("coin", "time", "period", "score", "dir", "strength", "pred_move", "sigma", "confidence", "p_up",
                "y_true")
_TRADE_COLS = ("coin", "period", "side", "entry_time", "exit_time", "exit_reason", "bars_held", "gross", "cost", "net",
               "mfe", "mae", "score", "confidence")


def _cols(frame, cols):
    """The columns of ``cols`` present in ``frame``, floats rounded for reading."""
    out = frame[[c for c in cols if c in frame.columns]].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(6)
    return out


def build_app(sig: SignalSet, prices: PriceSource, title: str = "nig_backtest"):
    """The Gradio page (call ``.launch()`` on it; in Colab ``launch(share=True)`` gives a link)."""
    import gradio as gr

    rows = sig.read()
    coins = sorted(rows["coin"].unique())
    periods = [str(p) for p in pd.unique(rows["period"])]
    hold = sig.horizon

    def picked(evt: gr.SelectData):
        """The row number of a clicked table cell."""
        return evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"## {title}\n{len(rows):,} signals, {len(coins)} coins, periods {periods}, output kind "
                    f"`{sig.kind}`, horizon {hold or '-'}")
        with gr.Tab("Signals"):
            with gr.Row():
                f_coins = gr.Dropdown(coins, multiselect=True, label="coins (empty = all)")
                f_periods = gr.CheckboxGroup(periods, label="periods")
                f_side = gr.Radio(["all", "long", "short"], value="all", label="direction")
                f_strength = gr.Number(0.0, label="min strength")
            with gr.Row():
                f_start = gr.Textbox(label="from (YYYY-MM-DD)")
                f_end = gr.Textbox(label="to")
                b_signals = gr.Button("show signals", variant="primary")
            s_count = gr.Markdown()
            s_table = gr.Dataframe(interactive=False, wrap=True)
            s_pick = gr.Number(0, precision=0, label="row to chart (click a row or type its number)")
            s_plot = gr.Plot()
            s_state = gr.State(pd.DataFrame())

            def show_signals(c, p, sd, st, a, b):
                f = filter_signals(rows, c, p, sd, st or 0.0, a or None, b or None)
                view = f.head(TABLE_ROWS).reset_index(drop=True)
                fig = signal_figure(view.iloc[0], prices, hold) if len(view) else None
                msg = f"{len(f):,} signals match (first {len(view)} shown, strongest first)"
                return msg, _cols(view, _SIGNAL_COLS), view, 0, fig

            def chart_signal(view, i):
                if view is None or view.empty or not 0 <= int(i) < len(view):
                    return None
                return signal_figure(view.iloc[int(i)], prices, hold)

            b_signals.click(show_signals, [f_coins, f_periods, f_side, f_strength, f_start, f_end],
                            [s_count, s_table, s_state, s_pick, s_plot])
            s_table.select(picked, None, s_pick)
            s_pick.change(chart_signal, [s_state, s_pick], s_plot)

        with gr.Tab("Strategy"):
            d = DEFAULT_CONTROLS
            with gr.Row():
                c_side = gr.Dropdown(SIDE_RULES.names(), value=d["side"], label="side rule")
                c_q = gr.Slider(0.05, 0.5, value=d["quantile_q"], step=0.05, label="quantile q (side = quantile)")
                c_top = gr.Slider(0.01, 1.0, value=d["top_share"], step=0.01, label="top share per period (1 = all)")
                c_strength = gr.Number(d["min_strength"], label="min strength")
            with gr.Row():
                c_sos = gr.Number(d["score_over_sigma"], label="strength > k sigma (0 = off; needs sigma)")
                c_conf = gr.Number(d["min_confidence"], label="min confidence (0 = off; needs confidence)")
                c_filters = gr.CheckboxGroup(ENTRY_FILTERS.names(), label="registered entry filters")
                c_fparams = gr.Textbox(d["filter_params"], label='filter params JSON, e.g. {"no_squeeze_short": '
                                                                 '{"max_ret_1d": 0.1}}')
            with gr.Row():
                c_hold = gr.Textbox(d["hold"], label='hold ("horizon", "8h", or empty = max bars)')
                c_bars = gr.Number(d["max_bars"], precision=0, label="max bars (when hold is empty)")
                c_tp = gr.Number(d["take_profit"], label="take profit (0 = off)")
                c_tpu = gr.Dropdown(list(UNITS), value=d["take_profit_unit"], label="TP unit")
                c_sl = gr.Number(d["stop_loss"], label="stop loss (0 = off)")
                c_slu = gr.Dropdown(list(UNITS), value=d["stop_loss_unit"], label="SL unit")
            with gr.Row():
                c_atr = gr.Textbox(d["atr_timeframe"], label='ATR frame ("4h"; empty = candles)')
                c_both = gr.Dropdown(["stop_first", "target_first"], value=d["both_hit"], label="bar hits both")
                c_cost = gr.Dropdown(COST_MODES.names(), value=d["cost_mode"], label="cost mode")
                c_fee = gr.Number(d["fee_per_side"], label="fee per side")
                c_move = gr.Number(d["move_share"], label="move share (cost mode move)")
            with gr.Row():
                c_pc = gr.Number(d["per_coin"], precision=0, label="open per coin (0 = no limit)")
                c_mo = gr.Number(d["max_open"], precision=0, label="max open in all (0 = no limit)")
                c_admit = gr.Dropdown(ADMISSION.names(), value=d["admit"], label="who gets the place")
                c_risk = gr.Number(d["risk_per_trade"], label="risk per trade")
                c_merge = gr.Checkbox(False, label="also val + test merged")
                c_test = gr.Checkbox(False, label="show TEST (decide on VAL first)")
            b_run = gr.Button("run", variant="primary")
            r_msg = gr.Markdown()
            r_summary = gr.Dataframe(interactive=False)
            r_plot = gr.Plot()
            r_state = gr.State(pd.DataFrame())
            keys = ["side", "quantile_q", "top_share", "min_strength", "score_over_sigma", "min_confidence", "filters",
                    "filter_params", "hold", "max_bars", "take_profit", "take_profit_unit", "stop_loss",
                    "stop_loss_unit", "atr_timeframe", "both_hit", "cost_mode", "fee_per_side", "move_share",
                    "per_coin", "max_open", "admit", "risk_per_trade"]
            inputs = [c_side, c_q, c_top, c_strength, c_sos, c_conf, c_filters, c_fparams, c_hold, c_bars, c_tp,
                      c_tpu, c_sl, c_slu, c_atr, c_both, c_cost, c_fee, c_move, c_pc, c_mo, c_admit, c_risk]

            def run(*vals):
                *ctrl, merge, show = vals
                try:
                    v = run_view(sig, prices, dict(zip(keys, ctrl, strict=True)), show, merge)
                except Exception as exc:  # noqa: BLE001 - a bad control value is shown, not raised into the page
                    return f"**Error:** {exc}", pd.DataFrame(), None, pd.DataFrame()
                return v["message"], v["summary"], v["figure"], v["trades"]

            b_run.click(run, [*inputs, c_merge, c_test], [r_msg, r_summary, r_plot, r_state])

        with gr.Tab("Trades"):
            with gr.Row():
                t_coins = gr.Dropdown(coins, multiselect=True, label="coins")
                t_out = gr.Radio(["all", "win", "loss"], value="all", label="outcome")
                t_exit = gr.CheckboxGroup(["take_profit", "stop_loss", "time", "data_end"], label="exit reason")
                t_period = gr.CheckboxGroup(periods, label="period")
                t_side = gr.Radio(["all", "long", "short"], value="all", label="side")
                t_order = gr.Radio(["worst", "best", "time"], value="worst", label="order")
                b_trades = gr.Button("show trades (run a strategy first)", variant="primary")
            t_count = gr.Markdown()
            t_table = gr.Dataframe(interactive=False, wrap=True)
            t_pick = gr.Number(0, precision=0, label="row to chart")
            t_plot = gr.Plot()
            t_state = gr.State(pd.DataFrame())

            def show_trades(trades, c, o, e, p, sd, order):
                f = filter_trades(trades, c, o, e, p, sd, order)
                view = f.head(TABLE_ROWS).reset_index(drop=True)
                net = f"mean net {100 * f['net'].mean():+.3f}%" if len(f) else ""
                fig = TradeBook(view).figure(0, prices) if len(view) else None
                return (f"{len(f):,} trades match {net} (first {len(view)} shown)", _cols(view, _TRADE_COLS), view,
                        0, fig)

            def chart_trade(view, i):
                if view is None or view.empty or not 0 <= int(i) < len(view):
                    return None
                return TradeBook(view).figure(int(i), prices)

            b_trades.click(show_trades, [r_state, t_coins, t_out, t_exit, t_period, t_side, t_order],
                           [t_count, t_table, t_state, t_pick, t_plot])
            t_table.select(picked, None, t_pick)
            t_pick.change(chart_trade, [t_state, t_pick], t_plot)
    return app
