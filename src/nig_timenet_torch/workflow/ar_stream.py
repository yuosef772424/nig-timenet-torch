"""One-call compositions of the autoregressive candle stream (single-coin and panel): raw OHLCV frames
in, one report out. The Colab notebooks walk the same steps one by one (thin runners, rule E4/G6); these
functions are the programmatic path and the import seam that keeps every module of the run reachable
from the package (rule G8)."""

from nig_timenet_torch.data.ar_windows import (
    build_windows,
    candle_features,
    contiguous_runs,
    fit_volume_median,
    panel_windows,
    split_panel,
    split_windows,
)
from nig_timenet_torch.eval.ar_stream import (
    direction_control,
    panel_direction_control,
    panel_rollout_metrics,
    rollout_metrics,
)
from nig_timenet_torch.trainer.ar_stream import ArStreamSettings, build_ar_stream_model, fit_ar_stream
from nig_timenet_torch.trainer.panel_ar_stream import (
    PanelArStreamSettings,
    build_panel_ar_stream_model,
    fit_panel_ar_stream,
)


def _windows(frames: dict, interval: str, context_len: int, future_len: int) -> dict:
    """Gap-free windows per coin: five scale-free features, cut only inside contiguous stretches, with
    end timestamps (the panel grid aligns coins on them)."""
    coins = {}
    for name, df in frames.items():
        feats = candle_features(df, fit_volume_median(df))
        coins[name] = build_windows(feats, contiguous_runs(df.index, interval), context_len,
                                    future_len, index=df.index)
    return coins


def run_ar_stream(frames: dict, interval: str, context_len: int, horizon_len: int, eval_horizon: int,
                  val_share: float = 0.15, test_share: float = 0.15, changes: dict | None = None,
                  device: str = "cpu") -> dict:
    """The single-coin stream end to end: windows -> split -> fit -> free-horizon rollout metrics and
    the shuffled-labels control. ``eval_horizon`` may differ from ``horizon_len`` (the windows carry the
    longer future); ``changes`` override any ``ArStreamSettings`` field (e.g. a tiny smoke run)."""
    future_len = max(horizon_len, eval_horizon)
    data = split_windows(_windows(frames, interval, context_len, future_len), val_share, test_share)
    settings = ArStreamSettings(context_len=context_len, horizon_len=horizon_len, device=device,
                                **(changes or {}))
    model = build_ar_stream_model(settings, n_features=data["train_X"].shape[2])
    history = fit_ar_stream(model, data["train_X"], data["train_Y"], data["train_dirs"],
                            data["val_X"], data["val_Y"], data["val_dirs"], settings)
    report = rollout_metrics(model, data["test_X"], data["test_Y"], data["test_dirs"],
                             data["test_last_ret"], device, horizon_len=eval_horizon)
    control = direction_control(model, data["test_X"], data["test_dirs"], device,
                                horizon_len=eval_horizon)
    return {"data": data, "settings": settings, "model": model, "history": history,
            "report": report, "control": control}


def run_panel_ar_stream(frames: dict, interval: str, context_len: int, horizon_len: int,
                        eval_horizon: int, val_share: float = 0.15, test_share: float = 0.15,
                        changes: dict | None = None, device: str = "cpu") -> dict:
    """The panel twin: the timestamp-aligned grid (all coins in lockstep) -> split -> fit -> pooled
    rollout metrics and control; the same knobs, plus ``cross_layers``, through ``PanelArStreamSettings``."""
    future_len = max(horizon_len, eval_horizon)
    panel = panel_windows(_windows(frames, interval, context_len, future_len))
    data = split_panel(panel, val_share, test_share)
    settings = PanelArStreamSettings(context_len=context_len, horizon_len=horizon_len, device=device,
                                     **(changes or {}))
    model = build_panel_ar_stream_model(settings, n_features=data["train_X"].shape[3],
                                        n_coins=data["train_X"].shape[1])
    history = fit_panel_ar_stream(model, data["train_X"], data["train_Y"], data["train_dirs"],
                                  data["val_X"], data["val_Y"], data["val_dirs"], settings)
    report = panel_rollout_metrics(model, data["test_X"], data["test_Y"], data["test_dirs"],
                                   data["test_last_ret"], device, horizon_len=eval_horizon)
    control = panel_direction_control(model, data["test_X"], data["test_dirs"], device,
                                       horizon_len=eval_horizon)
    return {"data": data, "settings": settings, "model": model, "history": history,
            "report": report, "control": control}
