"""PURPOSE: The flat model (one row per coin window, NIG regression + binary class head per price target) as run
    steps: inputs, anchors, model settings, trainer, fit settings, one chunked predict per stage.
TAGS: flat run, run steps, trainer config, stage inputs, anchors by name, head kinds, predict stage
PITFALLS: heads come from head_spec / targets_config (never a literal output name); price anchors and last_close are
    taken by column NAME (LAST_COLUMN_INDEX), never by position; the trainer config must stay byte-equal to the
    baseline's (it is hashed into the RunIdentity, tests/pins/trainer_config.json).
SOURCE: ported from feat/ar-stream aa0f3dd (workflow/flat_run.py); changes: trainer config from
    trainer.targets.targets_config (not flat_trainer_config), price targets read from the artifact, predict per stage.
"""

import copy

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.data_io.target_codec import anchor_column
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.batch_set import representative_batches
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.switches import param_groups
from nig_timenet_torch.trainer.targets import targets_config
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from nig_timenet_torch.workflow.run_common import input_key, price_targets
from nig_timenet_torch.workflow.settings import RunSettings, TrainSettings

__all__ = ["HEAD_KINDS", "LAMBDA_SCHEDULES", "anchors_by_name", "flat_inputs", "model_settings", "trainer_config",
           "make_trainer", "fit_settings_of", "predict_stage", "epoch_line", "row_coins", "batch_set"]

HEAD_KINDS = ("nig_regression", "binary_classification")
LAMBDA_SCHEDULES = {"lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"},
                    "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 5, "schedule": "cosine"}}


def anchors_by_name(last_candles: np.ndarray, idx: np.ndarray, targets,
                    codec_state: dict | None = None) -> dict:
    """{target: the price the head's label is relative to}, picked by column NAME, never by position: the codec's
    ``anchor_column`` (own last price ``last_<target>`` for "return", ``last_close`` for "logret_close"); without
    ``codec_state`` the own last price."""
    lc = np.asarray(last_candles)
    col = (lambda t: f"last_{t}") if codec_state is None else (lambda t: anchor_column(codec_state, t))
    return {t: lc[idx, LAST_COLUMN_INDEX[col(t)]] for t in targets}


def row_coins(dataset: dict, idx: np.ndarray) -> np.ndarray:
    """The coin name of each row ``idx`` (from ``asset_bounds``)."""
    names = np.empty(np.asarray(dataset["last_candles"]).shape[0], dtype=object)
    for b in dataset["asset_bounds"]:
        names[b["start"]:b["end"]] = b["name"]
    return names[np.asarray(idx)]


def batch_set(dataset: dict, idx: np.ndarray, batch_size: int, seed: int) -> list[np.ndarray]:
    """The representative batch set (MODEL_DEV_PROTOCOL section 1) of the rows ``idx``, as positions into them."""
    times = np.asarray(dataset["last_candles"])[np.asarray(idx), TS_COL].astype(np.int64)
    return representative_batches(row_coins(dataset, idx).astype(str), times, batch_size, seed=seed)


def flat_inputs(dataset: dict, idx: np.ndarray) -> tuple[torch.Tensor, dict]:
    """(x (n, T, F), {label_key: (n,)}) of the dataset rows ``idx``, float32, on the CPU; the heads are the artifact's
    price targets x ``HEAD_KINDS`` (label keys from ``head_spec``)."""
    x = torch.as_tensor(np.asarray(dataset[input_key(dataset)])[idx], dtype=torch.float32)
    keys = [head_spec(kind, t).label_key for t in price_targets(dataset) for kind in HEAD_KINDS]
    return x, {k: torch.as_tensor(np.asarray(dataset[k])[idx], dtype=torch.float32) for k in keys}


def model_settings(base: ModelSettings, targets: tuple[str, ...]) -> ModelSettings:
    """``base`` with ``price_targets`` / ``head_types`` overridden only for a target set that is not the default
    three (so the default run keeps its identity fingerprint)."""
    if targets == tuple(TARGET_COLUMNS):
        return base
    overrides = {**base.overrides, "price_targets": list(targets),
                 "head_types": {t: list(HEAD_KINDS) for t in targets}}
    return ModelSettings(base.architecture, overrides)


def trainer_config(targets, label_smoothing: float, train: TrainSettings | None = None) -> dict:
    """The oracle's trainer config (reference/data_default/config.json) with the run's label smoothing; the per-target
    part comes from ``trainer.targets.targets_config`` (HeadSpec), only the options are the run's. ``train`` carries
    the D-021 loss switches; a switch at its default leaves the config byte-equal to the baseline's (pinned)."""
    t = train or TrainSettings()
    options = {"nig_regression": {"use_calibration_loss": True, "lambda_reg_var": "lambda_reg",
                                  "lambda_calib_var": "lambda_calib"},
               "binary_classification": {"binary": True, "label_smoothing": label_smoothing}}
    if t.class_loss_weight != TrainSettings.class_loss_weight:
        options["binary_classification"]["loss_weight"] = t.class_loss_weight
    if t.beta_nll != TrainSettings.beta_nll or t.beta_nll_normalize != TrainSettings.beta_nll_normalize:
        options["nig_regression"].update(beta_nll=t.beta_nll, beta_nll_normalize=t.beta_nll_normalize)
    schedules = copy.deepcopy(LAMBDA_SCHEDULES)
    schedules["lambda_reg"]["end"] = t.lambda_reg_end
    schedules["lambda_calib"]["end"] = t.lambda_calib_end
    return {"targets": targets_config({t_: list(HEAD_KINDS) for t_ in targets}, options_by_kind=options),
            "loss": {"use_uncertainty_weighting": t.uncertainty_weighting,
                     "uncertainty_weighting_exclude_task_types": ["evidential"], "schedules": schedules},
            "constraints": []}


def make_trainer(settings: RunSettings, ms: ModelSettings, targets, seq_len: int, n_features: int,
                 device: torch.device) -> Trainer:
    """A new model and Trainer (seed the RNG before calling for a reproducible init). The optimizer's parameter groups
    come from ``trainer.switches.param_groups`` (``all_params`` = one group, as before D-021)."""
    t = settings.train
    model = build_model(ms, seq_len, n_features).to(device)
    return Trainer(model, trainer_config(targets, t.label_smoothing, t), selection_metric=t.selection_metric,
                   optimizer=lambda ps: torch.optim.AdamW(
                       param_groups(t.weight_decay_groups, model, ps, t.weight_decay), lr=t.lr,
                       weight_decay=t.weight_decay))


def fit_settings_of(settings: RunSettings) -> FitSettings:
    t = settings.train
    return FitSettings(epochs=t.epochs, batch_size=t.batch_size, seed=t.seed, early_stop=t.early_stop, trace=True,
                       grad_clip_norm=t.grad_clip_norm, ema_decay=t.ema_decay, init=t.init)


@torch.no_grad()
def predict_stage(trainer: Trainer, dataset: dict, idx: np.ndarray, x: torch.Tensor, ident: RunIdentity,
                  device: torch.device, *, stage: str, train_end, chunk: int) -> pd.DataFrame:
    """PredictionTable of the rows ``idx`` (inputs ``x``, aligned), in device-sized chunks."""
    parts = []
    for s in range(0, len(idx), chunk):
        meta = table_meta(dataset, idx[s:s + chunk], stage=stage, fold=0, train_end=train_end)
        parts.append(predict_table(trainer.model, x[s:s + chunk].to(device), meta, ident))
    return pd.concat(parts, ignore_index=True)


def epoch_line(row: dict, epochs: int) -> str:
    """One progress line per finished epoch for the Colab output."""
    def f(key):
        v = row.get(key)
        return "-" if v is None else f"{v:.4f}"
    secs = row.get("trace_epoch_seconds")
    return (f"epoch {row['epoch']}/{epochs}  train_loss {f('train_total_loss')}  val_loss {f('val_total_loss')}  "
            f"val_selection {f('val_selection_metric')}  lr {row['lr']:.2e}"
            + ("" if secs is None else f"  {secs:.0f}s"))
