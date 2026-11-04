"""The flat model (one row per coin window; NIG-TimeNet) as workflow steps: inputs, trainer, fit, predict, report.

Used by ``workflow.run`` for a single split and for every fold of a rolling study, so both go through the same
Trainer and the same code. The trainer config is the oracle's (high, low, close; NIG regression + binary class heads).
"""

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.model.architectures import build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["TARGETS", "flat_inputs", "flat_trainer_config", "make_flat_trainer", "fit_settings_of", "flat_table",
           "flat_report"]

TARGETS = ("high", "low", "close")


def input_key(dataset: dict) -> str:
    tfs = list(dataset["timeframes"])
    if len(tfs) != 1:
        raise ValueError(f"the flat run handles a single-timeframe dataset, got {tfs}")
    return f"X_{tfs[0]}"


def flat_inputs(dataset: dict, idx: np.ndarray) -> tuple[torch.Tensor, dict]:
    """(x (n, T, F), {label_key: (n,)}) of the dataset rows ``idx``, float32, on the CPU."""
    idx = np.asarray(idx)
    x = torch.as_tensor(np.asarray(dataset[input_key(dataset)])[idx], dtype=torch.float32)
    keys = [head_spec(kind, t).label_key for t in TARGETS for kind in ("nig_regression", "binary_classification")]
    return x, {k: torch.as_tensor(np.asarray(dataset[k])[idx], dtype=torch.float32) for k in keys}


def flat_trainer_config(label_smoothing: float) -> dict:
    targets = {}
    for t in TARGETS:
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        targets[reg.name] = {"true_key": reg.label_key, "task_type": reg.task_type, "output_keys": reg.output_keys,
                             "use_calibration_loss": True, "lambda_reg_var": "lambda_reg",
                             "lambda_calib_var": "lambda_calib"}
        targets[cls.name] = {"true_key": cls.label_key, "task_type": cls.task_type, "binary": True,
                             "output_keys": cls.output_keys, "label_smoothing": label_smoothing}
    return {"targets": targets,
            "loss": {"use_uncertainty_weighting": True, "uncertainty_weighting_exclude_task_types": ["evidential"],
                     "schedules": {"lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"},
                                   "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 5,
                                                    "schedule": "cosine"}}},
            "constraints": []}


def make_flat_trainer(settings: RunSettings, dataset: dict, device: torch.device) -> Trainer:
    """A new model and Trainer for the dataset's window shape (seed the RNG before calling for a reproducible init)."""
    x = np.asarray(dataset[input_key(dataset)])
    model = build_model(settings.model, x.shape[1], x.shape[2]).to(device)
    t = settings.train
    return Trainer(model, flat_trainer_config(t.label_smoothing),
                   optimizer=lambda ps: torch.optim.AdamW(ps, lr=t.lr, weight_decay=t.weight_decay),
                   selection_metric=t.selection_metric)


def fit_settings_of(settings: RunSettings) -> FitSettings:
    t = settings.train
    return FitSettings(epochs=t.epochs, batch_size=t.batch_size, seed=t.seed, early_stop=t.early_stop)


def flat_table(trainer: Trainer, dataset: dict, idx: np.ndarray, identity: RunIdentity, settings: RunSettings, *,
               stage: str, fold: int, train_end) -> pd.DataFrame:
    """PredictionTable of the rows ``idx`` (chunked, so a big test split fits the device)."""
    dev = next(trainer.model.parameters()).device
    x, _ = flat_inputs(dataset, idx)
    step, parts = settings.evaluation.batch_size * 16, []
    for s in range(0, len(idx), step):
        meta = table_meta(dataset, np.asarray(idx)[s:s + step], stage=stage, fold=fold, train_end=train_end)
        parts.append(predict_table(trainer.model, x[s:s + step].to(dev), meta, identity,
                                   batch_size=settings.evaluation.batch_size))
    return pd.concat(parts, ignore_index=True)


def flat_report(table: pd.DataFrame, dataset: dict, idx: np.ndarray) -> dict:
    """target_report of a flat prediction table against the dataset's labels of the same rows."""
    _, y = flat_inputs(dataset, idx)
    lc = np.asarray(dataset["last_candles"])
    return target_report(table, {k: v.numpy() for k, v in y.items()},
                         anchors={t: lc[np.asarray(idx), i] for i, t in enumerate(TARGETS)},
                         codec_state=TARGET_CODECS["return"]().fit(dataset))
