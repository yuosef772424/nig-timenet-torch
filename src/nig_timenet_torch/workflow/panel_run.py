"""The cross-asset model as workflow steps: grids, trainer, fit, predict, report. Two-dimensional end to end.

``PanelData`` holds, for one set of dataset rows, the lazy input (days -> {"x", "mask"}) and the target grids
(days, slots, K); the Trainer fits ``GridPanelModel`` on them directly (trainer/batching.py lets it index days), the
test code reads the model's grids directly (eval/panel_eval.py), and only ``panel_table`` turns a grid into the
long PredictionTable (one row per real slot) that the existing per-asset tools read.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.prediction_schema import OUTPUT_PREFIX
from nig_timenet_torch.data.panel import PanelGrid, PanelInputs, build_panel, panel_inputs
from nig_timenet_torch.eval.panel_eval import panel_report, predict_grid
from nig_timenet_torch.model.grid_panel import GridPanelModel
from nig_timenet_torch.model.panel import build_panel_model, panel_encoder
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.trainer import Trainer
from nig_timenet_torch.workflow.prediction_table import build_table, table_meta
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["PanelData", "panel_data", "panel_trainer_config", "make_panel_trainer",
           "panel_table", "panel_scores", "panel_columns"]


@dataclass(frozen=True)
class PanelData:
    grid: PanelGrid
    inputs: PanelInputs
    y: dict[str, torch.Tensor]              # y_cls, y_reg, y_rank: (days, slots, K)


def panel_data(dataset: dict, idx: np.ndarray, settings: RunSettings, *, reg_scale: np.ndarray | None = None
               ) -> PanelData:
    """The grid of rows ``idx``; pass the train grid's ``reg_scale`` for validation and test grids."""
    if len(dataset["timeframes"]) != 1:
        raise ValueError(f"the panel run handles a single-timeframe dataset, got {list(dataset['timeframes'])}")
    grid = build_panel(dataset, idx, settings.panel, reg_scale=reg_scale)
    return PanelData(grid, panel_inputs(dataset, grid),
                     {k: torch.as_tensor(v) for k, v in grid.target_arrays().items()})


def panel_trainer_config(settings: RunSettings) -> dict:
    """Per target t (column j): a classification, a regression and (w_ic > 0) a daily rank-IC task on the grids."""
    loss, targets = settings.panel_loss, {}
    mask = "mask"
    for j, t in enumerate(settings.panel.targets):
        targets[f"panel_cls_{t}"] = {"true_key": "y_cls", "task_type": "panel_classification", "column": j,
                                     "output_keys": {"logits": "logit", "mask": mask}, "loss_weight": loss.w_cls,
                                     "label_smoothing": loss.label_smoothing}
        targets[f"panel_reg_{t}"] = {"true_key": "y_reg", "task_type": "panel_regression", "column": j,
                                     "output_keys": {"pred": "mu", "mask": mask}, "loss_weight": loss.w_reg,
                                     "loss_fn": loss.loss_fn, "huber_delta": loss.huber_delta}
        if loss.w_ic > 0:
            targets[f"panel_ic_{t}"] = {"true_key": "y_rank", "task_type": "panel_rank_ic", "column": j,
                                        "output_keys": {"pred": loss.ic_output, "mask": mask}, "loss_weight": loss.w_ic}
    return {"targets": targets, "loss": {"use_uncertainty_weighting": False}, "constraints": []}


def make_panel_trainer(settings: RunSettings, dataset: dict, device: torch.device) -> Trainer:
    x = np.asarray(dataset[f"X_{dataset['timeframes'][0]}"])
    net = settings.panel_net
    encoder = panel_encoder(x.shape[1], x.shape[2], **net.encoder)
    core = build_panel_model(encoder, d_model=net.d_model, n_cross_layers=net.n_cross_layers, num_heads=net.num_heads,
                             cross_attention=net.cross_attention, dropout=net.dropout, attn_dropout=net.attn_dropout,
                             targets=settings.panel.targets)
    t = settings.train
    return Trainer(GridPanelModel(core).to(device), panel_trainer_config(settings),
                   optimizer=lambda ps: torch.optim.AdamW(ps, lr=t.lr, weight_decay=t.weight_decay),
                   selection_metric=t.selection_metric)


def panel_columns(targets: tuple[str, ...]) -> list[str]:
    """Output column names of the long table, in the order panel_table writes them."""
    return [f"{OUTPUT_PREFIX}panel_{kind}_{t}" for kind in ("logit", "mu") for t in targets]


def panel_table(pred: dict[str, np.ndarray], data: PanelData, dataset: dict, identity: RunIdentity, *, stage: str,
                fold: int, train_end) -> pd.DataFrame:
    """The long PredictionTable of a grid prediction: one row per real slot, outputs ``y_panel_logit_<t>`` and
    ``y_panel_mu_<t>`` (mu back in the dataset's regression units, relative to the day centre)."""
    mask = data.grid.mask
    rows = data.grid.rows[mask]
    meta = table_meta(dataset, rows, stage=stage, fold=fold, train_end=train_end)
    scale = data.grid.reg_scale.astype(np.float64)
    out = {}
    for j, t in enumerate(data.grid.targets):
        out[f"{OUTPUT_PREFIX}panel_logit_{t}"] = pred["logit"][..., j][mask]
        out[f"{OUTPUT_PREFIX}panel_mu_{t}"] = pred["mu"][..., j][mask] * scale[j]
    return build_table(out, meta, identity)


def panel_scores(trainer: Trainer, data: PanelData, settings: RunSettings) -> tuple[dict, dict]:
    """(grids, report): the model's grids over ``data`` and eval/panel_eval.panel_report of them."""
    pred = predict_grid(trainer.model, data.inputs, batch_days=settings.evaluation.batch_days)
    return pred, panel_report(pred, data.grid.target_arrays(), data.grid.targets)
