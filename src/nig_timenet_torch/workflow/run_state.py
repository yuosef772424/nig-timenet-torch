"""PURPOSE: The Run object every workflow step shares (the settings plus what the earlier steps produced) and ``need``,
    the check that a step ran after the step it depends on.
TAGS: run state, Run dataclass, shared mutable run, need check, prediction table of a stage
PITFALLS: ``Run`` is mutable and filled step by step (prepare -> build -> train -> evaluation steps); import it as
    ``workflow.run.Run`` (the public name), not from here; the training data stay on the CPU (B-019).
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from nig_timenet_torch.data_io.loading import LoadedData
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.staged import StagedResult
from nig_timenet_torch.trainer.trainer import FitResult, FitSettings, Trainer
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.settings import RunSettings

__all__ = ["Run", "need"]


@dataclass
class Run:
    settings: RunSettings
    art: LoadedData
    targets: tuple
    idx: dict                          # {"train" | "val" | "test": row indices of the artifact}
    bounds: tuple                      # (train_end, val_end)
    device: torch.device
    out: Path | None = None
    fold: int = 0                      # the fold of a rolling study (0 = a single split); written into the tables
    x: dict = field(default_factory=dict)       # {"train" | "val": inputs}
    y: dict = field(default_factory=dict)       # {"train" | "val": {label key: tensor}}
    ms: ModelSettings | None = None
    trainer: Trainer | None = None
    fit_settings: FitSettings | None = None
    ident: RunIdentity | None = None
    result: FitResult | None = None
    staged: StagedResult | None = None   # ST1: what train_staged produced (the baselines read its steps and stages)
    plan: list | None = None             # ST1: the StageSpec list train_staged ran
    extra: dict = field(default_factory=dict)  # numbers later steps add to the summary

    @property
    def dataset(self) -> dict:
        return self.art.dataset

    def table(self, stage: str) -> pd.DataFrame:
        """PredictionTable of ALL rows of ``stage`` (the model holds the weights ``weights_label`` names)."""
        need(self, "result", "train")
        if stage not in self.x:
            self.x[stage], self.y[stage] = flat_run.flat_inputs(self.dataset, self.idx[stage],
                                                                self.settings.split.cross_section,
                                                                self.settings.split.coin_channel,
                                                                self.settings.data.input_timeframes)
        return flat_run.predict_stage(self.trainer, self.dataset, self.idx[stage], self.x[stage], self.ident,
                                      self.device, stage=stage, train_end=self.bounds[0],
                                      chunk=self.settings.evaluation.chunk, fold=self.fold)


def need(run: Run, what: str, step: str) -> None:
    """Raise a clear error when a step runs before the step that sets ``Run.<what>`` (shared by the studies)."""
    if getattr(run, what) is None:
        raise RuntimeError(f"call {step}() before this step (Run.{what} is not set)")
