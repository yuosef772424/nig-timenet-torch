"""PURPOSE: Entry ``panel``: the multi-coin (panel) study of the owner task 2026-11-16 on the contract artifact (named
    arms x seeds on whole timestamps, each scored with the day metrics), thin over workflow.panel_study.
TAGS: panel job, colab entry, multi coin, cross section, arms, seeds, day metrics, winning days, resumable, smoke job
PITFALLS: exploratory (rule C5); the split is always a cross-section (``cross_section`` True, ``top_coins`` from the
    config); every arm keeps its checkpoint and, when finished, its row and frames under ``<run_dir>/<arm>_s<seed>``
    (a Drive folder), so a cut job continues where it stopped and a finished arm is never trained again; the job
    stops at ``--max-minutes`` minus a reserve and exits 0 with ``finished`` false in ``study.json`` (queue it again);
    a config key not in PanelConfig raises; the gates of rule C6 are NOT run here (an arm study on the baseline
    model class; a winner is gated before it becomes a default).

Config (``PanelConfig``): ``device``, ``run_dir`` (required), ``seeds`` (default 0, 1), ``arms`` (names of
``workflow.panel_study.ARMS``; default ``PANEL_001_ARMS``, the five arms of panel-001), ``extra_arms`` ({name:
{"model": {...}, "train": {...}}}), ``artifact``, ``base_tf``, ``max_rows`` (smoke mode: whole timestamps, evenly
strided), ``top_coins``, ``batch_size``, ``lr``, ``epochs``, ``patience``, ``selection_metric``, ``battery`` (the
evaluation battery per arm, default on), ``model_overrides`` (every arm), ``train`` (other TrainSettings fields, every
arm), ``split`` (other SplitSettings fields). Outputs: ``study.json``, ``study.md``, one folder per arm x seed.
"""

import sys
import time
from dataclasses import dataclass, field, replace

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.jobs.baseline import DEFAULT_ARTIFACT
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.workflow import panel_study
from nig_timenet_torch.workflow.flat_run import epoch_line
from nig_timenet_torch.workflow.settings import BatterySettings, EvalSettings, RunSettings
from nig_timenet_torch.workflow.study_settings import job_run_settings

__all__ = ["PanelConfig", "CONFIG", "run_settings", "arms_of", "main"]


@dataclass(frozen=True)
class PanelConfig:
    device: str
    run_dir: str
    seeds: tuple = (0, 1)
    arms: tuple = panel_study.PANEL_001_ARMS
    extra_arms: dict = field(default_factory=dict)
    artifact: str = DEFAULT_ARTIFACT
    base_tf: str | None = None
    max_rows: int | None = None
    top_coins: int | None = 100
    batch_size: int = 512
    lr: float = 3e-4
    epochs: int = 25
    patience: int = 6
    selection_metric: str = "val_rank_ic_reg"
    battery: bool = True               # also the evaluation battery per arm (T-profit sections; n_variants = arms)
    model_overrides: dict = field(default_factory=lambda: {"encoder": "gru"})
    train: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"PanelConfig.device must be one of {DEVICES}, got {self.device!r}")
        if not self.seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in self.seeds):
            raise ValueError(f"PanelConfig.seeds must be a non-empty list of integers, got {self.seeds!r}")
        unknown = sorted(set(self.arms) - set(panel_study.ARMS))
        if unknown or not (self.arms or self.extra_arms):
            raise ValueError(f"PanelConfig.arms must be names of {tuple(panel_study.ARMS)} (unknown: {unknown}); "
                             f"other arms go to extra_arms")
        if set(self.extra_arms) & set(panel_study.ARMS):
            raise ValueError("PanelConfig.extra_arms may not reuse the name of a built-in arm")
        if {"cross_section", "top_coins"} & set(self.split) or {"epochs", "selection_metric", "early_stop"} & set(
                self.train):
            raise ValueError("config.split / config.train may not set cross_section, top_coins, epochs, "
                             "selection_metric or early_stop: the config has its own keys for them")
        base = run_settings(self)                                  # the sections check their own values
        for arm in arms_of(self).values():
            panel_study.arm_settings(base, arm, 0, "x")            # an arm with an unknown field raises now


CONFIG = PanelConfig


def arms_of(cfg: PanelConfig) -> dict:
    """{arm name: overrides} in the order the study runs them."""
    return {**{name: panel_study.ARMS[name] for name in cfg.arms}, **cfg.extra_arms}


def run_settings(cfg: PanelConfig, code_sha: str = "") -> RunSettings:
    s = job_run_settings(device=cfg.device, run_dir=cfg.run_dir, artifact=cfg.artifact, base_tf=cfg.base_tf,
                         max_rows=cfg.max_rows, batch_size=cfg.batch_size, lr=cfg.lr,
                         model_overrides=cfg.model_overrides, code_sha=code_sha,
                         train={**cfg.train, "epochs": cfg.epochs, "selection_metric": cfg.selection_metric,
                                "early_stop": EarlyStopSettings(patience=cfg.patience)},
                         split={**cfg.split, "cross_section": True, "top_coins": cfg.top_coins})
    # the study writes the day metrics itself (it needs their numbers); the battery adds its per-row T-profit
    # sections, its CIs widened by the number of arms tried
    return replace(s, evaluation=EvalSettings(battery=BatterySettings(enabled=cfg.battery,
                                                                      n_variants=len(arms_of(cfg)))))


def main(argv: list[str] | None = None) -> int:
    started, t_mono = time.time(), time.monotonic()
    config, out, max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(PanelConfig, config)
    contract.write_device_json(out, cfg.device)
    deadline = contract.deadline(max_minutes, t_mono)          # the rest is for the reports and the publish
    study = panel_study.run_panel_study(
        run_settings(cfg, contract.code_sha()), arms_of(cfg), cfg.seeds, out, ckpt_root=cfg.run_dir, deadline=deadline,
        on_epoch=lambda row: print(epoch_line(row, cfg.epochs), flush=True))
    contract.write_manifest(out, config, input_fingerprint=study["data_fingerprint"], started=started)
    print(f"panel {'OK' if study['finished'] else 'NOT FINISHED (run the job again to continue)'}: "
          f"{sum(len(r) for r in study['arms'].values())} arm runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
