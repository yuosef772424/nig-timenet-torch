"""PURPOSE: Entry ``rolling``: the rolling walk-forward study of ROADMAP F2 on the contract artifact (folds x seeds
    through the run steps, per-fold tables and metrics, mean and spread per fold and across folds), thin over
    workflow.rolling_run.
TAGS: rolling job, colab entry, walk forward, folds, seeds, retrain every k, per fold metrics, smoke job
PITFALLS: exploratory (rule C5); the fold geometry (``rolling``: spans, step, embargo, val_share ...) replaces the
    single split of ``split`` (its coin exclusions and ``max_rows`` still apply per fold); every fold x seed has its own
    checkpoint folder under ``run_dir`` so a cut job resumes; ``train.staged`` makes every fold train staged;
    a config key not in RollingConfig raises and ``rolling`` / ``train`` are built strictly from their dataclasses;
    the gates of rule C6 are not run here (the study reads a configuration that passed them, ROADMAP F2);
    ``rolling.fold_ids`` runs a share of the folds: the process that completes the set writes the summary, the pooled
    days and the manifest, the others print "rolling part OK" and write none of them.

Config (``RollingConfig``): ``device``, ``run_dir`` (required), ``rolling`` (RollingSettings fields, required: the spans
have no defaults), ``artifact``, ``base_tf``, ``max_rows`` (smoke mode), ``batch_size``, ``lr``, ``model_overrides``,
``train`` (other TrainSettings fields, e.g. ``epochs``, ``staged``), ``split``, ``input_timeframes`` (the frames read
side by side, e.g. ["4h", "1h"] of an artifact made by data_io/lower_timeframe.py). Outputs: ``rolling_summary.json``
/ ``.md``, ``rolling_folds.csv`` and one folder per fold x seed (test report, test table, battery files).
"""

import sys
import time
from dataclasses import dataclass, field

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.core.config import from_dict, updated
from nig_timenet_torch.jobs.baseline import DEFAULT_ARTIFACT
from nig_timenet_torch.workflow import panel_report, rolling_run
from nig_timenet_torch.workflow.settings import RollingSettings, RunSettings
from nig_timenet_torch.workflow.study_settings import job_run_settings

__all__ = ["RollingConfig", "CONFIG", "run_settings", "rolling_settings", "main"]


@dataclass(frozen=True)
class RollingConfig:
    device: str
    run_dir: str
    rolling: dict
    artifact: str = DEFAULT_ARTIFACT
    base_tf: str | None = None
    max_rows: int | None = None
    batch_size: int = 512
    lr: float = 1e-3
    model_overrides: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    day_metrics: bool = False          # D-032: every fold also writes the day metrics; the study pools their days
    architecture: str = "nig_timenet_v2"   # model.architectures.ARCHITECTURES ("snapshot_panel": owner 2026-11-18)
    input_timeframes: list = field(default_factory=list)   # [] = the artifact's one frame; ["4h", "1h"] side by side

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"RollingConfig.device must be one of {DEVICES}, got {self.device!r}")
        run_settings(self)                              # the sections check their own values
        r = rolling_settings(self)
        if self.day_metrics and r.step < r.test_span:
            raise ValueError("RollingConfig.day_metrics pools the folds' test days: it needs step >= test_span "
                             "(test windows that do not overlap)")


CONFIG = RollingConfig


def run_settings(cfg: RollingConfig, code_sha: str = "") -> RunSettings:
    s = job_run_settings(device=cfg.device, run_dir=cfg.run_dir, artifact=cfg.artifact, base_tf=cfg.base_tf,
                         max_rows=cfg.max_rows, batch_size=cfg.batch_size, lr=cfg.lr,
                         model_overrides=cfg.model_overrides, train=cfg.train, split=cfg.split, code_sha=code_sha,
                         architecture=cfg.architecture, input_timeframes=tuple(cfg.input_timeframes))
    return updated(s, {"evaluation.battery.day_metrics": cfg.day_metrics})


def rolling_settings(cfg: RollingConfig) -> RollingSettings:
    return from_dict(RollingSettings, cfg.rolling)


def main(argv: list[str] | None = None) -> int:
    started, t_mono = time.time(), time.monotonic()
    config, out, max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(RollingConfig, config)
    contract.write_device_json(out, cfg.device)
    summary = rolling_run.run_rolling(run_settings(cfg, contract.code_sha()), rolling_settings(cfg), out,
                                      ckpt_root=cfg.run_dir, deadline=contract.deadline(max_minutes, t_mono))
    if summary.get("finished") is False:     # the deadline came first: the fold checkpoints resume
        contract.write_manifest(out, config, input_fingerprint=summary["data_fingerprint"], started=started)
        print(f"rolling NOT FINISHED (deadline): {summary['n_rows']} fold x seed row(s) this run; run the job again")
        return 0
    if summary.get("complete") is False:     # rolling.fold_ids: another process finishes the study
        print(f"rolling part OK: {summary['n_folds']} folds on disk, the summary waits for the other parts")
        return 0
    if cfg.day_metrics:        # the days of every fold's test window, pooled per seed (the folds do not overlap)
        for seed in rolling_settings(cfg).seeds:
            folders = [rolling_run.fold_dir(out, p["fold"], seed) for p in summary["per_fold"]]
            panel_report.panel_from_folders(folders, out_dir=out / f"pooled_days_seed_{seed}")
    contract.write_manifest(out, config, input_fingerprint=summary["data_fingerprint"], started=started)
    print(f"rolling OK: {summary['n_folds']} folds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
