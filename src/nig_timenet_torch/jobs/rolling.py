"""PURPOSE: Entry ``rolling``: the rolling walk-forward study of ROADMAP F2 on the contract artifact (folds x seeds
    through the run steps, per-fold tables and metrics, mean and spread per fold and across folds), thin over
    workflow.rolling_run.
TAGS: rolling job, colab entry, walk forward, folds, seeds, retrain every k, per fold metrics, smoke job
PITFALLS: exploratory (rule C5); the fold geometry (``rolling``: spans, step, embargo, val_share ...) replaces the
    single split of ``split`` (its coin exclusions and ``max_rows`` still apply per fold); every fold x seed has its own
    checkpoint folder under ``run_dir`` so a cut job resumes; ``train.staged`` makes every fold train staged;
    a config key not in RollingConfig raises and ``rolling`` / ``train`` are built strictly from their dataclasses;
    the gates of rule C6 are not run here (the study reads a configuration that passed them, ROADMAP F2).

Config (``RollingConfig``): ``device``, ``run_dir`` (required), ``rolling`` (RollingSettings fields, required: the spans
have no defaults), ``artifact``, ``base_tf``, ``max_rows`` (smoke mode), ``batch_size``, ``lr``, ``model_overrides``,
``train`` (other TrainSettings fields, e.g. ``epochs``, ``staged``), ``split``. Outputs: ``rolling_summary.json`` /
``.md``, ``rolling_folds.csv`` and one folder per fold x seed (test report, test table, battery files).
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.jobs.baseline import DEFAULT_ARTIFACT
from nig_timenet_torch.workflow import rolling_run
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

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"RollingConfig.device must be one of {DEVICES}, got {self.device!r}")
        run_settings(self), rolling_settings(self)  # the sections check their own values


CONFIG = RollingConfig


def run_settings(cfg: RollingConfig, code_sha: str = "") -> RunSettings:
    return job_run_settings(device=cfg.device, run_dir=cfg.run_dir, artifact=cfg.artifact, base_tf=cfg.base_tf,
                            max_rows=cfg.max_rows, batch_size=cfg.batch_size, lr=cfg.lr,
                            model_overrides=cfg.model_overrides, train=cfg.train, split=cfg.split, code_sha=code_sha)


def rolling_settings(cfg: RollingConfig) -> RollingSettings:
    return from_dict(RollingSettings, cfg.rolling)


def _code_sha() -> str:
    return subprocess.check_output(["git", "-C", os.environ["NIG_REPO_DIR"], "rev-parse", "HEAD"], text=True).strip()


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    config, out, _max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(RollingConfig, config)
    contract.write_device_json(out, cfg.device)
    summary = rolling_run.run_rolling(run_settings(cfg, _code_sha()), rolling_settings(cfg), out,
                                      ckpt_root=cfg.run_dir)
    contract.write_manifest(out, config, input_fingerprint=summary["data_fingerprint"], started=started)
    print(f"rolling OK: {summary['n_folds']} folds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
