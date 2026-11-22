"""PURPOSE: Entry ``staged``: the staged-training study of ROADMAP ST1 on the contract artifact (per seed the staged
    arm and the equal-step baselines B1 / B2 / B3), thin over workflow.staged_study.
TAGS: staged job, colab entry, staged training, baselines b1 b2 b3, seeds, study json, smoke job
PITFALLS: exploratory (rule C5), the model is the default preset + ``model_overrides``; the run folder of every arm is
    ``<run_dir>/<arm>_s<seed>/ckpt`` so a cut job resumes (a finished arm only re-reads its checkpoint); a config
    key not in StagedConfig raises, and ``staged`` / ``train`` are built strictly from their own dataclasses; the
    accept rule needs >= 2 seeds and b1 + b2 (rule C4); the gates of rule C6 are NOT run here (an exploratory arm
    study on the baseline model class; a winner is gated before any full training).

Config (``StagedConfig``): ``device``, ``run_dir`` (required), ``seeds`` (default 0, 1), ``artifact``, ``base_tf``,
``max_rows`` (smoke mode), ``batch_size``, ``lr``, ``model_overrides`` (over the default model), ``train`` (other
TrainSettings fields, e.g. ``steps_per_execution``), ``split`` (SplitSettings fields), ``staged`` (StageSettings
fields; absent = the card defaults), ``baselines`` (default b1, b2, b3). Outputs: ``study.json``, ``study.md`` and one
folder per arm x seed with its summary and test report.
"""

import sys
import time
from dataclasses import dataclass, field, replace

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.jobs.baseline import DEFAULT_ARTIFACT
from nig_timenet_torch.trainer.staged import StageSettings
from nig_timenet_torch.workflow import staged_study
from nig_timenet_torch.workflow.settings import RunSettings
from nig_timenet_torch.workflow.staged_report import eval_line
from nig_timenet_torch.workflow.study_settings import job_run_settings

__all__ = ["StagedConfig", "CONFIG", "run_settings", "main"]


@dataclass(frozen=True)
class StagedConfig:
    device: str
    run_dir: str
    seeds: tuple = (0, 1)
    artifact: str = DEFAULT_ARTIFACT
    base_tf: str | None = None
    max_rows: int | None = None
    batch_size: int = 512
    lr: float = 3e-4
    model_overrides: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    staged: dict = field(default_factory=dict)
    baselines: tuple = staged_study.BASELINES

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"StagedConfig.device must be one of {DEVICES}, got {self.device!r}")
        if not self.seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in self.seeds):
            raise ValueError(f"StagedConfig.seeds must be a non-empty list of integers, got {self.seeds!r}")
        if not set(self.baselines) <= set(staged_study.BASELINES):
            raise ValueError(f"StagedConfig.baselines must be a subset of {staged_study.BASELINES}")
        run_settings(self)  # the sections check their own values


CONFIG = StagedConfig


def run_settings(cfg: StagedConfig, code_sha: str = "") -> RunSettings:
    s = job_run_settings(device=cfg.device, run_dir=cfg.run_dir, artifact=cfg.artifact, base_tf=cfg.base_tf,
                         max_rows=cfg.max_rows, batch_size=cfg.batch_size, lr=cfg.lr,
                         model_overrides=cfg.model_overrides, train=cfg.train, split=cfg.split, code_sha=code_sha)
    return replace(s, train=replace(s.train, staged=from_dict(StageSettings, cfg.staged)))


def main(argv: list[str] | None = None) -> int:
    started, t_mono = time.time(), time.monotonic()
    config, out, max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(StagedConfig, config)
    contract.write_device_json(out, cfg.device)
    settings = run_settings(cfg, contract.code_sha())
    study = staged_study.run_staged_study(settings, cfg.seeds, out, baselines=cfg.baselines, ckpt_root=cfg.run_dir,
                                          on_eval=lambda r: print(eval_line(r), flush=True),
                                          deadline=contract.deadline(max_minutes, t_mono))
    contract.write_manifest(out, config, input_fingerprint=study["data_fingerprint"], started=started)
    print(f"staged {'OK' if study['finished'] else 'NOT FINISHED (run the job again to continue)'}: "
          f"{study['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
