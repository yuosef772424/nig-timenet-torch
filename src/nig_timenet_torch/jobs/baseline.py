"""PURPOSE: Entry ``baseline``: trains one seed of the default NIG-TimeNet model on the contract artifact, then writes
    the summary, health report and test report (ROADMAP T5r); a thin entry over workflow.run.
TAGS: baseline job, colab entry, training run, test report, health report, resume, smoke job
PITFALLS: every data fact (price targets, codec mode and scale, timeframe, split dates) is read from the artifact, never
    assumed; a job config key not in BaselineConfig raises; max_rows (smoke mode) strides each split down, everything
    else is the same code path; the run steps live in workflow/run.py, not here.

Entry ``baseline``: one seed of the default model on the contract artifact (ROADMAP T5r, job ``baseline-001``).

The default model is what TF ``main.ipynb`` builds (BUILD_PLAN.md V1): ``nig_timenet_v2`` with the whole
``ANTI_MEMORIZATION_CONFIG`` (GRU encoder + level passthrough), classification label smoothing 0.1.

Config (``BaselineConfig``, strict: an unknown key raises): ``device`` (required, the job's device), ``seed``,
``epochs``, ``batch_size``, ``run_dir`` (required; a Drive folder: the checkpoint lives there, so a cut session resumes
by running the same job again), optional ``artifact`` (default ``DEFAULT_ARTIFACT``), ``embargo_hours`` (default 0: the
split already purges by label end), ``health_max_rows``, ``split_dates`` (``{"train_end", "val_end"}``; default: the
artifact's, else the 80/10/10 time quantiles), ``max_rows`` (smoke mode: at most this many rows per split, evenly
strided; default off), ``base_tf`` (the bar interval the job expects, e.g. "4h": another artifact is refused;
default: accept the artifact's). Stops cleanly before ``--max-minutes`` (the per-epoch checkpoint is the resume point).
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.flat_run import epoch_line
from nig_timenet_torch.workflow.settings import EvalSettings, RunSettings, SplitSettings, TrainSettings

__all__ = ["BaselineConfig", "CONFIG", "DEFAULT_ARTIFACT", "DEFAULT_HEALTH_MAX_ROWS", "SUMMARY", "TEST_REPORT",
           "run_settings", "main"]

DEFAULT_ARTIFACT = "/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1"
DEFAULT_HEALTH_MAX_ROWS = 100_000
SUMMARY, TEST_REPORT = steps.SUMMARY, steps.TEST_REPORT


@dataclass(frozen=True)
class BaselineConfig:
    """The job's ``config`` object. Built with the strict ``from_dict`` (unknown keys raise, rule B5); the colab job
    validator builds it too, so a bad job file is refused at queue time, not on the GPU. Value checks live in the
    RunSettings sections it is mapped to (``run_settings``)."""
    device: str
    seed: int
    epochs: int
    batch_size: int
    run_dir: str
    artifact: str = DEFAULT_ARTIFACT
    embargo_hours: float = 0.0
    health_max_rows: int = DEFAULT_HEALTH_MAX_ROWS
    split_dates: dict | None = None
    max_rows: int | None = None
    base_tf: str | None = None  # the bar interval the job expects ("4h"); a different artifact is refused at load

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"BaselineConfig.device must be one of {DEVICES}, got {self.device!r}")
        if self.split_dates is not None and (not isinstance(self.split_dates, dict)
                                             or set(self.split_dates) != {"train_end", "val_end"}):
            raise ValueError("BaselineConfig.split_dates must be None or {train_end, val_end}, "
                             f"got {self.split_dates!r}")
        run_settings(self)  # the sections check epochs, batch_size, seed, run_dir, max_rows, embargo_hours, ...


CONFIG = BaselineConfig  # colab.job validates a job's ``config`` against the entry module's CONFIG


def run_settings(cfg: BaselineConfig, code_sha: str = "") -> RunSettings:
    """The job config as the run's settings (everything the job does not name is the RunSettings default)."""
    sd = cfg.split_dates or {}
    return RunSettings(
        run_dir=cfg.run_dir, code_sha=code_sha,
        data=LoaderSettings(name=cfg.artifact, mmap=True, expect_base_tf=cfg.base_tf),
        split=SplitSettings(embargo_hours=cfg.embargo_hours, max_rows=cfg.max_rows, **sd),
        train=TrainSettings(epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed, device=cfg.device),
        evaluation=EvalSettings(health_max_rows=cfg.health_max_rows))


def _code_sha() -> str:
    return subprocess.check_output(["git", "-C", os.environ["NIG_REPO_DIR"], "rev-parse", "HEAD"], text=True).strip()


def main(argv: list[str] | None = None) -> int:
    started, t_mono = time.time(), time.monotonic()
    config, out, max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(BaselineConfig, config)
    dev_rep = contract.write_device_json(out, cfg.device)
    budget = max_minutes * 60
    deadline = t_mono + budget - min(600.0, 0.2 * budget)  # the rest is for the health report and the outputs
    r = steps.prepare(run_settings(cfg, _code_sha()), out)
    steps.build(r)
    steps.train(r, deadline=deadline, on_epoch=lambda row: print(epoch_line(row, cfg.epochs), flush=True))
    steps.health_step(r)
    rep = steps.test_step(r)
    print(f"test evaluation ({rep['n_rows']} rows, best-epoch weights):", flush=True)
    print("\n".join(steps.test_lines(rep)), flush=True)
    steps.write_summary(r, {"device": cfg.device, "gpu_name": dev_rep.get("gpu_name")})
    steps.battery_step(r)  # after the summary: a battery failure keeps every other output
    contract.write_manifest(out, config, input_fingerprint=r.art.manifest.data_fingerprint, started=started)
    print(f"baseline OK: seed {cfg.seed}, epochs {r.result.epochs_done}/{cfg.epochs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
