"""PURPOSE: Entry ``gates``: runs the MODEL_DEV_PROTOCOL gates (E1 -> E2 -> E3 x seeds) on one configuration and
    writes gate_report.json / .md keyed by its configuration hash (ROADMAP G2); a thin entry over workflow.gate_run.
TAGS: gates job, colab entry, gate report, configuration hash, e1 e2 e3, seeds, rule c6
PITFALLS: the configuration is ``architecture`` + ``model_overrides`` (merged over the default model's overrides) +
    ``train`` (TrainSettings fields except seed / epochs / device); the full job of the same configuration must use
    the same values or its key differs; without ``e3_source`` the report cannot pass (E3 not run).

Config (``GatesConfig``, strict: an unknown key raises): ``device`` and ``run_dir`` (required; the work folder of the
envs and the short runs), ``architecture``, ``model_overrides``, ``train``, and the ``GateSettings`` fields of
``workflow.gate_run`` (``seeds``, ``e1``, ``e1_epochs``, ``e2_epochs``, ``e2_permutation_seed``, ``e3_source``,
``e3_train_end``, ``e3_n_coins``, ``e3_months``, ``e3_epochs``). Outputs: ``gate_report.json``, ``gate_report.md``.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.workflow import gate_run
from nig_timenet_torch.workflow.settings import RunSettings, TrainSettings, default_model_settings

__all__ = ["GatesConfig", "CONFIG", "run_settings", "gate_settings", "main"]

_NOT_CONFIGURATION = ("seed", "epochs", "device")      # set per env / seed / job, never by the configuration


@dataclass(frozen=True)
class GatesConfig:
    device: str
    run_dir: str
    architecture: str = "nig_timenet_v2"
    model_overrides: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    seeds: tuple = (0, 1, 2)
    e1: dict = field(default_factory=dict)
    e1_epochs: int = 3
    e2_epochs: int = 3
    e2_permutation_seed: int = 0
    e3_source: str = ""
    e3_train_end: str = ""
    e3_n_coins: int = 20
    e3_months: int = 6
    e3_epochs: int = 1

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"GatesConfig.device must be one of {DEVICES}, got {self.device!r}")
        bad = sorted(set(self.train) & set(_NOT_CONFIGURATION))
        if bad:
            raise ValueError(f"GatesConfig.train may not set {bad}: the gate run sets them per env and seed")
        run_settings(self), gate_settings(self)


CONFIG = GatesConfig


def run_settings(cfg: GatesConfig, code_sha: str = "") -> RunSettings:
    """The configuration under test as RunSettings (the default model's overrides, then ``model_overrides``)."""
    base = default_model_settings()
    model = (ModelSettings(cfg.architecture, {**base.overrides, **cfg.model_overrides})
             if cfg.architecture == base.architecture else ModelSettings(cfg.architecture, dict(cfg.model_overrides)))
    train = from_dict(TrainSettings, {**cfg.train, "device": cfg.device})
    return RunSettings(run_dir=cfg.run_dir, code_sha=code_sha, model=model, train=train)


def gate_settings(cfg: GatesConfig) -> gate_run.GateSettings:
    names = {f.name for f in fields(gate_run.GateSettings)}
    return gate_run.GateSettings(**{k: getattr(cfg, k) for k in names})


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    config, out, _max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(GatesConfig, config)
    contract.write_device_json(out, cfg.device)
    sha = (subprocess.check_output(["git", "-C", os.environ["NIG_REPO_DIR"], "rev-parse", "HEAD"], text=True).strip()
           if "NIG_REPO_DIR" in os.environ else "")
    report = gate_run.run_gates(run_settings(cfg, sha), gate_settings(cfg), out)
    s = report["summary"]
    print(f"gates {report['key'][:16]}: passed {s['passed']}; gates {s['gates']}; failed {len(s['failed'])} checks")
    contract.write_manifest(out, config, input_fingerprint=report["envs"]["E1"]["fingerprint"], started=started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
