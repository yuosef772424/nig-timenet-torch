"""PURPOSE: Entry ``device_smoke``: proves that the runner, the device assertion and the output contract work
    on this VM.
TAGS: device smoke, colab entry, device assertion, runner proof, rule e1
PITFALLS: the config is DeviceSmokeConfig (strict: only ``device``); the written smoke.json echoes the config it was
    given and the package file it imported, so a run shows which checkout it used.

Entry ``device_smoke``: prove that the runner, the device assertion and the contract work on this VM (ROADMAP T3).

Asserts the job's device (rule E1), computes on it, and writes ``smoke.json`` (the imported package's file, so the
run can show it used the pinned checkout, the temp dir it saw (rule E2: the runner points it into the work dir) and
the config it was given) next to ``device.json`` and ``manifest.json``.
Config keys: ``device`` (``"cuda"`` or ``"cpu"``, required: the same value the job's ``device`` field carries).
"""

import json
import sys
import tempfile
import time
from dataclasses import dataclass

import nig_timenet_torch
from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED
from nig_timenet_torch.core.config import from_dict

__all__ = ["CONFIG", "DeviceSmokeConfig", "main"]


@dataclass(frozen=True)
class DeviceSmokeConfig:
    device: str

    def __post_init__(self):
        if self.device not in EXPECTED:
            raise ValueError(f"DeviceSmokeConfig.device must be one of {EXPECTED}, got {self.device!r}")


CONFIG = DeviceSmokeConfig  # colab.job validates a job's ``config`` against the entry module's CONFIG


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    config, out, _max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(DeviceSmokeConfig, config)
    contract.write_device_json(out, cfg.device)
    (out / "smoke.json").write_text(json.dumps({"package_file": nig_timenet_torch.__file__, "config": config,
                                                "tempdir": tempfile.gettempdir(),
                                                "version": nig_timenet_torch.__version__}), encoding="utf-8")
    contract.write_manifest(out, config, input_fingerprint="none (device smoke)", started=started)
    print("device_smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
