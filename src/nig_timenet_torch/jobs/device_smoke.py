"""Entry ``device_smoke``: prove that the runner, the device assertion and the contract work on this VM (ROADMAP T3).

Asserts the job's device (rule E1), computes on it, and writes ``smoke.json`` (the imported package's file, so the
run can show it used the pinned checkout, the temp dir it saw (rule E2: the runner points it into the work dir) and
the config it was given) next to ``device.json`` and ``manifest.json``.
Config keys: ``device`` (``"cuda"`` or ``"cpu"``, required: the same value the job's ``device`` field carries).
"""

import json
import sys
import tempfile
import time

import nig_timenet_torch
from nig_timenet_torch.colab import contract


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    config, out, _max_minutes, _rest = contract.entry_args(argv)
    contract.write_device_json(out, config["device"])
    (out / "smoke.json").write_text(json.dumps({"package_file": nig_timenet_torch.__file__, "config": config,
                                                "tempdir": tempfile.gettempdir(),
                                                "version": nig_timenet_torch.__version__}), encoding="utf-8")
    contract.write_manifest(out, config, input_fingerprint="none (device smoke)", started=started)
    print("device_smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
