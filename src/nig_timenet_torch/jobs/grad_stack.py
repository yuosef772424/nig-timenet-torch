"""PURPOSE: Entry ``grad_stack``: the gradient stacking experiment (workflow/grad_stack.py) as a Colab job.
TAGS: grad stack job, colab entry, gradient features, experiment
PITFALLS: ``config`` = the GradStackConfig fields (strict); ``data`` "" builds synthetic E1 in the out folder.

Outputs: ``summary.json``, ``test_report_by_variant.csv``, ``gate_rows.csv``, ``gate_verdicts.csv``,
``modelA_grads_preds.npz`` (plus ``device.json`` / ``manifest.json`` of the run contract).
"""

import sys
import time

from nig_timenet_torch.colab import contract
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.workflow.grad_stack import GradStackConfig, run_grad_stack

__all__ = ["CONFIG", "main"]

CONFIG = GradStackConfig


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    config, out, _max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(GradStackConfig, config)
    contract.write_device_json(out, "cuda" if cfg.device == "auto" else cfg.device)
    run_grad_stack(cfg, out / "result")
    contract.write_manifest(out, config, input_fingerprint=cfg.data or "synthetic-E1", started=started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
