"""Export reference values of the TF project's trainer/schedules.py (ROADMAP T1; read by
tests/trainer/test_schedules_parity.py).

Usage:  python reference/export_schedules.py <path to crypto-signal-prediction/trainer/schedules.py>

The TF file is executed as it is, with numpy and a stub for the two names it takes from the shared notebook
namespace (``tf.keras.callbacks.Callback`` for the ParamScheduler class statement, typing names); only the pure
functions ``schedule_value``, ``cosine_warm_restarts`` and ``build_lr_schedule_fn`` are called. Output:
reference/data_schedules/schedules.json (float64 values, repr-exact).
"""

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Callable, Optional

import numpy as np

OUT = Path(__file__).resolve().parent / "data_schedules" / "schedules.json"
SOURCE_COMMITS = {"aeafc564f25c78f252936affbd374f68aa49b8d9": "the port base", "cea9cd58b2689fcc66a4e000c154428988c9df7f":
                  "main when exported; trainer/schedules.py has the same sha256 as at the port base"}

VALUE_CASES = [(0.0, 0.05, 5), (0.0, 0.1, 5), (1.0, 0.0, 10), (0.2, 0.9, 3), (0.0, 1.0, 0)]
RESTART_CASES = [(5e-5, 5e-7, 10, 1.5), (1e-3, 0.0, 4, 2.0), (3e-4, 1e-5, 3, 1.0)]
LR_CONFIGS = {
    "tf_default": {"lr_initial": 5e-5, "lr_min": 5e-7, "lr_warmup_epochs": 3,
                   "lr_schedule": {"type": "cosine_restarts", "cycle_length": 10, "cycle_mult": 1.5}},
    "cosine_total_20": {"lr_initial": 1e-3, "lr_min": 1e-5, "lr_warmup_epochs": 0,
                        "lr_schedule": {"type": "cosine", "total_epochs": 20}},
    "cosine_warmup_2": {"lr_initial": 1e-3, "lr_min": 1e-5, "lr_warmup_epochs": 2,
                        "lr_schedule": {"type": "cosine", "total_epochs": 10}},
    "cosine_no_total": {"lr_initial": 2e-4, "lr_min": 0.0, "lr_schedule": {"type": "cosine"}},
    "constant": {"lr_initial": 3e-4, "lr_min": 1e-6, "lr_warmup_epochs": 4, "lr_schedule": {"type": "constant"}},
    "constant_no_schedule_key": {"lr_initial": 3e-4, "lr_min": 1e-6},
    "restarts_defaults": {"lr_initial": 1e-3, "lr_min": 1e-5, "lr_schedule": {"type": "cosine_restarts"}},
}


def main(schedules_py: str) -> None:
    src = Path(schedules_py).read_text(encoding="utf-8")
    tf = types.SimpleNamespace(keras=types.SimpleNamespace(callbacks=types.SimpleNamespace(Callback=object)),
                               Variable=object)
    ns = {"np": np, "tf": tf, "Optional": Optional, "Callable": Callable}
    exec(compile(src, schedules_py, "exec"), ns)

    out = {
        "_doc": "Values computed by executing the TF project's trainer/schedules.py (reference/export_schedules.py).",
        "source_file": "trainer/schedules.py", "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "source_repo": "yuosef772424/crypto-signal-prediction", "source_commits": SOURCE_COMMITS,
        "schedule_value": [
            {"start": s, "end": e, "warmup_epochs": w, "schedule": kind,
             "values": [float(ns["schedule_value"](s, e, w, kind, ep)) for ep in range(0, 14)]}
            for (s, e, w) in VALUE_CASES for kind in ("linear", "cosine", "exponential", "sqrt")],
        "cosine_warm_restarts": [
            {"lr_initial": a, "lr_min": b, "cycle_length": c, "cycle_mult": m,
             "values": [float(ns["cosine_warm_restarts"](ep, a, b, c, m)) for ep in range(0, 90)]}
            for (a, b, c, m) in RESTART_CASES],
        "lr_schedule_fn": {
            name: {"opt_cfg": cfg, "values": [float(ns["build_lr_schedule_fn"](cfg)(ep)) for ep in range(0, 45)]}
            for name, cfg in LR_CONFIGS.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
