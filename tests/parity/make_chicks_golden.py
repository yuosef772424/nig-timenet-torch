"""Regenerates tests/parity/chicks_golden/ from the OLD project's ``run_full_analysis`` (not a test).

Needs a checkout of crypto-signal-prediction (read-only; default /home/user/crypto-signal-prediction) with numpy,
pandas, scipy, scikit-learn and matplotlib; TensorFlow is NOT needed (an empty stub module stands in for it and a
fake replays this project's synthetic PredictionTable columns as the old model's outputs).

    PYTHONPATH=src:. python -m tests.parity.make_chicks_golden [--old PATH]

Step 1 (this process) writes the synthetic world (``tests/parity/chicks_world.py``, the same arrays the parity test
rebuilds) as .npz; step 2 (``tests/parity/run_old_chicks.py``, a separate interpreter: the two projects must not share
one) runs the old code on it and writes, per case, ``printed.txt``, ``analysis_outputs/`` (every saved file but the
PNGs), ``figures.json`` (a signature of each figure saved) and ``tables.json`` (the returned structures)."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tests.parity.chicks_world import CASES, old_arrays

GOLDEN = Path(__file__).parent / "chicks_golden"
OLD_DEFAULT = "/home/user/crypto-signal-prediction"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default=OLD_DEFAULT)
    a = ap.parse_args(argv)
    import json

    with tempfile.TemporaryDirectory() as tmp:
        for case, (tf, n_days, n_coins, args) in CASES.items():
            npz = Path(tmp) / f"{case}.npz"
            np.savez(npz, **old_arrays(tf, n_days, n_coins, args))
            target = GOLDEN / case
            if target.exists():
                shutil.rmtree(target)
            subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("run_old_chicks.py")), a.old, str(npz),
                            str(target)], check=True)
            (target / "args.json").write_text(json.dumps(args), encoding="utf-8")
    print("golden written to", GOLDEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
