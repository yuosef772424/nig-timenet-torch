"""Regenerates tests/parity/verification_golden/ from the OLD project's run_full_verification (not a test).

Needs a checkout of crypto-signal-prediction (read-only; default /home/user/crypto-signal-prediction) with numpy, pandas
and scikit-learn; TensorFlow is NOT needed (a fake model returns this project's synthetic PredictionTable columns).

    PYTHONPATH=src:. python -m tests.parity.make_verification_golden [--old PATH]

Step 1 (this process) writes the synthetic world (tests/eval/verification_world.py, the same arrays the parity test
rebuilds) as .npz; step 2 (tests/parity/run_old_verification.py, a separate interpreter: the two projects must not
share one) runs the old code on it and writes the golden files."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tests.parity.old_world import CASES, world_arrays

GOLDEN = Path(__file__).parent / "verification_golden"
OLD_DEFAULT = "/home/user/crypto-signal-prediction"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default=OLD_DEFAULT)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        for case, tf in CASES.items():
            npz = Path(tmp) / f"{case}.npz"
            np.savez(npz, **world_arrays(tf))
            subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("run_old_verification.py")), a.old,
                            str(npz), str(GOLDEN / case)], check=True)
    print("golden written to", GOLDEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
