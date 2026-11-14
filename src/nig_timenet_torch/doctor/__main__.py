"""PURPOSE: ``python -m nig_timenet_torch.doctor``: command-line entry of the readiness report (doctor/readiness.py).
TAGS: doctor, entry point, cli, readiness report
PITFALLS: runs only as ``__main__`` (a plain import, e.g. by the module scans, must not parse sys.argv or exit).
"""

import sys

from nig_timenet_torch.doctor.readiness import main

if __name__ == "__main__":
    sys.exit(main())
