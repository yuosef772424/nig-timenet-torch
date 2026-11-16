# Recovered from a past session (scratch file, not a repo module). Ran on the sandbox: runs one shell command on the Colab runtime through colab_ops.bridge (needs CB_URL, COLAB_BRIDGE_TOKEN).
# Original name: scratchpad/cbx.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
import sys
from nig_timenet_torch.colab_ops.bridge import from_env
b = from_env()
r = b.exec(sys.argv[1], timeout=float(sys.argv[2]) if len(sys.argv) > 2 else 120.0)
print(r.get("stdout", "")[-6000:]); e = r.get("stderr", "")
if e: print("STDERR:", e[-2000:])
print("exit", r.get("exit_code"))
