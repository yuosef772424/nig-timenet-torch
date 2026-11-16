# Recovered from a past session (scratch file, not a repo module). Ran on the sandbox: downloads files from the Colab runtime through the bridge file transfer (slow, ~80 KB/s).
# Original name: scratchpad/cbdl.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
import asyncio, os, sys, time
from colab_bridge.local import transfer
from colab_bridge.local.client import BridgeClient
from colab_bridge.local.config import Settings

async def main(remote_dir, local_dir, names):
    s = Settings.load(); s.url, s.token = os.environ["CB_URL"].rstrip("/"), os.environ["COLAB_BRIDGE_TOKEN"]
    c = BridgeClient(s)
    try:
        for n in names:
            t0 = time.time()
            dest = os.path.join(local_dir, n)
            if os.path.exists(dest):
                print("have", n, flush=True); continue
            r = await transfer.download(c, f"{remote_dir}/{n}", dest, overwrite=False, roots=[local_dir], max_bytes=4 << 30)
            print(n, os.path.getsize(dest), round(time.time() - t0, 1), "s", flush=True)
    finally:
        await c.aclose()

asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
