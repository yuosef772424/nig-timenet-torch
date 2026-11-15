"""PURPOSE: Synchronous client of the Colab bridge agent (exec, start, output, wait, runtime status, small file put)
    over the signed HTTP protocol of colab-mcp-bridge; what the CLI and the notebooks call.
TAGS: colab bridge client, remote exec, background process, wait for process, runtime status, file put
PITFALLS: needs the optional ``colab_bridge`` package (imported on first use); URL / token only from CB_URL and
    COLAB_BRIDGE_TOKEN (never printed); ``wait`` long-polls <= 40 s per call, so a caller's own timeout must exceed
    its ``limit``; a process outlives a dropped tunnel, a runtime restart loses it; ``put`` is for small text files.
"""

import asyncio
import base64
import os
import time
from typing import Any

__all__ = ["Bridge", "from_env"]


def _client(url: str, token: str):
    from colab_bridge.local.client import BridgeClient
    from colab_bridge.local.config import Settings

    s = Settings.load()
    s.url, s.token = url.rstrip("/"), token
    return BridgeClient(s)


class Bridge:
    def __init__(self, url: str, token: str):
        if not url or not token:
            raise ValueError("set CB_URL (the tunnel URL) and COLAB_BRIDGE_TOKEN (the BRIDGE_TOKEN of the runtime)")
        self.url, self._token = url, token

    def call(self, op: str, args: dict | None = None, *, mutating: bool = False, max_wait: float = 0.0) -> Any:
        async def go():
            c = _client(self.url, self._token)
            try:
                return await c.call(op, args or {}, mutating=mutating, max_wait=max_wait)
            finally:
                await c.aclose()

        return asyncio.run(go())

    def exec(self, command: str, timeout: float = 120.0, max_bytes: int = 30000) -> dict:
        return self.call(
            "exec",
            {"command": command, "timeout": timeout, "max_output_bytes": max_bytes},
            mutating=True,
            max_wait=timeout + 90,
        )

    def start(self, command: str, name: str) -> str:
        return self.call(
            "process_start", {"command": command, "name": name, "merge_stderr": True}, mutating=True, max_wait=120
        )["process_id"]

    def status(self, pid: str, wait: float = 0.0) -> dict:
        return self.call("process_status", {"process_id": pid, "wait_seconds": wait})

    def tail(self, pid: str, lines: int = 40) -> str:
        return self.call("process_output", {"process_id": pid, "tail_lines": lines}).get("text", "")

    def wait(self, pid: str, limit: float = 400.0) -> dict:
        """Long-poll until the process leaves ``running`` or ``limit`` seconds pass; returns the last status."""
        t0 = time.time()
        while True:
            st = self.status(pid, wait=min(40.0, max(1.0, limit - (time.time() - t0))))
            if st["status"] != "running" or time.time() - t0 >= limit:
                return st

    def runtime(self) -> dict:
        d = self.call("runtime_status")
        return {
            "gpu": d["gpu"]["devices"][0]["name"] if d["gpu"]["available"] else None,
            "cores": d["cpu"]["logical_cores"],
            "ram_gb": d["ram"]["total_gb"],
            "runtime_id": d.get("runtime_id"),
            "processes_running": d.get("processes_running"),
        }

    def put(self, local_path: str, remote_path: str) -> dict:
        """Copy a small text file to the runtime (base64 through exec)."""
        data = base64.b64encode(open(local_path, "rb").read()).decode()
        return self.exec(f"echo {data} | base64 -d > {remote_path} && wc -c {remote_path}")


def from_env() -> Bridge:
    return Bridge(os.environ.get("CB_URL", ""), os.environ.get("COLAB_BRIDGE_TOKEN", ""))
