# Recovered from a past session (scratch file, not a repo module). Ran on the sandbox CPU: runs arm.py jobs N at a time (python pool.py JOBS.json N).
# Original name: scratchpad/pool.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
"""Scratch: run arm.py jobs, N at a time. python pool.py JOBS.json N"""
import json, subprocess, sys, pathlib
jobs = json.load(open(sys.argv[1])); n = int(sys.argv[2]); S = pathlib.Path(__file__).parent
running = []
def start(j):
    out = pathlib.Path(j["out"]); out.mkdir(parents=True, exist_ok=True)
    if (out / f"{j['name']}_s{j['seed']}" / "row.json").exists():
        return None
    cmd = [sys.executable, str(S / "arm.py"), j["artifact"], j["out"], "--name", j["name"], "--seed", str(j["seed"]),
           "--model", json.dumps(j.get("model", {})), "--train", json.dumps(j.get("train", {})), "--epochs", str(j.get("epochs", 10)),
           "--threads", "2"] + (["--oracle"] if j.get("oracle") else []) + j.get("extra", [])
    return subprocess.Popen(cmd, stdout=open(out / f"{j['name']}_s{j['seed']}.log", "w"), stderr=subprocess.STDOUT)
for j in jobs:
    p = start(j)
    if p is None: continue
    running.append(p)
    while len(running) >= n:
        running = [q for q in running if q.poll() is None]
        if len(running) >= n:
            running[0].wait()
for q in running: q.wait()
print("pool done")
