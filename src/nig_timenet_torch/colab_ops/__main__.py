"""PURPOSE: ``python -m nig_timenet_torch.colab_ops <command>``: the Colab steps of every session as one CLI
    (rt, exec, start, wait, out, put, setup, arms, doctor, runs).
TAGS: colab ops cli, setup runtime, start job, wait job, read output, repeated steps
PITFALLS: needs CB_URL and COLAB_BRIDGE_TOKEN in the environment; ``setup`` / ``arms`` / ``runs`` start a BACKGROUND
    process and print its id (follow it with ``wait`` / ``out``); output of ``exec`` is capped at 30 kB.
"""

import argparse
import json
import sys
from pathlib import Path

from nig_timenet_torch.colab_ops import recipes
from nig_timenet_torch.colab_ops.bridge import from_env

SCRIPTS = Path(__file__).parent / "scripts"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colab_ops", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rt", help="GPU / cores / RAM / runtime id of the connected runtime")
    e = sub.add_parser("exec", help="run one command and wait")
    e.add_argument("command")
    e.add_argument("--timeout", type=float, default=120)
    s = sub.add_parser("start", help="start a background process")
    s.add_argument("command")
    s.add_argument("name")
    w = sub.add_parser("wait", help="wait for a process (long poll) and print its last lines")
    w.add_argument("pid")
    w.add_argument("--limit", type=float, default=400)
    w.add_argument("--lines", type=int, default=25)
    o = sub.add_parser("out", help="last lines of a process")
    o.add_argument("pid")
    o.add_argument("--lines", type=int, default=40)
    u = sub.add_parser("put", help="copy a small file to the runtime")
    u.add_argument("local")
    u.add_argument("remote")
    t = sub.add_parser("setup", help="clone / update a branch, install, copy the artifact (idempotent)")
    t.add_argument("branch")
    t.add_argument("--dest", default="/content/nig-timenet-torch")
    t.add_argument("--no-artifact", action="store_true")
    t.add_argument("--no-install", action="store_true")
    t.add_argument(
        "--torch-key", default=None, help="torch version the env cache key is for (a fresh Colab's, e.g. 2.14.1+cu130)"
    )
    a = sub.add_parser("arms", help="small-sample encoder x lr arms (quick_arms.py) in the background")
    a.add_argument("out")
    a.add_argument("--arms", default="gru:3e-4,transformer:3e-4")
    a.add_argument("--rows", type=int, default=20000)
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="cuda")
    a.add_argument("--no-battery", action="store_true")
    a.add_argument("--spe", type=int, default=1, help="steps_per_execution")
    a.add_argument("--n-variants", type=int, default=1)
    d = sub.add_parser("doctor", help="model doctor on a finished run's best weights, in the background")
    d.add_argument("run_dir")
    d.add_argument("out")
    d.add_argument("--encoder", default="gru")
    d.add_argument("--lr", type=float, default=3e-4)
    d.add_argument("--rows", type=int, default=20000)
    d.add_argument("--epochs", type=int, default=6)
    d.add_argument("--device", default="cuda")
    r = sub.add_parser("runs", help="single-model runs of the sequential-gradient branch (needs `setup` of it)")
    r.add_argument("models")
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--lr-min", type=float, default=1e-6)
    r.add_argument("--epochs", type=int, default=10)
    r.add_argument("--rows", type=int, default=320000)
    r.add_argument("--device", default="cuda")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    b = from_env()
    if a.cmd == "rt":
        print(json.dumps(b.runtime()))
    elif a.cmd == "exec":
        r = b.exec(a.command, a.timeout)
        print(r.get("stdout", ""), r.get("stderr", ""), f"exit {r.get('exit_code')}")
    elif a.cmd == "start":
        print(b.start(a.command, a.name))
    elif a.cmd == "wait":
        st = b.wait(a.pid, a.limit)
        print(st["status"], st.get("exit_code"), round(st["duration_seconds"]))
        print(b.tail(a.pid, a.lines))
    elif a.cmd == "out":
        print(b.tail(a.pid, a.lines))
    elif a.cmd == "put":
        print(b.put(a.local, a.remote).get("stdout", ""))
    elif a.cmd == "setup":
        r = b.exec(
            recipes.setup_script(
                a.branch, a.dest, artifact=not a.no_artifact, install=not a.no_install, torch_key=a.torch_key
            ),
            1500,
        )
        print(r.get("stdout", ""), r.get("stderr", ""), f"exit {r.get('exit_code')}")
        return int(r.get("exit_code") or 0)
    elif a.cmd == "arms":
        print(
            b.start(
                recipes.arms_line(
                    a.out,
                    arms=a.arms,
                    rows=a.rows,
                    epochs=a.epochs,
                    seed=a.seed,
                    device=a.device,
                    battery=not a.no_battery,
                    n_variants=a.n_variants,
                    spe=a.spe,
                ),
                "arms",
            )
        )
    elif a.cmd == "doctor":
        print(
            b.start(
                recipes.doctor_line(
                    a.run_dir, a.out, encoder=a.encoder, lr=a.lr, rows=a.rows, epochs=a.epochs, device=a.device
                ),
                "doctor",
            )
        )
    elif a.cmd == "runs":
        b.put(str(SCRIPTS / "run_models.py"), "/content/run_models.py")
        print(b.start(recipes.runs_line(a.models, a.lr, a.lr_min, a.epochs, a.rows, device=a.device), "runs"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
