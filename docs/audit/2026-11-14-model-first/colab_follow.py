"""Follow, resume and read the multi-coin (panel) arms on a Colab runtime, without the bridge.

    python docs/audit/2026-11-14-model-first/colab_follow.py status     # what is running, epochs done per arm
    python docs/audit/2026-11-14-model-first/colab_follow.py resume     # start every arm that is neither finished
                                                                        # nor running (it continues from its
                                                                        # checkpoint), in the background
    python docs/audit/2026-11-14-model-first/colab_follow.py results    # the test table of the finished arms, the
                                                                        # battery verdict lines, a copy on Drive

The arms are the ones of TRANSFORMER_HEAD_RESULTS.md section 9 (top 100 coins of the Drive artifact, whole timestamps,
selection by the cross-sectional rank IC). A finished arm has OUT/arms_<name>_s<seed>.json; an unfinished one keeps
its checkpoint in OUT/<name>_s<seed>/ckpt and ``quick_arms.py`` resumes from it when started again.
"""

import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path("/content/nig-timenet-torch")
ARTIFACT = "/content/art1"
OUT = pathlib.Path("/content/out_panel")
DRIVE = pathlib.Path("/content/drive/MyDrive/crypto/nig_timenet_results/panel_top100")
COMMON = ["--rows", "400000", "--epochs", "25", "--patience", "6", "--spe", "32", "--cross-section", "--top-coins",
          "100", "--selection", "val_rank_ic_reg", "--battery", "--n-variants", "3", "--arms", "gru:3e-4", "--device",
          "cuda"]
#: tag -> model overrides: B = per-coin model on whole-timestamp batches, R = + relative inputs, RE = + mean block
ARMS = {"_pB": {}, "_pR": {"cross_input": "relative"},
        "_pRE": {"cross_input": "relative", "cross_coin_layers": 1, "cross_coin_kind": "mean"}}
SEEDS = (0, 1)


def name(tag: str, seed: int) -> str:
    return f"gru_lr3e-4{tag}_s{seed}"


def running() -> str:
    return subprocess.run(["pgrep", "-af", "quick_arms.py"], capture_output=True, text=True).stdout


def epochs_done(tag: str, seed: int) -> str:
    log = OUT / f"follow_{name(tag, seed)}.log"
    summ = OUT / name(tag, seed) / "baseline_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        return f"finished: {s['epochs_done']} epochs, best epoch {s['best_epoch']}"
    if log.exists():
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("epoch")]
        return lines[-1][:90] if lines else "started, no epoch yet"
    return "no summary yet (started through the bridge: its log is not on disk)"


def status() -> None:
    procs = running()
    for tag in ARMS:
        for seed in SEEDS:
            alive = f"--seed {seed} " in procs and f"--tag {tag}" in "".join(
                ln for ln in procs.splitlines() if f"--seed {seed} " in ln)
            print(f"{name(tag, seed):28s} {'RUNNING ' if alive else '        '} {epochs_done(tag, seed)}")


def resume() -> None:
    procs = running()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, model in ARMS.items():
        for seed in SEEDS:
            if (OUT / f"arms_gru_lr3e-4{tag}_s{seed}.json").exists():
                continue
            if any(f"--seed {seed} " in ln and f"--tag {tag} " in ln + " " for ln in procs.splitlines()):
                continue
            cmd = [sys.executable, "-u", "docs/audit/2026-11-14-model-first/quick_arms.py", ARTIFACT, str(OUT),
                   "--seed", str(seed), "--model", json.dumps(model), "--tag", tag, *COMMON]
            log = open(OUT / f"follow_{name(tag, seed)}.log", "a")
            subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print("started", name(tag, seed))
    status()


def results() -> None:
    print(f"{'arm':26s} {'best val rank IC':>16s} {'epoch':>9s}  {'TEST rank IC high / low / close':32s} close AUC")
    for tag in ARMS:
        for seed in SEEDS:
            f = OUT / f"arms_gru_lr3e-4{tag}_s{seed}.json"
            if not f.exists():
                print(f"{name(tag, seed):26s} not finished")
                continue
            r = json.loads(f.read_text())
            ic = " / ".join(f"{r[f'targets.{t}.skill.rank_ic_by_timestamp']:.3f}" for t in ("high", "low", "close"))
            ep = f"{r['best_epoch']}/{r['epochs_done']}"
            print(f"{name(tag, seed):26s} {r['best_value']:16.4f} {ep:>9s}  {ic:32s} "
                  f"{r['targets.close.skill.class.auc']:.3f}")
    for tag in ARMS:
        for seed in SEEDS:
            rep = OUT / name(tag, seed) / "battery_report.txt"
            if rep.exists():
                print(f"\n===== battery, {name(tag, seed)} (TEST rows, cost 0.08 % per side) =====")
                text = rep.read_text(errors="replace")
                print(text[-2500:])
    if DRIVE.parent.parent.exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        for f in list(OUT.glob("arms_*.json")) + list(OUT.glob("*/baseline_summary.json")) + list(
                OUT.glob("*/test_report.json")) + list(OUT.glob("*/battery_report.txt")) + list(
                OUT.glob("*/battery_profit.json")):
            dest = DRIVE / f.relative_to(OUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        print(f"\ncopied the reports to {DRIVE}")


if __name__ == "__main__":
    {"status": status, "resume": resume, "results": results}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
