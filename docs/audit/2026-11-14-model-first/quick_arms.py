"""Small-sample arms on a contract artifact: encoder (gru | transformer) x learning rate, same rows and seed.

    python docs/audit/2026-11-14-model-first/quick_arms.py ARTIFACT OUT [--rows 20000] [--epochs 6] [--seed 0]
        [--arms gru:3e-4,transformer:3e-4] [--device auto]

One RunSettings per arm (default model = preset + MODEL_FIRST_FIXES, encoder and lr changed); prints the epoch lines
and the test scalars, writes OUT/<arm>/{baseline_summary,test_report}.json and OUT/arms.json (one row per arm).
"""
import argparse
import json
import pathlib
import time

from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.flat_run import epoch_line
from nig_timenet_torch.workflow.settings import (BatterySettings, EvalSettings, ModelSettings, RunSettings,
                                                 SplitSettings, TrainSettings, default_model_settings)


def arm_settings(a, encoder: str, lr: float, out: pathlib.Path) -> RunSettings:
    base = default_model_settings()
    model = ModelSettings(base.architecture, {**base.overrides, "encoder": encoder})
    return RunSettings(
        run_dir=str(out / "ckpt"), data=LoaderSettings(name=a.artifact, mmap=True),
        split=SplitSettings(max_rows=a.rows), model=model,
        train=TrainSettings(epochs=a.epochs, batch_size=a.batch_size, seed=a.seed, lr=lr, device=a.device),
        evaluation=EvalSettings(health_max_rows=2000, battery=BatterySettings(enabled=False)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact"), p.add_argument("out")
    p.add_argument("--rows", type=int, default=20000), p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default="gru:3e-4,transformer:3e-4"), p.add_argument("--device", default="auto")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    rows = []
    for spec in a.arms.split(","):
        enc, lr = spec.split(":")
        name, folder = f"{enc}_lr{lr}", out / f"{enc}_lr{lr}"
        t0 = time.time()
        print(f"\n===== arm {name} =====", flush=True)
        r = steps.build(steps.prepare(arm_settings(a, enc, float(lr), folder), folder))
        steps.train(r, on_epoch=lambda row: print(epoch_line(row, a.epochs), flush=True))
        rep = steps.test_step(r)
        summ = steps.write_summary(r)
        flat = dict(steps._flat(rep))
        keep = {k: v for k, v in flat.items() if "per_coin" not in k and ("rank_ic" in k or "auc" in k
                or k.endswith("mae_skill") or "skill" in k and "nll" in k)}
        rows.append({"arm": name, "best_epoch": summ["best_epoch"], "best_value": summ["best_value"],
                     "epochs_done": summ["epochs_done"], "seconds": round(time.time() - t0, 1), **keep})
        print("\n".join(steps.test_lines(rep)), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "arms.json").write_text(json.dumps(rows, indent=1, default=float))
    print("\nARMS:", json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
