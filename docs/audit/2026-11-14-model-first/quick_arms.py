"""Small-sample arms on a contract artifact: encoder (gru | transformer) x learning rate, same rows and seed.

    python docs/audit/2026-11-14-model-first/quick_arms.py ARTIFACT OUT [--rows 20000] [--epochs 6] [--seed 0]
        [--arms gru:3e-4,transformer:3e-4] [--device auto]

One RunSettings per arm (default model = preset + MODEL_FIRST_FIXES, encoder and lr changed); prints the epoch lines
and the test scalars, writes OUT/<arm>/{baseline_summary,test_report}.json and OUT/arms.json (one row per arm).
"""

import argparse
import hashlib
import json
import pathlib
import time

from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.flat_run import epoch_line
from nig_timenet_torch.workflow.settings import (
    BatterySettings,
    EvalSettings,
    ModelSettings,
    RunSettings,
    SplitSettings,
    TrainSettings,
    default_model_settings,
)


def arm_settings(a, encoder: str, lr: float, out: pathlib.Path, head: str = "dense") -> RunSettings:
    base = default_model_settings()
    model = ModelSettings(
        base.architecture, {**base.overrides, "encoder": encoder, **({"nig_head": head} if head != "dense" else {}),
                            **json.loads(a.model)}
    )
    return RunSettings(
        run_dir=str(out / "ckpt"),
        data=LoaderSettings(name=a.artifact, mmap=True),
        split=SplitSettings(max_rows=a.rows),
        model=model,
        train=TrainSettings(
            epochs=a.epochs, batch_size=a.batch_size, seed=a.seed, lr=lr, device=a.device, steps_per_execution=a.spe,
            early_stop=EarlyStopSettings(patience=a.patience),
        ),
        evaluation=EvalSettings(
            health_max_rows=2000, battery=BatterySettings(enabled=a.battery, n_variants=a.n_variants)
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact"), p.add_argument("out")
    p.add_argument("--rows", type=int, default=20000), p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default="gru:3e-4,transformer:3e-4"), p.add_argument("--device", default="auto")
    p.add_argument("--battery", action="store_true", help="run the evaluation battery (val + test, T-profit)")
    p.add_argument("--spe", type=int, default=1, help="steps_per_execution (speed only; weights must not change)")
    p.add_argument("--n-variants", type=int, default=1, help="variants tried so far (widens the battery CIs)")
    p.add_argument("--patience", type=int, default=15, help="early-stop patience in epochs (the run default is 15)")
    p.add_argument("--model", default="{}", help="JSON of extra model overrides for every arm")
    p.add_argument("--tag", default="", help="suffix of the arm folders (to keep --model variants apart)")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    rows = []
    for spec in a.arms.split(","):
        enc, lr, *rest = spec.split(":")  # encoder:lr[:nig_head]  (nig_head: dense | scalar_joint | scalar_factorized)
        head = rest[0] if rest else "dense"
        name = f"{enc}_lr{lr}" + ("" if head == "dense" else f"_{head}") + a.tag
        folder = out / f"{name}_s{a.seed}"
        t0 = time.time()
        print(f"\n===== arm {name} =====", flush=True)
        r = steps.build(steps.prepare(arm_settings(a, enc, float(lr), folder, head), folder))
        steps.train(r, on_epoch=lambda row: print(epoch_line(row, a.epochs), flush=True))
        rep = steps.test_step(r)
        if a.battery:
            steps.battery_step(r)
        summ = steps.write_summary(r)
        flat = dict(steps._flat(rep))
        keep = {
            k: v
            for k, v in flat.items()
            if "per_coin" not in k
            and ("rank_ic" in k or "auc" in k or k.endswith("mae_skill") or "skill" in k and "nll" in k)
        }
        h = hashlib.sha256()  # fingerprint of the weights the run ended with (live model)
        for k, v in sorted(r.trainer.model.state_dict().items()):
            h.update(k.encode() + v.detach().cpu().contiguous().numpy().tobytes())
        rows.append(
            {
                "arm": name,
                "weights_sha256": h.hexdigest()[:16],
                "best_epoch": summ["best_epoch"],
                "best_value": summ["best_value"],
                "epochs_done": summ["epochs_done"],
                "stopped_early": summ["stopped_early"],
                "seconds": round(time.time() - t0, 1),
                **keep,
            }
        )
        print("\n".join(steps.test_lines(rep)), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"arms_{name}_s{a.seed}.json").write_text(json.dumps(rows[-1], indent=1, default=float))
        (out / "arms.json").write_text(json.dumps(rows, indent=1, default=float))
    print("\nARMS:", json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
