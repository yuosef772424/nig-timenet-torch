"""Report B, owner idea 2026-11-16: does a direction-balance penalty help the SINGLE-COIN model?

    python docs/audit/2026-11-16-panel/single_coin_balance_look.py ARTIFACT OUT [--seeds 0,1] [--weights 1,5]
        [--epochs 12] [--device auto]

The single-coin trainer as it is (one row = one coin window, shuffled batches, no timestamp channel, GRU, lr 3e-4,
selection ``val_raw_loss``), once without and once per weight with ``TrainSettings.direction_balance``. Every arm is
scored on ALL test rows with the day metrics and the direction-bias rows of ``eval/uncertainty_use.py`` (the arms of
``workflow.panel_study``, here on a flat split). Writes OUT/<arm>_s<seed>/ and OUT/balance_look.json / .md.
"""

import argparse
import json
import pathlib

from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.workflow import panel_study
from nig_timenet_torch.workflow.settings import (
    BatterySettings,
    EvalSettings,
    ModelSettings,
    RunSettings,
    TrainSettings,
    default_model_settings,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact"), p.add_argument("out")
    p.add_argument("--seeds", default="0,1"), p.add_argument("--weights", default="1,5")
    p.add_argument("--epochs", type=int, default=12), p.add_argument("--device", default="auto")
    a = p.parse_args()
    base = default_model_settings()
    settings = RunSettings(
        run_dir=str(pathlib.Path(a.out) / "ckpt"), data=LoaderSettings(name=a.artifact, local_dir="", mmap=True),
        model=ModelSettings(base.architecture, {**base.overrides, "encoder": "gru"}),
        train=TrainSettings(epochs=a.epochs, batch_size=256, lr=3e-4, device=a.device, steps_per_execution=16,
                            early_stop=EarlyStopSettings(patience=4)),
        evaluation=EvalSettings(battery=BatterySettings(enabled=False)))
    arms = {"single_coin": {}, **{f"single_coin_balance_{w}": {"train": {"direction_balance": float(w)}}
                                  for w in a.weights.split(",")}}
    rows = []
    for seed in (int(s) for s in a.seeds.split(",")):
        for name, arm in arms.items():
            row = panel_study.run_arm(settings, name, arm, seed, a.out)
            report = json.loads((pathlib.Path(a.out) / f"{name}_s{seed}" / "panel_report.json").read_text())
            row["direction_bias"] = report["uncertainty"]["direction_bias"]
            row["absolute_relative_sign"] = report["test"]["absolute_relative_sign"]["gross"]
            rows.append(row)
            b = row["direction_bias"]
            print(f"{name} seed {seed}: predicted up {b['share_pred_up']:.3f} (realised {b['share_real_up']:.3f}), "
                  f"per coin {b['per_coin_pred_up_min']:.2f} .. {b['per_coin_pred_up_max']:.2f}, metric 1 "
                  f"{row['absolute']['win_share'] * 100:.1f} %, metric 2 {row['neutral']['win_share'] * 100:.1f} %, "
                  f"rank IC close {row['test_rank_ic']['close']:+.4f}", flush=True)
    out = pathlib.Path(a.out)
    (out / "balance_look.json").write_text(json.dumps(rows, indent=1, default=float))
    lines = ["| arm | seed | best / epochs | predicted up (realised) | per-coin predicted-up range | one-sign "
             "timestamps | metric 1 winning days [CI] | mean day net | metric 2 winning days | rank IC close |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b, m1, m2 = r["direction_bias"], r["absolute"], r["neutral"]
        lines.append(f"| {r['arm']} | {r['seed']} | {r['best_epoch']} / {r['epochs_done']} | {b['share_pred_up']:.3f} "
                     f"({b['share_real_up']:.3f}) | {b['per_coin_pred_up_min']:.2f} .. {b['per_coin_pred_up_max']:.2f} | "
                     f"{b['one_sign_timestamps']:.2f} | {m1['win_share'] * 100:.1f} % [{m1['win_lo'] * 100:.1f}, "
                     f"{m1['win_hi'] * 100:.1f}] | {m1['mean_net'] * 100:+.4f} % | {m2['win_share'] * 100:.1f} % | "
                     f"{r['test_rank_ic']['close']:+.4f} |")
    (out / "balance_look.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
