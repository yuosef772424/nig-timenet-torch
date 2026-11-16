"""CPU smoke runner of ROADMAP ST1 (staged study) and F2 (rolling study) on the synthetic environment E1 (rule C5: a
smoke check of the machinery, not a result).

    python docs/results/cpu_smoke_st1_f2.py WORK_DIR staged '<StageSettings JSON>'
    python docs/results/cpu_smoke_st1_f2.py WORK_DIR rolling '<RollingSettings JSON>'

E1 = 8 coins x 3200 bars (4h, 12,800 rows), a small transformer (d_model 32, 1 layer, no feed-forward), batch 128,
lr 1e-3, two seeds for the staged study (staged arm + B1 / B2 / B3 at equal steps); the rolling study takes its seeds
from the JSON.
"""

import json
import pathlib
import sys
import time

from nig_timenet_torch.core.config import from_dict, updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, write_synthetic
from nig_timenet_torch.trainer.staged import StageSettings
from nig_timenet_torch.workflow.rolling_run import run_rolling
from nig_timenet_torch.workflow.settings import ModelSettings, RollingSettings, RunSettings, default_model_settings
from nig_timenet_torch.workflow.staged_study import run_staged_study

root, which = pathlib.Path(sys.argv[1]), sys.argv[2]
art = root / "e1"
if not art.exists():
    write_synthetic(art, SyntheticSettings(n_coins=8, n_bars=3200, seed=0), source_commit="0" * 40)
base = default_model_settings()
small = {"encoder": "transformer", "ffn": "none", "d_model": 32, "num_layers": 1, "num_heads": 4, "num_kv_heads": 2,
         "head_hidden": 32, "class_head_hidden": 16}
s = RunSettings(run_dir=str(root / "run"), data=LoaderSettings(name=str(art)),
                model=ModelSettings(base.architecture, {**base.overrides, **small}))
s = updated(s, {"train.device": "cpu", "train.batch_size": 128, "train.lr": 1e-3, "train.epochs": 3,
                "evaluation.battery.enabled": False})
t0 = time.time()
if which == "staged":
    study = run_staged_study(updated(s, {"train.staged": from_dict(StageSettings, json.loads(sys.argv[3]))}), (0, 1),
                             root / "staged")
    print(study["verdict"])
else:
    run_rolling(s, RollingSettings(**json.loads(sys.argv[3])), root / "rolling")
    print((root / "rolling" / "rolling_summary.md").read_text())
print("seconds", round(time.time() - t0))
