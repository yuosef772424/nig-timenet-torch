"""Step 2 of tests/parity/make_verification_golden.py (not a test): runs the OLD project's run_full_verification.

A standalone script (it imports nothing of this project): ``python run_old_verification.py OLD_REPO WORLD.npz OUT_DIR``.
The old workflow modules are executed into one shared namespace exactly as the old notebook does (workflow._loader),
with a fake model whose ``predict`` returns the arrays stored in the world file (TensorFlow is not needed)."""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

TARGETS = ("high", "low", "close")


def main(old_repo: str, npz: str, out: str) -> None:
    sys.path.insert(0, old_repo)
    import numpy as np
    import pandas as pd
    from core.schema import LAST_COLUMNS
    from workflow import _loader

    z = np.load(npz)
    args = json.loads(str(z["args"]))
    tf = str(z["tf"])
    ns = {"__name__": "old_parity", "np": np, "pd": pd, "LAST_COLUMNS": LAST_COLUMNS,
          "CONFIG": {"forecast_horizon": int(z["horizon"])}}
    _loader.load_into(ns, only=("splits", "selective_eval", "candle_baseline", "verification"))

    def split(stage):
        y = {}
        for t in TARGETS:
            y[f"y_{t}_reg"] = z[f"{stage}_reg_{t}"]
            y[f"y_{t}_class"] = z[f"{stage}_cls_{t}"]
        return {f"X_{tf}": z[f"{stage}_X"], "y": y, "last_candles": z[f"{stage}_lc"],
                "reg_target_scale": float(z["scale"])}

    class FakeModel:
        def __init__(self, outputs):
            self.outputs = outputs     # {id(X): {key: array}}

        def predict(self, X, batch_size=None, verbose=0):
            return {k: v.reshape(-1, 1) for k, v in self.outputs[id(X)].items()}

    train, val, test = split("train"), split("val"), split("test")
    outputs = {id(split_[f"X_{tf}"]): {k[len(f"{name}_out_"):]: z[k] for k in z.files if k.startswith(f"{name}_out_")}
               for name, split_ in (("val", val), ("test", test))}
    buf = io.StringIO()
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)                  # the old code prints the folder it saved to: keep that line relative
    with contextlib.redirect_stdout(buf):
        ver = ns["run_full_verification"](FakeModel(outputs), train, val, test, tf, price_targets=TARGETS,
                                          out_dir="old_out", verbose=True, **args)
    ver["summary"].to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    (out_dir / "printed.txt").write_text(buf.getvalue(), encoding="utf-8")
    for name in ("selective", "trades"):
        ver["details"][name].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    cb = ver["details"].get("candle")
    if cb:
        cb["table"].to_csv(out_dir / "candle_table.csv", index=False, encoding="utf-8")
        verdicts = {t: {"best": v["best_baseline"], "diff": [float(x) for x in v["auc_diff_vs_best"]],
                        "gain": [float(x) for x in v["auc_gain_when_combined"]], "verdict": v["verdict"],
                        "days": v["n_test_days"]} for t, v in cb["verdicts"].items()}
        (out_dir / "candle_verdicts.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
