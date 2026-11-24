"""Step 2 of tests/parity/make_chicks_golden.py (not a test): runs the OLD project's ``run_full_analysis``.

A standalone script: ``python run_old_chicks.py OLD_REPO WORLD.npz OUT_DIR`` (it imports nothing of this project except
``figsig.py`` next to it by path). The old evaluation modules are executed into one shared namespace exactly as the
old notebook does (``evaluation._loader``) with TensorFlow replaced by an empty stub module, and the one place that
calls the model (``_iter_model_batches``) is replaced by a fake that replays the arrays stored in the world file; the
old code runs unchanged on top of that. Writes into OUT_DIR: ``printed.txt`` (stdout), ``analysis_outputs/`` (every
file of the old out_dir except the PNGs), ``figures.json`` (name -> figure signature of each saved figure) and
``tables.json`` (the returned structures)."""

import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

TARGETS = ("high", "low", "close")


def _load_figsig():
    spec = importlib.util.spec_from_file_location("figsig", Path(__file__).with_name("figsig.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jsonable(v):
    import numpy as np
    import pandas as pd

    if isinstance(v, pd.DataFrame):
        return [{str(k): _jsonable(x) for k, x in row.items()} for row in v.to_dict("records")]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return v if isinstance(v, (bool, int)) else (bool(v) if isinstance(v, np.bool_) else int(v))
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if v is None or isinstance(v, str):
        return v
    return str(v)


def _split(z, stage: str, tf: str) -> dict:
    """{coin: split} in the old test_dict layout, coins in order of first appearance, rows in table order."""
    import numpy as np

    asset = z[f"{stage}_asset"]
    out = {}
    for a in dict.fromkeys(asset.tolist()):
        rows = np.flatnonzero(asset == a)
        lc = z[f"{stage}_lc"][rows]
        last_close = lc[:, 2]
        out[str(a)] = {f"X_{tf}": z[f"{stage}_X"][rows],
                       "base_params": np.stack([last_close, last_close], axis=1).astype("float32"),
                       "last_candles": lc, "y": {t: z[f"{stage}_reg_{t}"][rows] for t in TARGETS}}
    return out, {str(a): np.flatnonzero(asset == a) for a in dict.fromkeys(asset.tolist())}


def main(old_repo: str, npz: str, out: str) -> None:
    sys.path.insert(0, old_repo)
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    from evaluation import _loader

    figsig = _load_figsig()
    z = np.load(npz)
    tf, args = str(z["tf"]), json.loads(str(z["args"]))
    ns = {"__name__": "old_chicks", "np": np, "pd": pd}
    _loader.load_into(ns, exclude=("live", "legacy_uncertainty", "legacy_assets", "io_patterns"))

    test_dict, test_rows = _split(z, "test", tf)
    val_dict, val_rows = _split(z, "val", tf)
    pools = {}
    for stage, rows in (("test", test_rows), ("val", val_rows)):
        order = np.concatenate(list(rows.values()))
        outputs = {k[len(f"{stage}_out_"):]: z[k][order] for k in z.files if k.startswith(f"{stage}_out_")}
        first = np.concatenate([d[f"X_{tf}"] for d in (test_dict if stage == "test" else val_dict).values()])
        pools[stage] = (first[:3].tobytes(), outputs)

    def fake_iter(model, X, batch_size=0):
        x0 = np.asarray(X[0] if isinstance(X, (tuple, list)) else X)
        sig = x0[:3].tobytes()
        outputs = next(o for s, o in pools.values() if s == sig)
        for i in range(0, len(x0), batch_size or len(x0)):
            sl = slice(i, i + (batch_size or len(x0)))
            yield tuple(np.asarray(a)[sl] for a in X), {k: v[sl].reshape(-1, 1) for k, v in outputs.items()}

    ns["_iter_model_batches"] = fake_iter
    sigs = {}
    save_figure = ns["save_figure"]

    def recording_save_figure(fig, name, out_dir=ns["DEFAULT_OUTPUT_DIR"], dpi=130):
        sigs[name] = figsig.fig_signature(fig)
        return save_figure(fig, name, out_dir, dpi)

    ns["save_figure"] = recording_save_figure
    scale = float(z["scale"])
    specs = [dataclasses.replace(s, relative_to_entry=True, reg_scale=scale) for s in ns["DEFAULT_PRICE_TARGETS"]]
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = ns["run_full_analysis"](object(), test_dict, [tf], specs, out_dir="analysis_outputs",
                                      val_dict=val_dict, **args)
    printed = buf.getvalue().replace(str(out_dir / "analysis_outputs"), "<OUT>")
    (out_dir / "printed.txt").write_text(printed, encoding="utf-8")
    for p in (out_dir / "analysis_outputs").rglob("*.png"):
        p.unlink()
    (out_dir / "figures.json").write_text(json.dumps(sigs, indent=0, ensure_ascii=False), encoding="utf-8")
    keep = {k: res[k] for k in ("asset_summary", "asset_ci", "coverage", "verification_summary", "aggregated_results",
                                "movement_analysis", "confidence_analysis", "batch_analysis", "trade_selection",
                                "integrity_diagnostics", "trading_metrics")}
    keep["trade_candidates_n"] = None if res["trade_candidates"] is None else len(res["trade_candidates"])
    trust = res["trust_report"]
    keep["trust"] = {k: trust[k] for k in ("overall", "by_group", "reliability_curve", "calibration")}
    keep["calibrators"] = {t: {k: v for k, v in c.items() if k != "predict"}
                           for t, c in (res["calibrators"] or {}).items()}
    pat = res["pattern_discovery"]
    keep["patterns"] = {"summary": pat["summary"], "heads": pat["heads"], "skipped": pat["skipped"],
                        "per_head": {h: {k: r[k] for k in ("n", "features", "tree_rules", "tree_accuracy",
                                                            "tree_naive_baseline_accuracy", "tree_balanced_accuracy",
                                                            "feature_importance", "clusters")}
                                     for h, r in pat["per_head"].items()}}
    keep["flat_columns"] = list(res["flat_df"].columns)
    (out_dir / "tables.json").write_text(json.dumps(_jsonable(keep), indent=0, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
