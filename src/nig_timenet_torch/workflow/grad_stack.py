"""PURPOSE: Gradient stacking experiment (owner request 2026-11-10, quick, outside the roadmap): model A on the
    earlier half of train, its per-layer gradients feed model B on the later half; all variants through the test.
TAGS: grad stack, gradient features, stacking, time split halves, per-layer gradients, experiment
PITFALLS: grad_target "loss" uses the row's label, so it LEAKS on val / test (upper bound only); the gradient
    features carry model A's knowledge of H1, so compare B with "full" (all train rows), not only with "data_h2".


1. Split the TRAIN rows by time into two halves (H1 = earlier, H2 = later; H1 rows whose label ends after H2 starts
   are purged).
2. Model A is trained on H1 only. It then produces, for every row of H2 / val / test, its predictions and the
   per-sample gradient at EVERY learning layer (the gradient w.r.t. that layer's output activation, plus the input
   gradient), summarised per layer as [l2 norm, mean, std, max |g|].
3. Model B is trained on H2 with model A's gradient features as input:
     grad       = gradients only                  (x_B = gradient features, repeated over the window)
     grad_data  = gradients + the original window (x_B = concat(x, gradient features) on the feature axis)
   Controls on the same rows: data_h2 (B's rows, data only) and full (all train rows, data only).
4. Every variant goes through the project's own test: ``run.stage_report`` (target_report on the test split) and
   the gate rows ``eval.gates.stage_rows`` (gates B / C / canary / rule D; on E1 judged against the oracle).

Gradient target (``--grad-target``):
  output (default) = d(sum of model A's outputs)/d(layer): label-free, usable at prediction time, no leakage;
  loss             = d(model A's loss)/d(layer): needs the label of the row, so on val / test it LEAKS the label
                     (kept only to show the upper bound; never read it as a usable result).

Run: ``run_grad_stack(GradStackConfig(...), out)``; job entry ``jobs/grad_stack.py``; CLI
``python -m nig_timenet_torch.workflow.grad_stack --out <dir> --config '{"data": "<artifact dir>", "device": "cuda"}'``.
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, oracle_keys, write_synthetic
from nig_timenet_torch.eval import gates
from nig_timenet_torch.eval.vol_baseline import window_ranges
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.workflow import flat_run, run, run_common
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings

STATS = ("norm", "mean", "std", "maxabs")
VARIANTS = ("A", "data_h2", "grad", "grad_data", "full")


# ------------------------------------------------------------------------------------------------ split
def time_halves(ds: dict, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H1, H2) of the train rows by feature-end time; H1 rows whose label ends after H2's first row are purged."""
    _, fe, le = sample_times(ds)
    t = fe[train_idx]
    order = np.argsort(t, kind="stable")
    cut = t[order[len(order) // 2]]
    h2 = np.sort(train_idx[t >= cut])
    h1 = np.sort(train_idx[(t < cut) & (le[train_idx] <= cut)])
    return h1, h2


# ------------------------------------------------------------------------------------------------ gradients
def _layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """The learning layers: leaf modules that own trainable parameters."""
    return [(n, m) for n, m in model.named_modules()
            if not list(m.children()) and any(p.requires_grad for p in m.parameters(recurse=False))]


def _summ(g: torch.Tensor) -> torch.Tensor:
    g = g.reshape(g.shape[0], -1)
    return torch.stack([g.norm(dim=1), g.mean(dim=1), g.std(dim=1, unbiased=False), g.abs().amax(dim=1)], dim=1)


def grad_features(trainer, x: torch.Tensor, y: dict, *, target: str, batch: int, device) -> tuple[np.ndarray,
                                                                                                    list[str],
                                                                                                    np.ndarray]:
    """(features (n, G), names, predictions (n, P)) of model A on the rows ``x``. Eval mode: dropout off, batch norm
    on its moving statistics, so every row's gradient is its own (no cross-row term) and the batch-sum backward
    gives the exact per-row activation gradients."""
    model = trainer.model
    model.eval()
    layers = _layers(model)
    acts: dict[str, torch.Tensor] = {}

    def hook(name):
        def f(_m, _i, out):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                acts[name] = out
        return f

    handles = [m.register_forward_hook(hook(n)) for n, m in layers]
    feats, preds, names = [], [], None
    try:
        for s in range(0, len(x), batch):
            acts.clear()
            # autograd.grad (no .backward, no optimizer state touched); cudnn off: an RNN backward in eval mode
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                xb = x[s:s + batch].to(device).requires_grad_(True)
                out = model(xb)
                if target == "output":
                    scalar = sum(v.sum() for v in out.values())
                else:   # the loss is a batch MEAN: x n gives each row its own gradient (up to batch-level terms)
                    yb = {k: v[s:s + batch].to(device) for k, v in y.items()}
                    scalar = trainer.compute_losses(None, yb, outputs=out)[1] * xb.shape[0]
                nm = ["input"] + [n for n, _ in layers if n in acts]
                grads = torch.autograd.grad(scalar, [xb] + [acts[n] for n in nm[1:]], allow_unused=True)
            keep = [(n, g) for n, g in zip(nm, grads, strict=True) if g is not None]
            names = names or [f"{n}.{st}" for n, _ in keep for st in STATS]
            feats.append(torch.cat([_summ(g) for _, g in keep], dim=1).detach().cpu())
            preds.append(torch.cat([v.detach().reshape(v.shape[0], -1).cpu() for v in out.values()], dim=1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats).numpy(), names, torch.cat(preds).numpy()


class FeatureScaler:
    """signed log1p then z-score, fitted on model B's train rows only."""

    def fit(self, f: np.ndarray) -> "FeatureScaler":
        g = np.sign(f) * np.log1p(np.abs(f))
        self.mu, self.sd = g.mean(0), g.std(0) + 1e-8
        return self

    def transform(self, f: np.ndarray) -> np.ndarray:
        return ((np.sign(f) * np.log1p(np.abs(f)) - self.mu) / self.sd).astype(np.float32)


def assemble(x: torch.Tensor, g: np.ndarray, mode: str) -> torch.Tensor:
    gt = torch.as_tensor(g).unsqueeze(1).expand(-1, x.shape[1], -1)
    return gt.contiguous() if mode == "grad" else torch.cat([x, gt], dim=2)


# ------------------------------------------------------------------------------------------------ one variant
def fit_variant(base: run.Run, name: str, train_idx: np.ndarray, x: dict, y: dict, root: Path) -> run.Run:
    """Train one model through the project's run steps on the given rows / inputs (val is the common selection
    split, test is never used for training or selection)."""
    rd = root / name
    shutil.rmtree(rd, ignore_errors=True)     # trainer.fit resumes a folder that holds a checkpoint
    r = replace(base, settings=replace(base.settings, run_dir=str(rd)), idx={**base.idx, "train": train_idx},
                x=dict(x), y=dict(y), ms=None, trainer=None, fit_settings=None, ident=None, result=None, extra={})
    t0 = time.monotonic()
    run.train(run.build(r))
    print(f"[{name}] trained on {len(train_idx)} rows, F={x['train'].shape[2]}, best epoch {r.result.best_epoch}, "
          f"{time.monotonic() - t0:.0f} s", flush=True)
    return r


def evaluate(r: run.Run, name: str, x_raw_test: torch.Tensor, oracle: bool) -> tuple[dict, list[dict]]:
    """The project's test: target_report on the test split + the gate rows (gates.stage_rows)."""
    ds = r.dataset
    table = r.table("test")
    rep = run.stage_report(r, "test", table)
    idx, lc, codec = r.idx["test"], np.asarray(ds["last_candles"]), run_common.codec_state(ds)
    fo = list(ds["feature_order"])
    xn = x_raw_test.numpy()
    try:
        ranges = window_ranges(xn, fo)
    except (ValueError, KeyError):
        ranges = None
    orc = ({t: (np.asarray(ds[oracle_keys(t)[0]])[idx], np.asarray(ds[oracle_keys(t)[1]])[idx]) for t in r.targets}
           if oracle else None)
    st = gates.GateStage(env="E1" if oracle else "E3", seed=r.settings.train.seed, table=table,
                         labels={k: v.numpy() for k, v in r.y["test"].items()}, report=rep,
                         anchors=flat_run.anchors_by_name(lc, idx, r.targets, codec),
                         last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec, ranges=ranges,
                         last_move=xn[:, -1, fo.index("close")] if "close" in fo else None, oracle=orc)
    rows = gates.stage_rows(st)
    for row in rows:
        row["variant"] = name
    return rep, rows


def _flat(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


# ------------------------------------------------------------------------------------------------ main
def model_a_features(model_a: run.Run, base: run.Run, h2_xy: tuple, h2: np.ndarray, a: "GradStackConfig",
                     out: Path) -> tuple[dict, list[str]]:
    """Model A's gradient features (+ predictions if asked) on H2 / val / test, scaled on H2; raw values saved to
    ``modelA_grads_preds.npz``. -> ({"train" | "val" | "test": (n, G) float32}, names)."""
    t0 = time.monotonic()
    G, P = {}, {}
    for k, (xx, yy) in {"train": h2_xy, "val": (base.x["val"], base.y["val"]),
                        "test": (base.x["test"], base.y["test"])}.items():
        G[k], names, P[k] = grad_features(model_a.trainer, xx, yy, target=a.grad_target, batch=1024,
                                          device=base.device)
    if a.with_preds:
        names = names + [f"predA.{i}" for i in range(P["train"].shape[1])]
        G = {k: np.concatenate([G[k], P[k]], axis=1) for k in G}
    sc = FeatureScaler().fit(G["train"])
    print(f"gradient features: {len(names)} columns, {time.monotonic() - t0:.0f} s", flush=True)
    idx = {"train": h2, "val": base.idx["val"], "test": base.idx["test"]}
    np.savez_compressed(out / "modelA_grads_preds.npz", names=np.array(names),
                        **{f"grad_{k}": v for k, v in G.items()}, **{f"pred_{k}": v for k, v in P.items()},
                        **{f"idx_{k}": v for k, v in idx.items()})
    return {k: sc.transform(v) for k, v in G.items()}, names


def write_report(results: dict, rows: list[dict], out: Path, meta: dict) -> dict:
    """The comparison files (test report per variant, gate rows / verdicts, summary.json) and a printed digest."""
    summary = pd.DataFrame({n: _flat(r) for n, r in results.items()})
    summary.to_csv(out / "test_report_by_variant.csv")
    gdf = pd.DataFrame(rows)
    gdf.to_csv(out / "gate_rows.csv", index=False)
    blocking = gdf[gdf["blocking"].astype(bool)]
    verdicts = blocking.groupby(["variant", "verdict"]).size().unstack(fill_value=0).reindex(list(results))
    verdicts.to_csv(out / "gate_verdicts.csv")
    failed = blocking[blocking.verdict == "fail"]
    meta["gate_fails"] = {v: sorted(f"{r.gate}|{r.check}|{r.head}" for r in failed[failed.variant == v].itertuples())
                          for v in results}
    (out / "summary.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    key_rows = [k for k in summary.index if k.endswith(("rank_ic_by_timestamp", "spearman_per_coin.mean",
                                                         "class.auc", "decile_error_spearman", "sharp.coverage"))]
    print("\n== gate verdicts (blocking rows) ==\n", verdicts)
    print("\n== key test metrics ==\n", summary.loc[key_rows].round(4).to_string())
    print("\nfailed blocking checks per variant:", {v: len(f) for v, f in meta["gate_fails"].items()},
          "(list: summary.json)")
    return meta


@dataclass(frozen=True)
class GradStackConfig:
    """Every setting of the experiment (strict: an unknown key raises in ``from_dict``)."""
    data: str = ""                      # artifact dir; "" = build synthetic E1 (with the oracle) under the out folder
    e1: dict = field(default_factory=lambda: {"n_coins": 32, "n_bars": 2000})   # SyntheticSettings fields
    architecture: str = "tiny_mlp"      # ARCHITECTURES key; "" = the project default model
    epochs: int = 5
    batch_size: int = 256
    seed: int = 0
    device: str = "cpu"                 # "cpu" / "cuda" / "auto"
    max_rows: int | None = None         # smoke stride of the split (None = all rows)
    grad_target: str = "output"         # "output" (label-free) or "loss" (LEAKS the label on val / test)
    with_preds: bool = False            # add model A's raw outputs to the gradient features
    variants: tuple = VARIANTS

    def __post_init__(self):
        object.__setattr__(self, "variants", tuple(self.variants))
        if self.grad_target not in ("output", "loss"):
            raise ValueError(f"grad_target must be 'output' or 'loss', got {self.grad_target!r}")
        bad = sorted(set(self.variants) - set(VARIANTS))
        if bad or "A" not in self.variants:
            raise ValueError(f"variants must include 'A' and come from {VARIANTS}, got unknown {bad}")
        SyntheticSettings(**self.e1)


def run_grad_stack(cfg: GradStackConfig, out) -> dict:
    """The whole experiment (module docstring); writes its files to ``out`` and returns the summary dict."""
    a = cfg
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.monotonic()

    data = a.data
    if not data:
        data = str(out / "E1")
        write_synthetic(data, SyntheticSettings(**a.e1), source_commit="0" * 40, overwrite=True)
    s = RunSettings(run_dir=str(out / "runs"), code_sha="grad_stack", data=LoaderSettings(name=data, local_dir="",
                                                                                         mmap=False),
                    split=SplitSettings(max_rows=a.max_rows),
                    train=TrainSettings(epochs=a.epochs, batch_size=a.batch_size, seed=a.seed, device=a.device))
    if a.architecture:
        s = replace(s, model=ModelSettings(a.architecture))
    base = run.prepare(s)
    ds = base.dataset
    oracle = oracle_keys(base.targets[0])[0] in ds
    for st in ("test",):
        base.x[st], base.y[st] = flat_run.flat_inputs(ds, base.idx[st])
    h1, h2 = time_halves(ds, base.idx["train"])
    _, fe, _ = sample_times(ds)
    split_info = {k: {"rows": int(len(v)), "from": str(pd.Timestamp(fe[v].min())), "to": str(pd.Timestamp(fe[v].max()))}
                  for k, v in {"H1": h1, "H2": h2, "val": base.idx["val"], "test": base.idx["test"]}.items()}
    print("split:", json.dumps(split_info), flush=True)
    xs = {k: flat_run.flat_inputs(ds, v) for k, v in {"H1": h1, "H2": h2}.items()}
    want = list(a.variants)
    root = out / "runs"
    results, rows, models = {}, [], {}

    def common(xtr, ytr, xva=None, xte=None):
        return ({"train": xtr, "val": base.x["val"] if xva is None else xva,
                 "test": base.x["test"] if xte is None else xte},
                {"train": ytr, "val": base.y["val"], "test": base.y["test"]})

    # model A on H1
    x_, y_ = common(*xs["H1"])
    models["A"] = fit_variant(base, "A", h1, x_, y_, root)
    Gs, names = model_a_features(models["A"], base, xs["H2"], h2, a, out)

    if "data_h2" in want:
        x_, y_ = common(*xs["H2"])
        models["data_h2"] = fit_variant(base, "data_h2", h2, x_, y_, root)
    for mode in ("grad", "grad_data"):
        if mode in want:
            x_, y_ = common(assemble(xs["H2"][0], Gs["train"], mode), xs["H2"][1],
                            assemble(base.x["val"], Gs["val"], mode), assemble(base.x["test"], Gs["test"], mode))
            models[mode] = fit_variant(base, mode, h2, x_, y_, root)
    if "full" in want:
        x_, y_ = common(base.x["train"], base.y["train"])
        models["full"] = fit_variant(base, "full", base.idx["train"], x_, y_, root)

    for name in [v for v in want if v in models]:
        rep, rr = evaluate(models[name], name, base.x["test"], oracle)
        results[name] = rep
        rows += rr

    meta = {"config": asdict(a), "split": split_info, "grad_features": names, "oracle": bool(oracle),
            "seconds": round(time.monotonic() - t_all, 1)}
    return write_report(results, rows, out, meta)


def main(argv=None) -> dict:
    """CLI: ``python -m nig_timenet_torch.workflow.grad_stack --out <dir> [--config '<json of GradStackConfig>']``."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="{}", help="JSON of GradStackConfig fields")
    a = p.parse_args(argv)
    return run_grad_stack(from_dict(GradStackConfig, json.loads(a.config)), a.out)


if __name__ == "__main__":
    main()
