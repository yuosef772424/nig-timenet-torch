"""PURPOSE: Shared pieces of the model doctor: the Check row every instrument returns, verdict helpers, the report
    table, the DoctorContext (one prepared / built / trained run through the workflow steps), the purged
    time-ordered probe (ridge with the penalty chosen on an inner split) and the simple-feature baseline.
TAGS: doctor, check row, readiness table, probe, ridge probe, purged folds, simple baseline, har features, context
PITFALLS: a probe score is OUT OF FOLD on time-ordered folds with an embargo between train and test rows (rows are
    coin-major in an artifact, so contiguous ROW folds would split by coin, not by time); every number is reported
    next to a baseline, never alone; verdict "info" = a number to read, not judged.

A Check is one line of the readiness table: card, stage, check, value, baseline, verdict (pass / warn / fail / info)
and ``action`` = what to change when it fails (the card section says more). Instruments return ``list[Check]``.
"""

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.stats.metrics import spearman
from nig_timenet_torch.workflow import flat_run, run
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings, default_model_settings

__all__ = ["Check", "VERDICTS", "worst", "render_table", "write_report", "DoctorContext", "make_context",
           "fresh_trainer", "stride", "time_folds", "ridge_oof", "probe_scores", "simple_features", "reg_key",
           "predict_outputs", "row_times", "judge"]

VERDICTS = ("pass", "info", "warn", "fail")


@dataclass
class Check:
    card: str                     # "A" .. "D"
    stage: str                    # where in the pipeline: "model/stats", "trainer/optimizer", "data/split", ...
    check: str
    value: object                 # a number or a short string
    baseline: object = ""         # the trivial / reference value the number is read against
    verdict: str = "info"
    action: str = ""              # what to change when it fails
    detail: dict = field(default_factory=dict)   # extra numbers for the JSON report

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, not {self.verdict!r}")


def worst(checks) -> str:
    """The worst verdict of ``checks`` ("pass" when there is none to judge)."""
    order = {"info": 0, "pass": 0, "warn": 1, "fail": 2}
    v = max((c.verdict for c in checks), key=lambda s: order[s], default="pass")
    return "pass" if v == "info" else v


def judge(value: float, *, fail_below=None, warn_below=None, fail_above=None, warn_above=None) -> str:
    """pass / warn / fail of a number against one-sided bounds; a non-finite value fails."""
    if value is None or not math.isfinite(value):
        return "fail"
    if (fail_below is not None and value < fail_below) or (fail_above is not None and value > fail_above):
        return "fail"
    if (warn_below is not None and value < warn_below) or (warn_above is not None and value > warn_above):
        return "warn"
    return "pass"


def _cell(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.4g}" if (abs(v) >= 1e-3 or v == 0) and abs(v) < 1e5 else f"{v:.2e}"
    return str(v).replace("|", "/").replace("\n", " ")


def render_table(checks, *, only=None) -> str:
    """Markdown table of ``checks`` (``only``: a set of verdicts to keep)."""
    rows = [c for c in checks if only is None or c.verdict in only]
    lines = ["| card | stage | check | value | baseline | verdict | action |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {c.card} | {c.stage} | {c.check} | {_cell(c.value)} | {_cell(c.baseline)} | {c.verdict} | "
              f"{_cell(c.action)} |" for c in rows]
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.generic,)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return _jsonable(np.asarray(o).tolist())
    return o


def write_report(out, name: str, checks, header: str = "") -> None:
    """``<name>.json`` (every Check with its detail) and ``<name>.md`` (the table) in ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(_jsonable([asdict(c) for c in checks]), indent=1), encoding="utf-8")
    (out / f"{name}.md").write_text((header + "\n\n" if header else "") + render_table(checks) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- context
@dataclass
class DoctorContext:
    """One run the instruments examine: ``run`` is a workflow Run that was prepared, built and trained (or restored);
    ``quick`` trims the heavier checks; ``is_e1`` = the artifact carries the synthetic oracle columns."""
    run: run.Run
    out: Path
    quick: bool = True
    seed: int = 0
    notes: dict = field(default_factory=dict)

    @property
    def dataset(self) -> dict:
        return self.run.dataset

    @property
    def model(self) -> torch.nn.Module:
        return self.run.trainer.model

    @property
    def targets(self) -> tuple:
        return self.run.targets

    @property
    def is_e1(self) -> bool:
        return any(k.startswith("oracle_") for k in self.dataset)

    def rows(self, stage: str, max_rows: int | None = None) -> np.ndarray:
        """Artifact row indices of ``stage`` (evenly strided down to ``max_rows``)."""
        return stride(self.run.idx[stage], max_rows)

    def xy(self, stage: str, max_rows: int | None = None):
        """(artifact rows, x tensor, {label key: numpy}) of ``stage``."""
        idx = self.rows(stage, max_rows)
        x, y = flat_run.flat_inputs(self.dataset, idx)
        return idx, x, {k: v.numpy() for k, v in y.items()}


def stride(idx: np.ndarray, max_rows: int | None) -> np.ndarray:
    idx = np.asarray(idx)
    if max_rows is None or len(idx) <= max_rows:
        return idx
    return idx[np.unique(np.linspace(0, len(idx) - 1, max_rows).astype(np.int64))]


def reg_key(target: str) -> str:
    return head_spec("nig_regression", target).label_key


def row_times(dataset: dict, idx: np.ndarray) -> np.ndarray:
    return np.asarray(dataset["last_candles"])[np.asarray(idx), TS_COL].astype(np.int64)


def make_context(artifact: str, out, *, run_dir: str | None = None, model: dict | None = None,
                 train: dict | None = None, architecture: str | None = None, no_preset: bool = False,
                 max_rows: int | None = None, epochs: int = 3, batch_size: int = 512, seed: int = 0,
                 quick: bool = True, trained: bool = False) -> DoctorContext:
    """Prepare, build and train (or, with ``trained``, restore the finished ``run_dir``) the configuration: the
    default model and training settings with ``model`` / ``train`` overrides. ``max_rows`` strides every split."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = default_model_settings()
    arch = architecture or base.architecture
    keep = arch == base.architecture and not no_preset
    ms = ModelSettings(arch, {**(base.overrides if keep else {}), **(model or {})})
    tr = replace(TrainSettings(**{"device": "cpu", "batch_size": batch_size, "epochs": epochs, **(train or {})}),
                 seed=seed)
    settings = RunSettings(run_dir=str(run_dir or out / "run"), model=ms, train=tr,
                           split=SplitSettings(max_rows=max_rows),
                           data=LoaderSettings(name=str(artifact), mmap=True, local_dir=""))
    r = run.build(run.prepare(settings))
    if trained:
        run.restore(r)
    else:
        run.train(r)
    return DoctorContext(r, out, quick=quick, seed=seed)


def fresh_trainer(ctx: DoctorContext, seed: int | None = None, *, train: dict | None = None,
                  model: dict | None = None):
    """A NEW model + Trainer of the context's configuration (fitted scalers included), not trained; ``train`` /
    ``model`` override single settings for an experiment (e.g. ``{"lr": 1e-2}``)."""
    from nig_timenet_torch.core.seeds import seed_everything

    r = ctx.run
    s = r.settings
    if train:
        s = replace(s, train=replace(s.train, **train))
    ms = r.ms if not model else ModelSettings(r.ms.architecture, {**r.ms.overrides, **model})
    seed_everything(ctx.seed if seed is None else seed)
    trainer = flat_run.make_trainer(s, ms, r.targets, r.x["train"].shape[1], r.x["train"].shape[2], r.device)
    fit_scalers(trainer.model, r.x["train"], r.y["train"])
    return trainer


@torch.no_grad()
def predict_outputs(model, x: torch.Tensor, chunk: int = 4096, train_mode: bool = False) -> dict:
    """{output key: numpy (N,)} of ``model(x)`` in eval mode (``train_mode``: train mode, for the mode-gap check; the
    BatchNorm moving statistics are restored afterwards)."""
    was = model.training
    state = {k: v.clone() for k, v in model.state_dict().items()} if train_mode else None
    model.train(train_mode)
    dev = next(model.parameters()).device
    try:
        parts = [model(x[i:i + chunk].to(dev)) for i in range(0, len(x), chunk)]
    finally:
        if state is not None:
            model.load_state_dict(state)
        model.train(was)
    return {k: torch.cat([p[k] for p in parts]).float().cpu().numpy().reshape(len(x), -1).squeeze(-1)
            for k in parts[0]}


# ------------------------------------------------------------------------------------------------------ probes
def time_folds(times: np.ndarray, n_folds: int = 4, embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train rows, test rows)] : the rows sorted by ``times`` and cut into ``n_folds`` contiguous time blocks; a
    train row is dropped when its time lies within ``embargo`` (same unit as ``times``) of the test block."""
    times = np.asarray(times)
    order = np.argsort(times, kind="stable")
    out = []
    for block in np.array_split(order, n_folds):
        if not len(block):
            continue
        lo, hi = times[block].min(), times[block].max()
        mask = np.ones(len(times), dtype=bool)
        mask[block] = False
        mask &= (times < lo - embargo) | (times > hi + embargo)
        out.append((np.flatnonzero(mask), block))
    return out


def _ridge_path(a_tr, y_tr, a_te, lambdas):
    """Predictions on ``a_te`` for every penalty in ``lambdas`` (the standardised train block's Gram matrix is built
    once and shared by every column of ``y_tr`` (n, k)); returns [(n_te, k)] per penalty. Primal or dual system,
    whichever is smaller, one Cholesky solve per penalty, in torch on ONE thread: threaded LAPACK on a busy machine
    is 100x slower on these small systems (measured: 0.6 s vs 4 ms per 576 x 576 solve)."""
    mu, sd = a_tr.mean(0), a_tr.std(0)
    keep = sd > 1e-10 * max(float(sd.max()), 1e-300)
    y0 = y_tr.mean(0)
    if not keep.any():
        return [np.tile(y0, (len(a_te), 1)) for _ in lambdas]
    a = torch.from_numpy((a_tr[:, keep] - mu[keep]) / sd[keep])
    b = torch.from_numpy((a_te[:, keep] - mu[keep]) / sd[keep])
    yc = torch.from_numpy(y_tr - y0)
    n, d = a.shape
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if d <= n:
            g, r, eye = a.T @ a, a.T @ yc, torch.eye(d, dtype=a.dtype)
            return [(b @ torch.cholesky_solve(r, torch.linalg.cholesky(g + lam * eye))).numpy() + y0
                    for lam in lambdas]
        g, k, eye = a @ a.T, b @ a.T, torch.eye(n, dtype=a.dtype)
        return [(k @ torch.cholesky_solve(yc, torch.linalg.cholesky(g + lam * eye))).numpy() + y0 for lam in lambdas]
    finally:
        torch.set_num_threads(threads)


def ridge_oof(a: np.ndarray, y: np.ndarray, folds, lambdas=(1.0, 10.0, 100.0, 1000.0, 1e4)) -> np.ndarray:
    """Out-of-fold ridge predictions of ``y`` ((n,) or (n, k): k targets share the decomposition) from the features
    ``a``; per fold and target the penalty is the one with the smallest squared error on the LAST fifth (in the
    fold's train-row order) of the train rows."""
    a, y = np.asarray(a, dtype=np.float64), np.asarray(y, dtype=np.float64)
    one = y.ndim == 1
    y = y[:, None] if one else y
    pred = np.full(y.shape, np.nan)
    for tr, te in folds:
        if len(tr) < 20 or not len(te):
            continue
        cut = max(1, int(len(tr) * 0.8))
        inner = _ridge_path(a[tr[:cut]], y[tr[:cut]], a[tr[cut:]], lambdas)
        err = np.stack([((p - y[tr[cut:]]) ** 2).mean(0) for p in inner])          # (lambdas, k)
        best = err.argmin(0) if cut < len(tr) else np.ones(y.shape[1], dtype=int)
        full = _ridge_path(a[tr], y[tr], a[te], lambdas)
        pred[te] = np.stack([full[best[j]][:, j] for j in range(y.shape[1])], axis=1)
    return pred[:, 0] if one else pred


def probe_scores(a: np.ndarray, signals: dict, folds, *, squares: bool = True) -> dict:
    """{signal: {"ic": pooled Spearman, "r2": 1 - SSE / SST}} of the out-of-fold ridge probe of every signal in
    ``signals`` ({name: (n,)}) from the features ``a`` (``squares``: the features and their squares, so a level
    hidden in a variance is readable)."""
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    if squares:
        a = np.concatenate([a, a * a], axis=1)
    names = list(signals)
    y = np.stack([np.asarray(signals[k], dtype=np.float64) for k in names], axis=1)
    pred = ridge_oof(a, y, folds)
    out = {}
    for j, k in enumerate(names):
        ok = np.isfinite(pred[:, j]) & np.isfinite(y[:, j])
        if ok.sum() < 20 or np.var(y[ok, j]) == 0:
            out[k] = {"ic": float("nan"), "r2": float("nan"), "n": int(ok.sum())}
            continue
        sse, sst = np.sum((y[ok, j] - pred[ok, j]) ** 2), np.sum((y[ok, j] - y[ok, j].mean()) ** 2)
        out[k] = {"ic": float(spearman(y[ok, j], pred[ok, j])), "r2": float(1.0 - sse / sst), "n": int(ok.sum())}
    return out


def simple_features(x: np.ndarray, feature_order) -> tuple[np.ndarray, list[str]]:
    """The strong SIMPLE baseline's features from the raw windows ``x`` (N, T, F): per channel the last value and the
    mean / sd / mean-absolute over the last 1 / 6 / all bars; with ``high`` and ``low`` channels also the HAR range
    terms (mean bar range over the last 1 / 6 / all bars, and their logs). What a model must beat (rule D)."""
    x = np.asarray(x, dtype=np.float64)
    names, cols = [], []
    t = x.shape[1]
    lags = sorted({1, min(6, t), t})
    for j, ch in enumerate(feature_order):
        v = x[:, :, j]
        for lag in lags:
            cols += [v[:, -lag:].mean(axis=1), np.abs(v[:, -lag:]).mean(axis=1)]
            names += [f"{ch}_mean{lag}", f"{ch}_abs{lag}"]
        cols.append(v.std(axis=1))
        names.append(f"{ch}_sd")
    fo = list(feature_order)
    if "high" in fo and "low" in fo:
        rng = x[:, :, fo.index("high")] - x[:, :, fo.index("low")]
        for lag in lags:
            m = rng[:, -lag:].mean(axis=1)
            cols += [m, np.log(np.maximum(m, 1e-12))]
            names += [f"range{lag}", f"log_range{lag}"]
    return np.stack(cols, axis=1), names
