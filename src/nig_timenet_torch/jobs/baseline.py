"""PURPOSE: Entry ``baseline``: trains one seed of the default NIG-TimeNet model on the contract artifact, then writes
    the summary, health report and test report (ROADMAP T5r).
TAGS: baseline job, colab entry, training run, test report, health report, resume, smoke job
PITFALLS: every data fact (price targets, codec mode and scale, timeframe, split dates) is read from the artifact, never
    assumed; price anchors and last_close are taken by column NAME (LAST_COLUMN_INDEX); a job config key not in
    BaselineConfig raises; max_rows (smoke mode) strides each split down, everything else is the same code path.

Entry ``baseline``: one seed of the default model on the contract artifact (ROADMAP T5r, job ``baseline-001``).

The default model is what TF ``main.ipynb`` builds (BUILD_PLAN.md V1): ``nig_timenet_v2`` with the whole
``ANTI_MEMORIZATION_CONFIG`` (GRU encoder + level passthrough), classification label smoothing 0.1. The price targets
(high, low, close for the TF artifact) are the artifact's ``price_targets``; a regression target is decoded with the
codec registered under the artifact's ``reg_target_mode`` (an unknown mode, or a missing ``reg_target_scale``, raises).

Config (``BaselineConfig``, strict: an unknown key raises): ``device`` (required, the job's device), ``seed``,
``epochs``, ``batch_size``, ``run_dir`` (required; a Drive folder: the checkpoint lives there, so a cut session resumes
by running the same job again), optional ``artifact`` (default ``DEFAULT_ARTIFACT``), ``embargo_hours`` (default 0: the
split already purges by label end), ``health_max_rows``, ``split_dates`` (``{"train_end", "val_end"}``; default: the
artifact's, else the 80/10/10 time quantiles), ``max_rows`` (smoke mode: at most this many rows per split, evenly
strided; default off).
Stops cleanly before ``--max-minutes`` (``Trainer.fit(deadline=...)``): the per-epoch checkpoint is the resume point.
Outputs: ``baseline_summary.json``, ``health_report.json``, ``health_report.md``, ``test_report.json``
(+ ``device.json``, ``manifest.json``).
The training data stay on the CPU and move to the model's device batch by batch (B-019).
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.device import EXPECTED as DEVICES
from nig_timenet_torch.colab.identity import identity_for_run
from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TARGET_COLUMNS, TS_COL
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.split_plan import purge_embargo_split, sample_times
from nig_timenet_torch.data_io.target_codec import TARGET_CODECS
from nig_timenet_torch.eval.target_report import target_report
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings, build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from nig_timenet_torch.workflow.health_report import REPORT_JSON, REPORT_MD, HealthRun, health_report
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta

__all__ = ["BaselineConfig", "CONFIG", "DEFAULT_ARTIFACT", "DEFAULT_HEALTH_MAX_ROWS", "SUMMARY", "TEST_REPORT",
           "anchors_by_name", "codec_state", "epoch_line", "evaluate_test", "input_key", "main", "model_settings",
           "price_targets", "split_bounds", "stride_rows", "test_lines", "trainer_config"]

DEFAULT_ARTIFACT = "/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1"
DEFAULT_HEALTH_MAX_ROWS = 100_000
LABEL_SMOOTHING = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
SUMMARY = "baseline_summary.json"


@dataclass(frozen=True)
class BaselineConfig:
    """The job's ``config`` object. Built with the strict ``from_dict`` (unknown keys raise, rule B5); the colab job
    validator builds it too, so a bad job file is refused at queue time, not on the GPU."""
    device: str
    seed: int
    epochs: int
    batch_size: int
    run_dir: str
    artifact: str = DEFAULT_ARTIFACT
    embargo_hours: float = 0.0
    health_max_rows: int = DEFAULT_HEALTH_MAX_ROWS
    split_dates: dict | None = None
    max_rows: int | None = None

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"BaselineConfig.device must be one of {DEVICES}, got {self.device!r}")
        for name in ("epochs", "batch_size", "health_max_rows"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"BaselineConfig.{name} must be an integer >= 1, got {v!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"BaselineConfig.seed must be an integer, got {self.seed!r}")
        if not isinstance(self.run_dir, str) or not self.run_dir:
            raise ValueError("BaselineConfig.run_dir must be a non-empty string")
        if self.max_rows is not None and (isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int)
                                          or self.max_rows < 1):
            raise ValueError(f"BaselineConfig.max_rows must be None or an integer >= 1, got {self.max_rows!r}")
        if self.embargo_hours < 0:
            raise ValueError("BaselineConfig.embargo_hours must be >= 0")
        if self.split_dates is not None and (not isinstance(self.split_dates, dict)
                                             or set(self.split_dates) != {"train_end", "val_end"}):
            raise ValueError("BaselineConfig.split_dates must be None or {train_end, val_end}, "
                             f"got {self.split_dates!r}")


CONFIG = BaselineConfig  # colab.job validates a job's ``config`` against the entry module's CONFIG


def price_targets(dataset: dict) -> tuple[str, ...]:
    """The artifact's ``price_targets``: a non-empty subset of ``TARGET_COLUMNS``, else raise (nothing is assumed)."""
    if "price_targets" not in dataset:
        raise ValueError("the artifact lacks 'price_targets' (a DATA_ARTIFACT_CONTRACT key)")
    targets = tuple(dataset["price_targets"])
    unknown = [t for t in targets if t not in TARGET_COLUMNS]
    if not targets or unknown or len(set(targets)) != len(targets):
        raise ValueError(f"artifact price_targets {list(targets)} must be distinct names from {TARGET_COLUMNS}")
    return targets


def codec_state(dataset: dict) -> dict:
    """The fitted state of the codec registered under the artifact's ``reg_target_mode`` (unknown mode raises
    UnknownNameError; a missing ``reg_target_scale`` raises ValueError)."""
    if "reg_target_mode" not in dataset:
        raise ValueError("the artifact lacks 'reg_target_mode' (a DATA_ARTIFACT_CONTRACT key)")
    return TARGET_CODECS[dataset["reg_target_mode"]]().fit(dataset)


def model_settings(targets: tuple[str, ...]) -> ModelSettings:
    """The anti-memorization preset; ``price_targets`` / ``head_types`` are overridden only for a target set that is
    not the default three (so the default run keeps its identity fingerprint)."""
    overrides = dict(ANTI_MEMORIZATION_CONFIG)
    if targets != tuple(TARGET_COLUMNS):
        overrides["price_targets"] = list(targets)
        overrides["head_types"] = {t: ["nig_regression", "binary_classification"] for t in targets}
    return ModelSettings("nig_timenet_v2", overrides)


def stride_rows(idx: np.ndarray, max_rows: int | None) -> np.ndarray:
    """At most ``max_rows`` of ``idx``, evenly strided over its whole range (smoke mode); ``None`` keeps all."""
    if max_rows is None or len(idx) <= max_rows:
        return idx
    return idx[np.unique(np.linspace(0, len(idx) - 1, max_rows).astype(np.int64))]


def anchors_by_name(last_candles: np.ndarray, idx: np.ndarray, targets) -> dict:
    """{target: own last price of the head}, picked by column NAME (``last_<target>``), never by position."""
    lc = np.asarray(last_candles)
    return {t: lc[idx, LAST_COLUMN_INDEX[f"last_{t}"]] for t in targets}


def trainer_config(targets, label_smoothing: float = LABEL_SMOOTHING) -> dict:
    """The oracle's trainer config (reference/data_default/config.json) with the anti-memorization label smoothing."""
    tasks = {}
    for t in targets:
        reg, cls = head_spec("nig_regression", t), head_spec("binary_classification", t)
        tasks[reg.name] = {"true_key": reg.label_key, "task_type": reg.task_type, "output_keys": reg.output_keys,
                             "use_calibration_loss": True, "lambda_reg_var": "lambda_reg",
                             "lambda_calib_var": "lambda_calib"}
        tasks[cls.name] = {"true_key": cls.label_key, "task_type": cls.task_type, "binary": True,
                             "output_keys": cls.output_keys, "label_smoothing": label_smoothing}
    return {"targets": tasks,
            "loss": {"use_uncertainty_weighting": True, "uncertainty_weighting_exclude_task_types": ["evidential"],
                     "schedules": {"lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"},
                                   "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 5,
                                                    "schedule": "cosine"}}},
            "constraints": []}


def _code_sha() -> str:
    return subprocess.check_output(["git", "-C", os.environ["NIG_REPO_DIR"], "rev-parse", "HEAD"], text=True).strip()


def input_key(dataset: dict) -> str:
    """``X_<tf>`` of a single-timeframe artifact (the TF main dataset is 4h; the test fixture is 1h)."""
    tfs = list(dataset["timeframes"])
    if len(tfs) != 1:
        raise ValueError(f"baseline handles a single-timeframe artifact, got {tfs}")
    return f"X_{tfs[0]}"


#: TF data/defaults.py train_pct / val_pct: used when neither the job config nor the artifact gives split dates
TRAIN_SHARE, VAL_SHARE = 0.80, 0.10


def split_bounds(dataset: dict, split_dates: dict | None) -> tuple:
    """(train_end, val_end): the job's ``split_dates``, else the artifact's ``split_dates``, else the 80 % / 90 %
    quantiles of the window times (the TF main dataset leaves ``split_dates`` None and splits by sample shares at train
    time; this is TF's ``raw_quantile`` method, not its default ``kept_share``, so the boundary can differ slightly)."""
    sd = split_dates or dataset.get("split_dates")
    if sd:
        return sd["train_end"], sd["val_end"]
    ts = np.sort(np.asarray(dataset["last_candles"])[:, TS_COL].astype(np.int64))
    q = [ts[min(len(ts) - 1, int(len(ts) * f))] for f in (TRAIN_SHARE, TRAIN_SHARE + VAL_SHARE)]
    return tuple(pd.Timestamp(int(v), tz="UTC") for v in q)


def epoch_line(row: dict, epochs: int) -> str:
    """One progress line per finished epoch for the Colab output."""
    def f(key):
        v = row.get(key)
        return "-" if v is None else f"{v:.4f}"
    secs = row.get("trace_epoch_seconds")
    return (f"epoch {row['epoch']}/{epochs}  train_loss {f('train_total_loss')}  val_loss {f('val_total_loss')}  "
            f"val_selection {f('val_selection_metric')}  lr {row.get('lr', 0):.2e}"
            + ("" if secs is None else f"  {secs:.0f}s"))


def _tensors(dataset: dict, idx: np.ndarray) -> tuple[torch.Tensor, dict]:
    x = torch.as_tensor(np.asarray(dataset[input_key(dataset)])[idx], dtype=torch.float32)
    kinds = ("nig_regression", "binary_classification")
    keys = [head_spec(kind, t).label_key for t in price_targets(dataset) for kind in kinds]
    y = {k: torch.as_tensor(np.asarray(dataset[k])[idx], dtype=torch.float32) for k in keys}
    return x, y


TEST_REPORT = "test_report.json"


def _flat(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flat(v, key + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield key, v


def test_lines(report: dict) -> list[str]:
    """Short printable summary of a test report: every scalar except the per-coin breakdowns."""
    return [f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in _flat(report) if "per_coin" not in k and "deciles" not in k]


@torch.no_grad()
def evaluate_test(trainer, dataset, test_idx, ident, dev, train_end, out, chunk: int = 20_000) -> dict:
    """target_report on ALL test rows (model holds the best weights after fit), in device-sized chunks.
    Writes ``test_report.json`` to ``out``; the test split is never used for training or selection."""
    if len(test_idx) == 0:
        raise ValueError("the split has no test rows")
    x_te, y_te = _tensors(dataset, test_idx)
    parts = []
    for s in range(0, len(test_idx), chunk):
        idx = test_idx[s:s + chunk]
        meta = table_meta(dataset, idx, stage="test", fold=0, train_end=train_end)
        parts.append(predict_table(trainer.model, x_te[s:s + chunk].to(dev), meta, ident))
    table = pd.concat(parts, ignore_index=True)
    lc = np.asarray(dataset["last_candles"])
    rep = target_report(table, {k: v.numpy() for k, v in y_te.items()},
                        anchors=anchors_by_name(lc, test_idx, price_targets(dataset)),
                        last_close=lc[test_idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec_state(dataset))
    (out / TEST_REPORT).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    return rep


def _health(trainer, dataset, val_idx, x_val, y_val, ident, run_dir, dev, max_rows, train_end) -> dict:
    """Health report on an evenly strided subsample of the validation rows (they must fit on the device)."""
    pick = np.unique(np.linspace(0, len(val_idx) - 1, min(len(val_idx), max_rows)).astype(np.int64))
    idx = val_idx[pick]
    xs = x_val[torch.as_tensor(pick)].to(dev)
    meta = table_meta(dataset, idx, stage="val", fold=0, train_end=train_end)
    table = predict_table(trainer.model, xs, meta, ident)
    lc = np.asarray(dataset["last_candles"])
    targets = price_targets(dataset)
    anchors = anchors_by_name(lc, idx, targets)
    probe_target = "close" if "close" in targets else targets[0]
    labels = {k: v[torch.as_tensor(pick)].numpy() for k, v in y_val.items()}
    run = HealthRun(run_dir=run_dir, model=trainer.model, x=xs, predictions=table, labels=labels, anchors=anchors,
                    last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]],
                    codec_state=codec_state(dataset), probe_points=("trunk_norm",),
                    probe_labels=tuple(head_spec(k, probe_target).label_key
                                       for k in ("nig_regression", "binary_classification")))
    health_report(run)
    return {"health_rows": int(len(idx))}


def _summary(cfg: BaselineConfig, ds: dict, art, ms, res, fit_settings, ident, plan_sizes, split, dev_rep, n_batches,
             extra: dict) -> dict:
    secs = [r["trace_epoch_seconds"] for r in res.history if "trace_epoch_seconds" in r]  # epochs of earlier jobs too
    return {"seed": cfg.seed, "epochs_requested": fit_settings.epochs, "epochs_done": res.epochs_done,
            "finished": res.epochs_done >= fit_settings.epochs or res.stopped_early,
            "stopped_early": res.stopped_early, "best_epoch": res.best_epoch, "best_value": res.best_value,
            "step_seconds_per_batch": statistics.median(secs) / n_batches if secs else None,
            "epoch_seconds_median": statistics.median(secs) if secs else None, "batch_size": fit_settings.batch_size,
            "gpu_name": dev_rep.get("gpu_name"), "device": cfg.device, "n_windows": plan_sizes,
            "split": split, "targets": list(price_targets(ds)), "label_smoothing": LABEL_SMOOTHING,
            "reg_target_mode": ds["reg_target_mode"], "reg_target_scale": float(ds["reg_target_scale"]),
            "input_key": input_key(ds), "max_rows": cfg.max_rows, "model_overrides": ms.overrides,
            "data_fingerprint": art.manifest.data_fingerprint, "identity_fingerprint": ident.fingerprint(),
            "run_dir": str(cfg.run_dir), "history": res.history, **extra}


def main(argv: list[str] | None = None) -> int:
    started = time.time()
    t_mono = time.monotonic()
    config, out, max_minutes, _rest = contract.entry_args(argv)
    cfg = from_dict(BaselineConfig, config)
    dev_rep = contract.write_device_json(out, cfg.device)
    dev = torch.device(cfg.device)
    art = load_dataset(LoaderSettings(name=cfg.artifact, mmap=True))  # rule D1 checked by default
    ds = art.dataset
    targets = price_targets(ds)
    codec_state(ds)  # refuses an unknown reg_target_mode / a missing reg_target_scale before any work
    input_key(ds)  # refuses a multi-timeframe artifact before any work

    fs, fe, le = sample_times(ds)
    train_end, val_end = split_bounds(ds, cfg.split_dates)
    plan = purge_embargo_split(le, fs, train_end, train_end, val_end, val_end,
                               pd.Timedelta(hours=cfg.embargo_hours), feature_end=fe)
    train_idx, val_idx, test_idx = (stride_rows(i, cfg.max_rows) for i in (plan.train_idx, plan.val_idx, plan.test_idx))
    x_tr, y_tr = _tensors(ds, train_idx)
    x_va, y_va = _tensors(ds, val_idx)

    seed_everything(cfg.seed)
    ms = model_settings(targets)
    model = build_model(ms, x_tr.shape[1], x_tr.shape[2]).to(dev)
    trainer = Trainer(model, trainer_config(targets), optimizer=lambda ps: torch.optim.AdamW(ps, lr=LR,
                                                                                            weight_decay=WEIGHT_DECAY))
    fit_settings = FitSettings(epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed, trace=True)
    ident = identity_for_run(trainer, fit_settings, ms, art.manifest, _code_sha())  # AFTER the Trainer

    budget = max_minutes * 60
    deadline = t_mono + budget - min(600.0, 0.2 * budget)  # the rest is for the health report and the outputs
    n_batches = -(-len(train_idx) // fit_settings.batch_size)
    print(f"data: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test windows "
          f"(max_rows {cfg.max_rows}), split train_end {train_end} val_end {val_end}; "
          f"{n_batches} batches per epoch on {dev}", flush=True)
    res = trainer.fit(x_tr, y_tr, x_va, y_va, fit_settings, run_dir=cfg.run_dir, identity=ident, deadline=deadline,
                      on_epoch=lambda row: print(epoch_line(row, fit_settings.epochs), flush=True))

    extra = _health(trainer, ds, val_idx, x_va, y_va, ident, cfg.run_dir, dev, cfg.health_max_rows, train_end)
    for name in (REPORT_JSON, REPORT_MD):
        shutil.copyfile(os.path.join(cfg.run_dir, name), out / name)
    test_rep = evaluate_test(trainer, ds, test_idx, ident, dev, train_end, out)
    print(f"test evaluation ({test_rep['n_rows']} rows, best-epoch weights):", flush=True)
    print("\n".join(test_lines(test_rep)), flush=True)

    sizes = {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))}
    summary = _summary(cfg, ds, art, ms, res, fit_settings, ident, sizes,
                       {"train_end": str(train_end), "val_end": str(val_end)}, dev_rep, n_batches, extra)
    (out / SUMMARY).write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    contract.write_manifest(out, config, input_fingerprint=art.manifest.data_fingerprint, started=started)
    print(f"baseline OK: seed {cfg.seed}, epochs {res.epochs_done}/{fit_settings.epochs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
