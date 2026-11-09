"""PURPOSE: The gate run: E1 -> E2 -> E3 x seeds through the run steps, gate A before / after each short run, gates
    B / C / canary on the test split, one report keyed by the configuration hash; gate_check refuses an ungated run.
TAGS: gate run, gate report, gate job, e1 e2 e3, seeds, configuration hash, job-side check, baseline arm, refusal
PITFALLS: every env is a contract artifact written under ``run_dir/envs`` and read back through the loader (the same
    code path as a real run); each env x seed run folder is deleted and rebuilt (gate runs never resume); without
    ``e3_source`` E2 permutes E1 and E3 is not run, so the report cannot pass (protocol: all three envs); the key
    leaves out the seed and the data, so the full job of the same configuration finds it.

``run_gates(settings, gates, out)``: ``settings`` is the configuration under test (its ``model`` and ``train``
sections; ``train.seed`` / ``train.epochs`` are replaced per env and seed; ``run_dir`` is the work folder).
``gate_check(run, report_path)`` is called by a training job after ``build``: a full training (more than one epoch,
no ``max_rows``) of a non-baseline configuration needs a gate report of the same key that passed, else
``GateRefused``; the baseline arm (default model and training settings) is exempt and the exemption is recorded.
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX
from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io.environments import permute_labels, real_slice
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.data_io.synthetic import SyntheticSettings, oracle_keys, write_synthetic
from nig_timenet_torch.eval import gates
from nig_timenet_torch.eval.vol_baseline import window_ranges
from nig_timenet_torch.trainer.grad_report import grad_report
from nig_timenet_torch.trainer.identity import configuration_hash
from nig_timenet_torch.workflow import flat_run, run, run_common
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings, default_model_settings

__all__ = ["GateSettings", "GateRefused", "run_gates", "gate_check", "configuration_key", "baseline_arm",
           "GATE_REPORT", "GATE_REPORT_MD", "REPORT_SCHEMA"]

GATE_REPORT, GATE_REPORT_MD = "gate_report.json", "gate_report.md"
REPORT_SCHEMA = "gate_report/1"
_PER_RUN_KEYS = ("seed", "epochs", "device", "trace_representative")   # not part of the configuration


@dataclass(frozen=True)
class GateSettings:
    """What the gate run does around the configuration (MODEL_DEV_PROTOCOL sections 2-3)."""
    seeds: tuple[int, ...] = (0, 1, 2)
    e1: dict = field(default_factory=dict)      # SyntheticSettings fields (default: the D-018 generator)
    e1_epochs: int = 3                          # protocol gate A: "E1: 3 epochs of the synthetic set"
    e2_epochs: int = 3
    e2_permutation_seed: int = 0
    e3_source: str = ""                         # the real artifact; "" = no E3 (E2 then permutes E1)
    e3_train_end: str = ""                      # the full run's train end; "" = its default split
    e3_n_coins: int = 20
    e3_months: int = 6
    e3_epochs: int = 3                          # 1 epoch never trains the confidence head (lambda_calib 0 at epoch 0)

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        if not self.seeds or min(self.e1_epochs, self.e2_epochs, self.e3_epochs, self.e3_n_coins,
                                 self.e3_months) < 1:
            raise ValueError("seeds non-empty; epochs, e3_n_coins and e3_months >= 1")
        from_dict(SyntheticSettings, self.e1)          # unknown generator keys raise now


class GateRefused(RuntimeError):
    """A full training of a configuration without a passing gate report of its key (rule C6)."""


def _identity_parts(r: run.Run) -> tuple:
    return r.fit_settings, r.ms, r.trainer.config, r.trainer.optimizer


def configuration_key(r: run.Run) -> str:
    """The configuration hash of a built run (``trainer.identity.configuration_hash``)."""
    run._need(r, "trainer", "build")
    return configuration_hash(*_identity_parts(r))


def baseline_arm(settings: RunSettings) -> bool:
    """True for today's default configuration: the default model and the default training settings except the
    per-run ones (seed, epochs, batch size, device, trace switch)."""
    t = replace(settings.train, seed=TrainSettings.seed, epochs=TrainSettings.epochs,
                batch_size=TrainSettings.batch_size, device=TrainSettings.device,
                trace_representative=TrainSettings.trace_representative)
    return settings.model == default_model_settings() and t == TrainSettings()


def _configuration(settings: RunSettings) -> dict:
    train = {k: v for k, v in asdict(settings.train).items() if k not in _PER_RUN_KEYS}
    return {"model": asdict(settings.model), "train": train}


# ------------------------------------------------------------------------------------------------ environments
def _write_envs(settings: RunSettings, g: GateSettings, root: Path) -> dict[str, dict]:
    """{env: {"path", "epochs", "fingerprint", "rows"}} of the envs this run can build (E3 only with a source)."""
    envs: dict[str, dict] = {}
    man = write_synthetic(root / "E1", from_dict(SyntheticSettings, g.e1), overwrite=True)
    envs["E1"] = {"path": str(root / "E1"), "epochs": g.e1_epochs, "manifest": man}
    base = load_dataset(LoaderSettings(name=str(root / "E1"), mmap=False, local_dir="")).dataset
    if g.e3_source:
        src = load_dataset(LoaderSettings(name=g.e3_source, mmap=True))
        end = g.e3_train_end or run_common.split_bounds(src.dataset, SplitSettings())[0]
        e3 = real_slice(src.dataset, train_end=end, n_coins=g.e3_n_coins, months=g.e3_months)
        e3["environment"]["source_fingerprint"] = src.manifest.data_fingerprint
        man = write_artifact(e3, root / "E3", source_commit=src.manifest.source_commit, overwrite=True,
                             producer="nig_timenet_torch.workflow.gate_run:e3")
        envs["E3"] = {"path": str(root / "E3"), "epochs": g.e3_epochs, "manifest": man}
        base = load_dataset(LoaderSettings(name=str(root / "E3"), mmap=False, local_dir="")).dataset
    e2 = permute_labels(base, seed=g.e2_permutation_seed)
    man = write_artifact(e2, root / "E2", source_commit=_source_commit(envs), overwrite=True,
                         producer="nig_timenet_torch.workflow.gate_run:e2")
    envs["E2"] = {"path": str(root / "E2"), "epochs": g.e2_epochs, "manifest": man,
                  "permutes": "E3" if g.e3_source else "E1 (no E3 source)"}
    return {k: envs[k] for k in gates.ENVS if k in envs}


def _source_commit(envs: dict) -> str:
    return envs.get("E3", envs["E1"])["manifest"]["export"]["source_commit"]


# ------------------------------------------------------------------------------------------------ one env x seed
@torch.no_grad()
def _losses(trainer, x, y, batch_size: int) -> dict:
    """{"model": val raw loss, "constant": the raw loss of the model's MEAN output on every row} (the canary)."""
    trainer.model.eval()
    dev = next(trainer.model.parameters()).device
    parts = [trainer.model(xb.to(dev)) for xb in x.split(batch_size)]
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    yd = {k: v.to(dev) for k, v in y.items()}
    const = {k: v.mean(dim=0, keepdim=True).expand_as(v) for k, v in out.items()}
    return {"model": float(trainer.compute_losses(None, yd, outputs=out)[1]),
            "constant": float(trainer.compute_losses(None, yd, outputs=const)[1])}


def _stage(r: run.Run, env: str, seed: int, *, with_oracle: bool, with_losses: bool) -> gates.GateStage:
    ds = r.dataset
    table = r.table("test")
    rep = run.stage_report(r, "test", table)
    idx, lc, codec = r.idx["test"], np.asarray(ds["last_candles"]), run_common.codec_state(ds)
    x = r.x["test"].numpy()
    fo = list(ds["feature_order"])
    try:
        ranges = window_ranges(x, fo)
    except (ValueError, KeyError):
        ranges = None
    oracle = ({t: (np.asarray(ds[oracle_keys(t)[0]])[idx], np.asarray(ds[oracle_keys(t)[1]])[idx])
               for t in r.targets} if with_oracle else None)
    losses = _losses(r.trainer, r.x["val"], r.y["val"], r.fit_settings.batch_size) if with_losses else None
    return gates.GateStage(env=env, seed=seed, table=table, labels={k: v.numpy() for k, v in r.y["test"].items()},
                           report=rep, anchors=flat_run.anchors_by_name(lc, idx, r.targets, codec),
                           last_close=lc[idx, LAST_COLUMN_INDEX["last_close"]], codec_state=codec,
                           ranges=ranges, last_move=x[:, -1, fo.index("close")] if "close" in fo else None,
                           oracle=oracle, losses=losses)


def _env_seed(settings: RunSettings, env: str, info: dict, seed: int, root: Path) -> tuple[list, dict, str]:
    run_dir = root / env / f"s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)         # a gate run never resumes (this job owns the folder)
    s = replace(settings, run_dir=str(run_dir), split=SplitSettings(),
                data=LoaderSettings(name=info["path"], mmap=False, local_dir=""),
                train=replace(settings.train, seed=seed, epochs=info["epochs"]))
    t0 = time.monotonic()
    r = run.build(run.prepare(s))
    batches = flat_run.batch_set(r.dataset, r.idx["train"], r.fit_settings.batch_size, seed)
    a0 = grad_report(r.trainer, r.x["train"], r.y["train"], batches)
    run.train(r)
    a1 = grad_report(r.trainer, r.x["train"], r.y["train"], batches)
    st = _stage(r, env, seed, with_oracle=env == gates.ENVS[0], with_losses=env == gates.ENVS[1])
    rows = gates.gate_a_rows(env, seed, "start", a0) + gates.gate_a_rows(env, seed, "end", a1) + gates.stage_rows(st)
    meta = {"seed": seed, "rows": {k: int(len(v)) for k, v in r.idx.items()}, "epochs_done": r.result.epochs_done,
            "best_epoch": r.result.best_epoch, "seconds": round(time.monotonic() - t0, 1),
            "gate_a_dead_start": a0["checks"]["A2"]["dead"], "gate_a_scheduled_off_start":
                a0["checks"]["A2"]["scheduled_off"]}
    return rows, meta, configuration_key(r)


def run_gates(settings: RunSettings, g: GateSettings | None = None, out=None) -> dict:
    """The gate run; returns the report and writes ``gate_report.json`` / ``.md`` to ``out`` (if given)."""
    g = g or GateSettings()
    t0 = time.monotonic()
    root = Path(settings.run_dir)
    envs = _write_envs(settings, g, root / "envs")
    rows, meta, keys = [], {}, set()
    for env, info in envs.items():
        meta[env] = {"path": info["path"], "fingerprint": info["manifest"]["export"]["data_sha256"],
                     "rows": info["manifest"]["export"]["rows"], "epochs": info["epochs"],
                     **({"permutes": info["permutes"]} if "permutes" in info else {}), "runs": []}
        for seed in g.seeds:
            r_rows, r_meta, key = _env_seed(settings, env, info, seed, root)
            rows += r_rows
            meta[env]["runs"].append(r_meta)
            keys.add(key)
            print(f"gates {env} seed {seed}: {r_meta['seconds']} s", flush=True)
    if len(keys) != 1:
        raise RuntimeError(f"the envs built different configurations (keys {sorted(keys)}): check seq_len / features")
    summary = gates.summarize(rows)
    report = {"schema": REPORT_SCHEMA, "key": keys.pop(), "configuration": _configuration(settings),
              "baseline_arm": baseline_arm(settings), "code_sha": run_common.code_sha(settings),
              "gate_settings": asdict(g), "envs": meta, "rows": rows, "summary": summary,
              "seconds": round(time.monotonic() - t0, 1)}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / GATE_REPORT).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        (out / GATE_REPORT_MD).write_text(gates.report_markdown(report), encoding="utf-8")
    return report


# ------------------------------------------------------------------------------------------------ job-side check
def gate_check(r: run.Run, report_path: str | None) -> dict:
    """Rule C6 for a training job, after ``build``: record and enforce the gate status of the configuration (see the
    module docstring); the record goes to ``r.extra['gate_check']`` (and so into the run summary)."""
    key = configuration_key(r)
    full = r.settings.split.max_rows is None and r.fit_settings.epochs > 1
    rec = {"key": key, "full_training": full, "baseline_arm": baseline_arm(r.settings), "report": report_path}
    found = None
    if report_path:
        found = json.loads(Path(report_path).read_text(encoding="utf-8"))
        rec.update(report_key=found.get("key"), report_passed=bool(found.get("summary", {}).get("passed")),
                   report_failed=found.get("summary", {}).get("failed", [])[:20])
    if not full:
        rec["result"] = "short run: not checked"
    elif rec["baseline_arm"]:
        rec["result"] = "baseline arm: exempt (MODEL_DEV_PROTOCOL section 2), failures recorded in its result file"
    elif found is None:
        raise GateRefused(f"full training of configuration {key[:16]} without a gate report (rule C6): run the "
                          "`gates` job for this configuration and pass its gate_report.json")
    elif found.get("key") != key:
        raise GateRefused(f"the gate report is for configuration {str(found.get('key'))[:16]}, this run is {key[:16]}")
    elif not rec["report_passed"]:
        raise GateRefused(f"the gate report of {key[:16]} did not pass: {rec['report_failed'][:5]}")
    else:
        rec["result"] = "gate report passed"
    r.extra["gate_check"] = rec
    return rec
