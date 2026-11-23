"""PURPOSE: A model loadable from a folder path alone: ``live_model.json`` (architecture, overrides, input shape, data
    settings, codec state, run identity) beside ``best.pt``; export from a finished Run, load for prediction.
TAGS: live model, model bundle, load from folder, export bundle, best weights, codec state, data settings,
    live trading
PITFALLS: ``coins`` is the TRAINING universe (the cross-asset columns breadth / rank / orth were computed over it,
    so a live run must read the same coins); the weights are the run's ``best.pt`` (the model state incl. the
    train-fitted scalers, so nothing is refitted live); ``load_live_model`` refuses a folder without the json or the
    weights and weights that do not fit the architecture the json names; a model with a coin channel (snapshot
    panel) is refused: its coin ids must follow the TRAINING coin order, not built live yet; ``data_settings`` is
    the artifact's own (present only in datasets built by ``nig_timenet_torch.data``, not in a TF export: then pass
    them to the live run); the json holds no secret and no data path. Exploratory (rule C5).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.trainer.checkpoint import BEST
from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["BUNDLE", "SCHEMA", "LiveModel", "export_live_model", "export_from_folder", "load_live_model", "main"]

BUNDLE, SCHEMA = "live_model.json", "live_model/1"
#: the dataset facts a live row needs to be built and read like the training rows
_FACTS = ("base_tf", "timeframes", "window_sizes", "feature_order", "forecast_horizon", "reg_target_mode",
          "reg_target_scale", "reg_target_clip", "price_norm_mode", "scaler_type", "price_targets", "stride")


@dataclass(frozen=True)
class LiveModel:
    """A model ready to predict: ``model`` (eval mode, weights loaded), its ``spec`` (the json), the run identity and
    the codec state that decodes its outputs to returns."""
    folder: Path
    spec: dict
    model: torch.nn.Module
    identity: RunIdentity
    codec_state: dict

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self.spec["targets"])


def _spec(architecture: str, overrides: dict, ds: dict, targets, x_shape, flags: tuple, ident: dict,
          train_end) -> dict:
    """The json of a model folder from its parts (the ONE place its keys are written); ``flags`` = (cross_section,
    coin_channel, input_timeframes)."""
    cross, coin, frames = flags
    return {"schema": SCHEMA, "architecture": architecture, "overrides": overrides, "seq_len": int(x_shape[1]),
            "model_n_features": int(x_shape[2]) - int(cross), "cross_section": bool(cross),
            "coin_channel": bool(coin), "input_timeframes": list(frames), "targets": list(targets),
            "dataset": {k: ds[k] for k in _FACTS if k in ds}, "data_settings": ds.get("data_settings"),
            "coins": [b["name"] for b in ds["asset_bounds"]], "codec_state": run_common.codec_state(ds),
            "train_end": str(train_end), "identity": ident}


def _write(folder: Path, spec: dict) -> Path:
    if not (folder / BEST).is_file():
        raise FileNotFoundError(f"{folder / BEST} not found: export a folder that holds the best weights")
    path = folder / BUNDLE
    path.write_text(json.dumps(spec, indent=1, default=run_common.json_default), encoding="utf-8")
    return path


def export_live_model(run: Run, folder=None) -> Path:
    """Write ``live_model.json`` for a finished ``run`` into ``folder`` (default: its run folder, which holds
    ``best.pt``); returns the json path. The weights must exist: train or ``restore`` first."""
    need(run, "ms", "build")
    s = run.settings
    spec = _spec(run.ms.architecture, run.ms.overrides, run.dataset, run.targets, run.x["train"].shape,
                 (s.split.cross_section, s.split.coin_channel, s.data.input_timeframes), run.ident.to_json(),
                 run.bounds[0])
    return _write(Path(folder or s.run_dir), spec)


def export_from_folder(folder, artifact, *, architecture: str | None = None, overrides: dict | None = None,
                       cross_section: bool = False, input_timeframes=()) -> Path:
    """The same json for a run that was trained before this existed: ``folder`` = its run folder (``best.pt``,
    ``checkpoint.pt``, ``baseline_summary.json``), ``artifact`` = the artifact it trained on. The model overrides and
    the split come from ``baseline_summary.json`` (``overrides`` / ``architecture`` override them; the summary does
    not record the architecture, so a non-default one must be given). The weights are loaded once to prove they fit."""
    from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
    from nig_timenet_torch.model.architectures import DEFAULT_ARCHITECTURE
    from nig_timenet_torch.trainer import checkpoint

    folder = Path(folder)
    summary_path = folder / "baseline_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    ckpt = checkpoint.load_checkpoint(folder)
    if ckpt is None:
        raise FileNotFoundError(f"{folder} holds no checkpoint.pt: the run identity cannot be read")
    ds = load_dataset(LoaderSettings(name=str(artifact))).dataset
    targets = run_common.price_targets(ds)
    x, _ = flat_run.flat_inputs(ds, np.arange(2), cross_section, False, tuple(input_timeframes))
    train_end = (summary.get("split") or {}).get("train_end")
    if not train_end:
        raise ValueError(f"{summary_path} has no split.train_end: write the summary (run.write_summary) first")
    spec = _spec(architecture or DEFAULT_ARCHITECTURE, overrides if overrides is not None
                 else summary.get("model_overrides", {}), ds, targets, x.shape,
                 (cross_section, False, tuple(input_timeframes)), json.loads(ckpt["identity"]), train_end)
    _build(folder, spec)                                 # a json that does not load is worse than none: prove it first
    return _write(folder, spec)


def _build(folder: Path, spec: dict, device: str = "cpu") -> torch.nn.Module:
    """The model of ``spec`` with the weights of ``folder`` loaded, eval mode."""
    if spec["coin_channel"]:
        raise NotImplementedError("a coin-channel (snapshot panel) model needs the training coin order for its coin "
                                  "ids; not supported by the live run yet")
    weights = folder / BEST
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} not found: the folder holds the json but not the best weights")
    model = build_model(ModelSettings(spec["architecture"], spec["overrides"]), spec["seq_len"],
                        spec["model_n_features"])
    try:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    except RuntimeError as e:
        raise ValueError(f"{weights} does not fit the architecture {spec['architecture']!r} {spec['overrides']} "
                         f"({spec['seq_len']} x {spec['model_n_features']}): {str(e)[:300]}") from e
    return model.to(device).eval()


def load_live_model(folder, device: str = "cpu") -> LiveModel:
    """The model of ``folder`` (a run folder with ``live_model.json`` and ``best.pt``), built from the json, weights
    loaded, in eval mode on ``device``."""
    folder = Path(folder)
    path = folder / BUNDLE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: call workflow.live_model.export_live_model on the finished run")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {spec.get('schema')!r}, this code reads {SCHEMA!r}")
    return LiveModel(folder, spec, _build(folder, spec, device), RunIdentity.from_json(spec["identity"]),
                     spec["codec_state"])


def main(argv=None) -> None:
    """``python -m nig_timenet_torch.workflow.live_model RUN_FOLDER --artifact ARTIFACT [--architecture NAME]
    [--overrides JSON] [--cross-section] [--timeframes 4h,1h]``: write ``live_model.json`` for a finished run."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("folder")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--architecture")
    ap.add_argument("--overrides", help="JSON object, replaces the model_overrides of baseline_summary.json")
    ap.add_argument("--cross-section", action="store_true")
    ap.add_argument("--timeframes", default="")
    a = ap.parse_args(argv)
    path = export_from_folder(a.folder, a.artifact, architecture=a.architecture,
                              overrides=json.loads(a.overrides) if a.overrides else None,
                              cross_section=a.cross_section, input_timeframes=[t for t in a.timeframes.split(",") if t])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
