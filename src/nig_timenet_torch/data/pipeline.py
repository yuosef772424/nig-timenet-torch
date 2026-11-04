"""The data pipeline as one call: settings -> dataset dict -> (optionally) a contract artifact on disk."""

import logging

from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data.settings import DataSettings

__all__ = ["build_artifact", "summary_lines"]

log = logging.getLogger(__name__)


def summary_lines(dataset: dict) -> list[str]:
    """A few human-readable lines about a built dataset."""
    n = len(dataset["last_candles"])
    lines = [f"{n} windows, {len(dataset['asset_bounds'])} coins, {len(dataset['feature_order'])} features, "
             f"timeframes {dataset['timeframes']}, window sizes {dataset['window_sizes']}, stride {dataset['stride']}"]
    for name, why in dataset["skipped_assets"].items():
        lines.append(f"skipped {name}: {why}")
    if dataset["sealed_dropped"]:
        lines.append(f"{len(dataset['sealed_dropped'])} sealed coins were not loaded")
    return lines


def build_artifact(settings: DataSettings, loader=None) -> tuple[dict, dict | None]:
    """Build the dataset; with ``settings.output.path`` also write it. Returns (dataset, manifest or None)."""
    dataset = build_dataset(settings, loader=loader)
    if settings.runtime.verbose:
        for line in summary_lines(dataset):
            log.info(line)
    o = settings.output
    if not o.path:
        return dataset, None
    manifest = write_artifact(dataset, o.path, source_commit=o.source_commit, producer=o.producer,
                              overwrite=o.overwrite)
    return dataset, manifest
