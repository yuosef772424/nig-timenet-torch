"""Readers of the on-disk dataset artifacts pinned in docs/DATA_ARTIFACT_CONTRACT.md (pure numpy, no torch)."""

from nig_timenet_torch.data_io.loading import LoadedData, LoaderSettings, load_dataset

__all__ = ["LoadedData", "LoaderSettings", "load_dataset"]
