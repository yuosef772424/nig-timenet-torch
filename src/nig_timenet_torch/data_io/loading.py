"""One way to load a prepared dataset: name resolution, format detection, memory-mapping, local copy, checks.

``name`` accepts both spellings the TF project uses:
  * a base name, ``preprocessing_output`` or ``preprocessing_output_20261027``: ``_latest`` is added and the newer of
    ``<name>_latest.dataset`` (folder) and ``<name>_latest.pkl.gz`` is read (``format`` can force one of them);
  * a full name, ``preprocessing_output_20261027_153000.pkl.gz`` / ``....dataset`` / an artifact folder: read as
    written.
A path with a directory part (or an absolute one) ignores ``directory``. Pure numpy: no torch import here.
"""

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from nig_timenet_torch.data_io import artifact, npy_dir

__all__ = ["LoaderSettings", "LoadedData", "resolve_dataset_path", "load_dataset", "FORMATS"]

FORMATS = ("auto", "pkl.gz", "npy_dir")
LATEST = "_latest"
DIR_SUFFIX = ".dataset"
FILE_SUFFIXES = (".pkl.gz", ".pkl", DIR_SUFFIX)
DRIVE_PREFIX = "/content/drive"
LOCAL_COPY_ROOT = "/content/pipeline_data"


@dataclass(frozen=True)
class LoaderSettings:
    directory: str = "."                       # folder that holds the file; ignored when `name` has a directory part
    name: str = "preprocessing_output"         # base name (+ "_latest") or full file/folder name, see module doc
    format: str = "auto"                       # "auto" | "pkl.gz" | "npy_dir"
    mmap: bool = True                          # npy_dir: big arrays stay on disk, only read pages enter memory
    local_dir: str = "auto"                    # "auto": copy to local disk if under /content/drive; "": never; path
    check_artifact: bool = True                # folder with an `export` manifest block: apply the D1 checks
    verify_hashes: bool = False                # re-hash every file of such an artifact (reads it all once)

    def __post_init__(self):
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, not {self.format!r}")
        if not self.name:
            raise ValueError("name must not be empty")


@dataclass(frozen=True)
class LoadedData:
    dataset: dict
    path: Path
    format: str                                # "pkl.gz" | "npy_dir"
    manifest: object | None = None             # artifact.ArtifactManifest when the D1 checks ran


def _is_dataset_dir(p: Path) -> bool:
    return p.is_dir() and (p / "manifest.json").is_file()


def _stamp(p: Path) -> float:
    return (p / "manifest.json" if p.is_dir() else p).stat().st_mtime


def resolve_dataset_path(directory: str | Path, name: str, fmt: str = "auto") -> Path:
    """The file or folder ``name`` stands for (module doc). Raises FileNotFoundError listing what was tried."""
    given = Path(name)
    base = given.parent if given.parent != Path(".") else Path(directory)
    leaf = given.name
    exact = base / leaf
    full_name = leaf.endswith(FILE_SUFFIXES)
    if exact.is_file() or _is_dataset_dir(exact):
        return exact
    if full_name:
        raise FileNotFoundError(f"no dataset named {leaf!r} in {base} (a full file name is read as written)")
    stem = leaf if leaf.endswith(LATEST) else leaf + LATEST
    tried = []
    if fmt in ("auto", "npy_dir"):
        tried.append(base / f"{stem}{DIR_SUFFIX}")
    if fmt in ("auto", "pkl.gz"):
        tried.append(base / f"{stem}.pkl.gz")
    found = [p for p in tried if p.is_file() or _is_dataset_dir(p)]
    if not found:
        raise FileNotFoundError(f"no dataset for name {name!r} (format {fmt}) in {base}: tried "
                                f"{[p.name for p in tried]}; give the full file name to read a dated copy")
    return max(found, key=_stamp)


def _local_target(path: Path, local_dir: str) -> Path | None:
    if local_dir == "":
        return None
    if local_dir == "auto":
        return Path(LOCAL_COPY_ROOT) / path.name if str(path).startswith(DRIVE_PREFIX) else None
    return Path(local_dir)


def _read_pickle(path: Path) -> dict:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = pickle.load(f)  # trusted input: files written by this project or by crypto-signal-prediction
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dataset dict, found {type(data).__name__}")
    return data


def load_dataset(settings: LoaderSettings | None = None) -> LoadedData:
    """Resolve, copy to local disk if asked, read, and (for an artifact folder) check."""
    settings = settings or LoaderSettings()
    path = resolve_dataset_path(settings.directory, settings.name, settings.format)
    if not path.is_dir():
        return LoadedData(_read_pickle(path), path, "pkl.gz")
    target = _local_target(path, settings.local_dir)
    if target is not None:
        path = npy_dir.localize(path, target)
    manifest_has_export = "export" in json.loads((path / "manifest.json").read_text())
    if settings.check_artifact and manifest_has_export:
        art = artifact.load_artifact(path, mmap=settings.mmap)
        if settings.verify_hashes:
            artifact.verify_hashes(path)
        return LoadedData(art.dataset, path, "npy_dir", art.manifest)
    if settings.verify_hashes and not manifest_has_export:
        raise ValueError(f"{path}: verify_hashes needs an artifact manifest (export block); this folder has none")
    return LoadedData(npy_dir.load(path, mmap=settings.mmap), path, "npy_dir")
