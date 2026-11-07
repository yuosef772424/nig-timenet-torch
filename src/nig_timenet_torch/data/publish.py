"""Named dataset output

PURPOSE: Publish a built artifact as <name>_<stamp>.dataset plus a <name>_latest.dataset copy in the old file-name
    convention.
TAGS: publish, named output, latest dataset, stamp, file name
PITFALLS: the _latest copy is replaced on every publish; pass the full dated name to a job to pin a run to one
    build.

Named output of a build, in the TF file-name convention: ``<name>_<stamp>.dataset`` plus ``<name>_latest.dataset``.

``data_io.loading.load_dataset`` reads either: the base name picks ``_latest``, the full name picks a dated copy.
"""

import datetime as dt
import shutil
from pathlib import Path

from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data.settings import OutputSettings

__all__ = ["publish_artifact", "STAMP_FORMAT"]

STAMP_FORMAT = "%Y%m%d_%H%M%S"
SUFFIX = ".dataset"


def publish_artifact(dataset: dict, o: OutputSettings) -> tuple[Path, dict]:
    """Write the dated folder and (if ``o.latest``) replace the ``_latest`` copy. Returns (dated path, manifest)."""
    stamp = o.stamp or dt.datetime.now(dt.timezone.utc).strftime(STAMP_FORMAT)
    root = Path(o.directory)
    root.mkdir(parents=True, exist_ok=True)
    dated = root / f"{o.name}_{stamp}{SUFFIX}"
    manifest = write_artifact(dataset, dated, source_commit=o.source_commit, producer=o.producer, overwrite=o.overwrite)
    if o.latest:
        latest = root / f"{o.name}_latest{SUFFIX}"
        tmp = latest.with_name(latest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(dated, tmp)
        shutil.rmtree(latest, ignore_errors=True)
        tmp.rename(latest)
    return dated, manifest
