"""PURPOSE: Run folders between the VM disk and Drive for the main notebook: where a run works, push its folder
    (reports, checkpoints) to Drive, restore an earlier run folder from Drive. Plain file copies, no network, no git.
TAGS: drive sync, run folder, push to drive, restore from drive, checkpoint copy, notebook paths, resume run
PITFALLS: restore refuses to replace a differing file unless overwrite=True; every file is copied through a temp name
    and renamed, so a cut copy never leaves a half file under the real name; with work_on='drive' there is nothing to
    push (the run already writes on Drive); never copies .git or the token.

The main notebook can work directly on Drive (``work_on='drive'``: slow writes, nothing lost when the VM dies, the
default and the old behaviour) or on the VM disk (``work_on='local'``: fast, then ``push_run`` after training). A run
folder is ``<root>/<name>/train`` (checkpoints, resume state, health report) and ``<root>/<name>/out`` (test report,
battery, summary). ``restore_run`` copies a finished or cut run folder from Drive into the working root, so training
resumes from its checkpoint (``Trainer.fit`` resumes a run dir that holds one).
"""

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["WORK_ON", "RunPaths", "SyncError", "run_paths", "copy_tree", "push_run", "restore_run", "PUSHED"]

WORK_ON = ("drive", "local")
PUSHED = "drive_sync.json"        # in the Drive run folder: when and what the last push copied
_SKIP = (".tmp", ".git")


class SyncError(Exception):
    """A copy the sync refuses (it would overwrite something different, or the source is not a run folder)."""


@dataclass(frozen=True)
class RunPaths:
    name: str
    work_on: str
    work_root: str          # where the run writes: <work_root>/<name>/{train,out}
    drive_root: str         # the durable copy: <drive_root>/<name>

    @property
    def run_dir(self) -> str:
        return str(Path(self.work_root) / self.name / "train")

    @property
    def out_dir(self) -> str:
        return str(Path(self.work_root) / self.name / "out")

    @property
    def drive_dir(self) -> str:
        return str(Path(self.drive_root) / self.name)

    @property
    def work_dir(self) -> str:
        return str(Path(self.work_root) / self.name)


def run_paths(name: str, drive_root: str, local_root: str, work_on: str = "drive") -> RunPaths:
    if work_on not in WORK_ON:
        raise SyncError(f"WORK_ON must be one of {WORK_ON}, not {work_on!r}")
    if not name or "/" in name or name in (".", ".."):
        raise SyncError(f"RUN_NAME must be a plain folder name, not {name!r}")
    return RunPaths(name, work_on, drive_root if work_on == "drive" else local_root, drive_root)


def _same_file(a: Path, b: Path) -> bool:
    return b.is_file() and a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def copy_tree(src, dst, *, overwrite: bool = False) -> dict:
    """Copy every file under ``src`` to ``dst`` (missing folders are made). A destination file that differs raises
    SyncError BEFORE anything is written unless ``overwrite``; identical files are skipped.
    -> {copied, skipped, bytes}."""
    src, dst = Path(src), Path(dst)
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and not any(s in p.parts or p.name.endswith(s)
                                                                          for s in _SKIP)]
    clash = [str(p.relative_to(src)) for p in files if (dst / p.relative_to(src)).exists()
             and not _same_file(p, dst / p.relative_to(src))]
    if clash and not overwrite:
        raise SyncError(f"{dst} already holds different files ({clash[:5]}{'...' if len(clash) > 5 else ''}); "
                        f"set overwrite=True to replace them")
    done = skipped = size = 0
    for p in files:
        target = dst / p.relative_to(src)
        if _same_file(p, target):
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(p, tmp)
        os.replace(tmp, target)
        done, size = done + 1, size + p.stat().st_size
    return {"copied": done, "skipped": skipped, "bytes": size}


def push_run(paths: RunPaths, *, overwrite: bool = True, echo=print) -> dict:
    """Copy the working run folder to ``paths.drive_dir`` (newer files replace older ones by default: the latest state
    of one run is the point). Nothing to do when the run already works on Drive."""
    if paths.work_on == "drive" or Path(paths.work_dir).resolve() == Path(paths.drive_dir).resolve():
        echo(f"the run already works on Drive ({paths.drive_dir}): nothing to push")
        return {"copied": 0, "skipped": 0, "bytes": 0, "already_on_drive": True}
    if not Path(paths.work_dir).is_dir():
        raise SyncError(f"nothing to push: {paths.work_dir} does not exist (run a step first)")
    res = copy_tree(paths.work_dir, paths.drive_dir, overwrite=overwrite)
    Path(paths.drive_dir, PUSHED).write_text(json.dumps({**res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                             encoding="utf-8")
    echo(f"pushed to Drive: {res['copied']} files copied ({res['bytes'] / 1e6:.1f} MB), {res['skipped']} unchanged "
         f"-> {paths.drive_dir}")
    return {**res, "already_on_drive": False}


def restore_run(source: str, paths: RunPaths, *, overwrite: bool = False, echo=print) -> dict:
    """Copy an earlier run folder into this run's working folder. ``source`` is a run name under ``paths.drive_root``
    or a path of a run folder (one that holds ``train`` and/or ``out``). Refuses to replace a differing file unless
    ``overwrite``. ``""`` does nothing."""
    if not source:
        return {"copied": 0, "skipped": 0, "bytes": 0, "restored_from": ""}
    src = Path(source) if os.path.isabs(source) else Path(paths.drive_root) / source
    if not any((src / sub).is_dir() for sub in ("train", "out")):
        raise SyncError(f"{src} is not a run folder (no train/ or out/ inside); check RESTORE_FROM")
    if src.resolve() == Path(paths.work_dir).resolve():
        echo(f"{src} is this run's own folder: nothing to restore")
        return {"copied": 0, "skipped": 0, "bytes": 0, "restored_from": str(src)}
    res = copy_tree(src, paths.work_dir, overwrite=overwrite)
    echo(f"restored {res['copied']} files ({res['bytes'] / 1e6:.1f} MB) from {src} -> {paths.work_dir}")
    return {**res, "restored_from": str(src)}
