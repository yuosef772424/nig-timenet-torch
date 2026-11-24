"""PURPOSE: A run name that is already taken by a run with other settings does not stop the notebook: the run takes the
    next free name (``<name>-2``, ``-3``, ...) and every later step uses its folders; the same settings resume.
TAGS: run name, name taken, new run, resume, run identity, notebook run folder, claim name, auto rename
PITFALLS: needs the BUILT run (the identity exists only after ``build``) so call it between build and train; it
    looks in the work folder AND the Drive copy (a push would otherwise overwrite another run's Drive folder); a
    folder without a checkpoint counts as free; jobs do not call it (a job's folders are named by its job file and
    a resume across a changed identity there must still raise).
"""

import dataclasses
import json
from pathlib import Path

from nig_timenet_torch.colab.drive_sync import RunPaths
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.identity import ResumeMismatchError, RunIdentity, check_resume
from nig_timenet_torch.workflow.run_state import Run, need

__all__ = ["ON_TAKEN", "claim_run_name", "candidate_name"]

ON_TAKEN = ("new_name", "error")
MAX_TRIES = 999


def candidate_name(name: str, n: int) -> str:
    """The n-th try of a run name: the name itself, then ``<name>-2``, ``<name>-3``..."""
    return name if n <= 1 else f"{name}-{n}"


def _train_dirs(paths: RunPaths) -> list[Path]:
    """The folders a run of this name may hold a checkpoint in: where it works and its Drive copy."""
    dirs = [Path(paths.run_dir), Path(paths.drive_dir) / "train"]
    return [d for i, d in enumerate(dirs) if d not in dirs[:i]]


def _verdict(paths: RunPaths, ident: RunIdentity) -> str | None:
    """None = the name is usable (no checkpoint, or one of this very run); else why it is taken."""
    for folder in _train_dirs(paths):
        ckpt = checkpoint.load_checkpoint(folder)
        if ckpt is None:
            continue
        try:
            check_resume(RunIdentity.from_json(json.loads(ckpt["identity"])), ident)
        except ResumeMismatchError as err:
            return f"{folder} holds a run with other settings or code ({str(err).splitlines()[1].strip()}, ...)"
    return None


def claim_run_name(run: Run, paths: RunPaths, *, on_taken: str = "new_name", echo=print) -> RunPaths:
    """The paths this built ``run`` works in. A name whose folders hold no checkpoint, or the checkpoint of the SAME
    run identity (same settings, seed, data, commit: the run resumes), is kept. A name taken by a run with another
    identity gets the next free ``-n`` name instead of the ``ResumeMismatchError`` the training step would raise;
    ``run.settings.run_dir`` and ``run.out`` move to the new folders (the caller rebinds ``paths``).
    ``on_taken="error"`` keeps the old behaviour: the paths come back unchanged and ``train`` raises."""
    if on_taken not in ON_TAKEN:
        raise ValueError(f"on_taken must be one of {ON_TAKEN}, got {on_taken!r}")
    need(run, "ident", "build")
    if on_taken == "error":
        return paths
    why = []
    for n in range(1, MAX_TRIES + 1):
        cand = dataclasses.replace(paths, name=candidate_name(paths.name, n))
        reason = _verdict(cand, run.ident)
        if reason is None:
            break
        why.append(f"{cand.name}: {reason}")
    else:
        raise RuntimeError(f"no free run name among {MAX_TRIES} tries of {paths.name!r}")
    if cand.name != paths.name:
        echo(f"the run name {paths.name!r} is taken by a run with other settings or code; this run is named "
             f"{cand.name!r}\n  " + "\n  ".join(why))
        run.settings = dataclasses.replace(run.settings, run_dir=cand.run_dir)
        run.out = Path(cand.out_dir)
        run.out.mkdir(parents=True, exist_ok=True)
    return cand
