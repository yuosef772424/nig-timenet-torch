"""Section 2 of the notebook (ROADMAP T3): turn the TF project's training data into a contract artifact, once.

``prepare_artifact`` runs the TF export tool (``tools/export_torch_artifact.py`` of the TF checkout) on the source
(``preprocessing_output_latest.pkl.gz``, the file TF ``main.ipynb`` trains on, or an npy_dir folder), then checks the
result with this project's own loader (``data_io/artifact.py``: ``load_artifact`` + ``verify_hashes``). It SKIPS the
export when the artifact already there is sound and was made from the same source:

  same source = same ``source`` fingerprint (file: size + sha256; folder: sha256 of its manifest.json)
                AND same export logic (sha256 of the tool) AND same TF lock (sha256 of holdout.json)
                AND the artifact still loads and re-hashes to the ``data_sha256`` the record names.

The record is a sidecar ``<artifact>.source.json`` NEXT to the artifact folder (a file inside would break
``verify_hashes``, which lists every file). An artifact made by hand (no sidecar) is re-exported, unless the caller
passes ``adopt_existing=True``: then a sound artifact is trusted as made from this source and only the sidecar is
written. That is a claim by the caller, not a check, so it is off by default.
"""

import json
import subprocess
import sys
from pathlib import Path

from nig_timenet_torch.data_io import artifact

LOCK_FILE = Path("docs", "research", "plan", "lock", "holdout.json")
TOOL_FILE = Path("tools", "export_torch_artifact.py")


class PrepareError(RuntimeError):
    """The export or the check of its result failed (the message says which)."""


def _sha(path: Path) -> str:
    return artifact._sha256_file(path)


def source_fingerprint(source) -> dict:
    p = Path(source)
    if p.is_dir():
        return {"kind": "npy_dir", "manifest_sha256": _sha(p / "manifest.json")}
    if p.is_file():
        return {"kind": "file", "size": p.stat().st_size, "sha256": _sha(p)}
    raise PrepareError(f"source {source} does not exist")


def export_key(tf_dir) -> dict:
    """What, besides the data, decides the export's output: the tool's code and the lock it applies."""
    root = Path(tf_dir)
    return {"export_tool_sha256": _sha(root / TOOL_FILE), "lock_sha256": _sha(root / LOCK_FILE)}


def sidecar_path(out_dir) -> Path:
    out = Path(out_dir)
    return out.with_name(out.name + ".source.json")


def sound_data_sha(out_dir) -> str | None:
    """The artifact's data_sha256 if it loads and re-hashes cleanly, else None (missing, half-written or corrupt)."""
    if not (Path(out_dir) / "manifest.json").is_file():
        return None
    try:
        art = artifact.load_artifact(out_dir)
        artifact.verify_hashes(out_dir)
    except (artifact.ArtifactError, OSError, ValueError, KeyError):
        return None
    return art.manifest.data_fingerprint


def _read_sidecar(out_dir) -> dict:
    try:
        return json.loads(sidecar_path(out_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_sidecar(out_dir, fp: dict, key: dict, data_sha: str) -> None:
    sidecar_path(out_dir).write_text(json.dumps({"source": fp, "key": key, "data_sha256": data_sha}, indent=1,
                                                sort_keys=True), encoding="utf-8")


def prepare_artifact(tf_dir, source, out_dir, adopt_existing: bool = False, python: str | None = None, echo=print,
                     run=subprocess.run) -> dict:
    """Export if needed, verify, and return {"action": "exported"|"skipped"|"adopted", "data_sha256", "path",
    "summary"}. Raises PrepareError when the export fails or its result does not verify."""
    fp, key = source_fingerprint(source), export_key(tf_dir)
    have = sound_data_sha(out_dir)
    side = _read_sidecar(out_dir)
    if have and side.get("source") == fp and side.get("key") == key and side.get("data_sha256") == have:
        echo(f"export skipped: {out_dir} is sound and was made from this source (data_sha256 {have[:12]})")
        return _result("skipped", out_dir, have)
    if have and adopt_existing and not side:
        _write_sidecar(out_dir, fp, key, have)
        echo(f"adopted the existing artifact {out_dir} as made from this source (caller's claim, not checked)")
        return _result("adopted", out_dir, have)
    cmd = [python or sys.executable, str(Path(tf_dir) / TOOL_FILE), "--dataset", str(source), "--out", str(out_dir)]
    echo("exporting: " + " ".join(cmd[1:]))
    r = run(cmd, cwd=str(tf_dir), capture_output=True, text=True)
    echo((r.stdout or "").strip())
    if r.returncode != 0:
        raise PrepareError(f"export failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()[-600:]}")
    have = sound_data_sha(out_dir)
    if not have:
        raise PrepareError(f"{out_dir} does not load or re-hash after the export; see the loader output "
                           f"(python -m nig_timenet_torch.data_io.artifact {out_dir} --verify-hashes)")
    _write_sidecar(out_dir, fp, key, have)
    return _result("exported", out_dir, have)


def _result(action: str, out_dir, data_sha: str) -> dict:
    m = artifact.read_manifest(out_dir)
    summary = (f"{action}: {len(m.coins)} coins, source {m.source_commit[:10]}, "
               f"max_bar_time {m.max_bar_time.isoformat()}, sealed excluded {len(m.sealed_excluded)}, "
               f"data_sha256 {data_sha[:12]} (hashes verified)")
    return {"action": action, "data_sha256": data_sha, "path": str(out_dir), "summary": summary}

