"""T2 (rule D1): the artifact loader refuses a directory without the TF export fields, or with inconsistent ones.

The artifact under test is the committed 6-coin fixture with an ``export`` block added to its manifest the way the TF
tool writes it (docs/DATA_ARTIFACT_CONTRACT.md, "Export manifest")."""

import dataclasses
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import artifact, npy_dir
from nig_timenet_torch.data_io.artifact import ArtifactError, load_artifact, read_manifest, verify_hashes
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.identity import ResumeMismatchError, SeedMismatchError, build_run_identity
from nig_timenet_torch.trainer.trainer import FitSettings
from tests.contract_doc import DATASET_DIR
from tests.trainer.fixture_setup import load_fixture
from tests.trainer.test_fit import fit, make_trainer

COMMIT = "a" * 40


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_block(path) -> dict:
    """What tools/export_torch_artifact.py would write for ``path`` (computed from the data, not copied from it)."""
    d = npy_dir.load(path)
    names = sorted(json.loads((path / "manifest.json").read_text())["arrays"])
    files = {f"{k}.npy": _sha(path / f"{k}.npy") for k in names}
    files["meta.pkl"] = _sha(path / "meta.pkl")
    last = pd.Timestamp(int(d["last_candles"][:, TS_COL].astype("int64").max()) + 3600 * 10**9, tz="UTC")
    return {"schema": artifact.SCHEMA, "source_commit": COMMIT, "coins": [b["name"] for b in d["asset_bounds"]],
            "sealed_excluded": ["SEALAUSDT", "SEALBUSDT"], "sealed_found": [], "time_seal": "2026-10-01T00:00:00+00:00",
            "max_bar_time": last.isoformat(), "files": files,
            "data_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def write_manifest(path, **changes) -> dict:
    man = json.loads((path / "manifest.json").read_text())
    man["export"] = {**export_block(path), **changes}
    for k in [k for k, v in man["export"].items() if v is ...]:
        del man["export"][k]                      # `field=...` removes the field
    (path / "manifest.json").write_text(json.dumps(man))
    return man["export"]


@pytest.fixture()
def art(tmp_path):
    dest = tmp_path / "artifact.dataset"
    shutil.copytree(DATASET_DIR, dest)
    write_manifest(dest)
    return dest


def test_a_complete_artifact_loads(art):
    a = load_artifact(art)
    m = a.manifest
    assert m.source_commit == COMMIT and m.coins == tuple(f"C{i}USDT" for i in range(6))
    assert m.sealed_excluded == ("SEALAUSDT", "SEALBUSDT") and m.time_seal == artifact.TIME_SEAL
    assert m.max_bar_time < artifact.TIME_SEAL and len(m.data_fingerprint) == 64
    assert set(a.dataset) == set(npy_dir.load(DATASET_DIR))
    assert a.dataset["last_candles"].shape[0] == 1227


def test_the_plain_fixture_without_an_export_block_is_refused():
    with pytest.raises(ArtifactError, match="no 'export' block"):
        load_artifact(DATASET_DIR)


@pytest.mark.parametrize("field", ["source_commit", "coins", "sealed_excluded", "max_bar_time", "time_seal", "schema",
                                   "data_sha256"])
def test_each_missing_required_field_is_refused(art, field):
    write_manifest(art, **{field: ...})
    with pytest.raises(ArtifactError, match=f"lacks required fields.*{field}"):
        load_artifact(art)


@pytest.mark.parametrize("changes, match", [
    ({"source_commit": "abc"}, "40-hex"),
    ({"source_commit": "A" * 40}, "40-hex"),
    ({"coins": []}, "non-empty list"),
    ({"sealed_excluded": []}, "non-empty list"),
    ({"sealed_excluded": ["X", "x"]}, "duplicate"),
    ({"time_seal": "2026-09-01"}, "time_seal"),
    ({"max_bar_time": "2026-10-01T00:00:00+00:00"}, "not before the time seal"),
    ({"max_bar_time": "2027-01-01"}, "not before the time seal"),
    ({"max_bar_time": "not a time"}, "not a timestamp"),
    ({"schema": "torch_artifact/2"}, "export.schema"),
    ({"data_sha256": "z" * 64}, "64-hex"),
    ({"files": {"x.npy": "short"}}, "export.files"),
])
def test_invalid_field_values_are_refused(art, changes, match):
    write_manifest(art, **changes)
    with pytest.raises(ArtifactError, match=match):
        load_artifact(art)


def test_a_sealed_coin_in_the_coin_list_is_refused_with_or_without_usdt_suffix(art):
    write_manifest(art, sealed_excluded=["c3"])
    with pytest.raises(ArtifactError, match="C3USDT.*sealed"):
        load_artifact(art)
    write_manifest(art, sealed_excluded=["C0USDT"])
    with pytest.raises(ArtifactError, match="C0USDT.*sealed"):
        load_artifact(art)


def test_coin_list_must_describe_the_data(art):
    write_manifest(art, coins=["C0USDT", "C1USDT"])
    with pytest.raises(ArtifactError, match="differ from the dataset's asset_bounds"):
        load_artifact(art)


def test_manifest_cannot_understate_the_data_period(art):
    write_manifest(art, max_bar_time="2020-01-01T00:00:00+00:00")
    with pytest.raises(ArtifactError, match="later than export.max_bar_time"):
        load_artifact(art)


def test_data_that_reaches_the_seal_is_refused_even_if_the_manifest_claims_otherwise(art):
    lc = np.load(art / "last_candles.npy")
    lc[-1, TS_COL] = float(artifact.TIME_SEAL.value)          # the last window's bar is AT the seal
    np.save(art / "last_candles.npy", lc)
    write_manifest(art, max_bar_time="2026-09-30T23:00:00+00:00")
    with pytest.raises(ArtifactError, match="at or after the time seal"):
        load_artifact(art)


def test_read_manifest_does_not_need_the_arrays(art):
    for f in art.glob("*.npy"):
        f.unlink()
    assert read_manifest(art).source_commit == COMMIT


def test_verify_hashes_detects_a_changed_file(art):
    verify_hashes(art)
    np.save(art / "y_high_reg.npy", np.load(art / "y_high_reg.npy") + 1)
    with pytest.raises(ArtifactError, match="y_high_reg.npy does not match"):
        verify_hashes(art)


def test_verify_hashes_needs_the_files_block(art):
    write_manifest(art, files=None)
    with pytest.raises(ArtifactError, match="export.files is missing"):
        verify_hashes(art)


def test_data_sha256_must_match_files(art):
    write_manifest(art, data_sha256="b" * 64)
    with pytest.raises(ArtifactError, match="data_sha256 does not match"):
        verify_hashes(art)


def test_cli_ok_and_refusal(art, capsys):
    assert artifact.main([str(art), "--verify-hashes"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK") and "hashes verified" in out
    assert artifact.main([str(DATASET_DIR)]) == 1
    assert "artifact refused" in capsys.readouterr().err


def test_artifact_to_identity_to_fit(art, tmp_path):
    """T2 end to end on the fixture: load the artifact, build the identity from its manifest, train one epoch."""
    a = load_artifact(art)
    settings = FitSettings(epochs=1, batch_size=32, seed=4)
    ident = build_run_identity(settings, ModelSettings(), a.manifest.data_fingerprint, code_sha=COMMIT)
    assert ident.data_fingerprint == a.manifest.data_fingerprint and ident.seed == 4
    x, y = load_fixture(96 + 48)
    data = (x[:96], {k: v[:96] for k, v in y.items()}, x[96:], {k: v[96:] for k, v in y.items()})
    assert fit(make_trainer(), data, settings, run_dir=tmp_path / "run", identity=ident).epochs_done == 1
    with pytest.raises(SeedMismatchError):
        fit(make_trainer(), data, dataclasses.replace(settings, seed=5), run_dir=tmp_path / "run", identity=ident)
    other_data = dataclasses.replace(ident, data_fingerprint="0" * 64)
    with pytest.raises(ResumeMismatchError, match="data_fingerprint"):
        fit(make_trainer(), data, settings, run_dir=tmp_path / "run", identity=other_data)
