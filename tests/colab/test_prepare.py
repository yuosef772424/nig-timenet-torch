"""colab/prepare.py: export once, verify with the project's loader, skip when sound and made from the same source."""

import shutil

import pytest

from nig_timenet_torch.colab import prepare
from tests.contract_doc import DATASET_DIR
from tests.data_io.helpers import write_manifest


class Done:
    def __init__(self, code=0, out="exported ok", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


@pytest.fixture()
def env(tmp_path):
    tf = tmp_path / "tf"
    (tf / "tools").mkdir(parents=True)
    (tf / "tools" / "export_torch_artifact.py").write_text("# tool v1\n")
    (tf / "docs" / "research" / "plan" / "lock").mkdir(parents=True)
    (tf / "docs" / "research" / "plan" / "lock" / "holdout.json").write_text('{"sealed": {}}')
    source = tmp_path / "preprocessing_output_latest.pkl.gz"
    source.write_bytes(b"pretend this is a few GB of pickle")
    out = tmp_path / "torch_artifacts" / "preprocessing_output_1"
    out.parent.mkdir()
    calls = []

    def run(cmd, cwd, capture_output, text):
        calls.append(cmd)
        assert cmd[2:4] == ["--dataset", str(source)] and cmd[4:] == ["--out", str(out)] and cwd == str(tf)
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(DATASET_DIR, out)
        write_manifest(out)
        return Done()

    return {"tf": tf, "source": source, "out": out, "run": run, "calls": calls}


def prep(e, **kw):
    return prepare.prepare_artifact(e["tf"], e["source"], e["out"], run=e["run"], echo=lambda *_: None, **kw)


def test_first_run_exports_verifies_and_records_the_source(env):
    r = prep(env)
    assert r["action"] == "exported" and len(env["calls"]) == 1 and "6 coins" in r["summary"]
    assert "hashes verified" in r["summary"] and prepare.sidecar_path(env["out"]).is_file()
    assert not (env["out"] / "source.json").exists()  # the record is NEXT to the artifact, not inside it


def test_second_run_skips_when_artifact_is_sound_and_source_is_the_same(env):
    prep(env)
    r = prep(env)
    assert r["action"] == "skipped" and len(env["calls"]) == 1


@pytest.mark.parametrize("what", ["source", "tool", "lock"])
def test_anything_that_decides_the_output_triggers_a_new_export(env, what):
    prep(env)
    path = {"source": env["source"], "tool": env["tf"] / "tools" / "export_torch_artifact.py",
            "lock": env["tf"] / "docs" / "research" / "plan" / "lock" / "holdout.json"}[what]
    path.write_bytes(path.read_bytes() + b"x")
    assert prep(env)["action"] == "exported" and len(env["calls"]) == 2


def test_a_corrupted_artifact_is_not_skipped(env):
    prep(env)
    npy = env["out"] / "X_1h.npy"
    raw = bytearray(npy.read_bytes())
    raw[-1] ^= 0xFF
    npy.write_bytes(bytes(raw))
    assert prep(env)["action"] == "exported" and len(env["calls"]) == 2


def test_a_half_written_artifact_is_not_skipped(env):
    prep(env)
    (env["out"] / "manifest.json").unlink()
    assert prep(env)["action"] == "exported"


def test_an_artifact_without_a_record_is_re_exported_unless_the_caller_adopts_it(env):
    prep(env)
    prepare.sidecar_path(env["out"]).unlink()  # e.g. the owner's hand-made artifact
    assert prep(env)["action"] == "exported" and len(env["calls"]) == 2
    prepare.sidecar_path(env["out"]).unlink()
    r = prep(env, adopt_existing=True)
    assert r["action"] == "adopted" and len(env["calls"]) == 2
    assert prep(env)["action"] == "skipped"  # the adoption wrote the record
    env["source"].write_bytes(b"other")
    assert prep(env, adopt_existing=True)["action"] == "exported"  # adoption never overrides a different source


def test_an_npy_dir_source_is_fingerprinted_by_its_manifest(env, tmp_path):
    src = tmp_path / "x.dataset"
    shutil.copytree(DATASET_DIR, src)
    fp = prepare.source_fingerprint(src)
    assert fp["kind"] == "npy_dir" and len(fp["manifest_sha256"]) == 64
    assert prepare.source_fingerprint(env["source"])["kind"] == "file"


def test_failures_are_reported_not_swallowed(env, tmp_path):
    with pytest.raises(prepare.PrepareError, match="does not exist"):
        prepare.prepare_artifact(env["tf"], tmp_path / "nope.pkl.gz", env["out"], run=env["run"])
    with pytest.raises(prepare.PrepareError, match=r"export failed \(exit 1\).*boom"):
        prepare.prepare_artifact(env["tf"], env["source"], env["out"], echo=lambda *_: None,
                                 run=lambda *a, **k: Done(1, "", "boom"))
    with pytest.raises(prepare.PrepareError, match="does not load or re-hash"):
        prepare.prepare_artifact(env["tf"], env["source"], tmp_path / "empty", echo=lambda *_: None,
                                 run=lambda *a, **k: Done(0))
