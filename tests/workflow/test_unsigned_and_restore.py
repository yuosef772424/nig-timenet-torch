"""Owner 2026-11-23: a run on an old-project pickle (derived manifest) goes through prepare -> build -> train, and
``restore`` says why it found nothing, suggests a run folder of a similar name and can read another run's folder."""

import gzip
import pickle
import shutil

import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings
from tests.contract_doc import DATASET_DIR
from tests.data_io.helpers import art  # noqa: F401  (fixture)

SMALL = {"d_model": 16, "num_layers": 1, "head_hidden": 16, "class_head_hidden": 8}


def _settings(data_name, run_dir, **over):
    base = RunSettings(run_dir=str(run_dir), data=LoaderSettings(name=str(data_name)))
    return updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "split.max_rows": 60,
                          "model.overrides": SMALL, **over})


def test_a_run_builds_on_an_old_project_pickle_and_its_identity_differs_from_the_signed_artifact(art, tmp_path):  # noqa: F811
    pkl = tmp_path / "crypto_model_latest.pkl.gz"
    with gzip.open(pkl, "wb") as f:
        pickle.dump(npy_dir.load(DATASET_DIR), f)
    old = steps.build(steps.prepare(_settings(pkl, tmp_path / "r1", **{"data.allow_unchecked": True})))
    signed = steps.build(steps.prepare(_settings(art, tmp_path / "r2")))
    assert old.art.manifest.derived and old.art.format == "pkl.gz" and not signed.art.manifest.derived
    assert old.ident.data_fingerprint.startswith("unsigned:") and old.ident.fingerprint() != signed.ident.fingerprint()
    assert old.x["train"].shape == signed.x["train"].shape                  # the same arrays, the same model input
    with pytest.raises(Exception, match="allow_unchecked=True"):            # still refused without the flag
        steps.prepare(_settings(pkl, tmp_path / "r3"))


def test_restore_tells_a_missing_folder_from_an_empty_one_and_suggests_the_run_you_meant(art, tmp_path):  # noqa: F811
    root = tmp_path / "runs"
    (root / "notebook-main5" / "train").mkdir(parents=True)
    (root / "notebook-main5" / "train" / "checkpoint.pt").write_bytes(b"x")       # a finished run
    (root / "other-run" / "train").mkdir(parents=True)
    run = steps.build(steps.prepare(_settings(art, root / "notebook-main56" / "train")))   # the typo
    with pytest.raises(FileNotFoundError) as e:
        steps.restore(run)
    msg = str(e.value)
    assert "does not exist" in msg and "typo" in msg and "did you mean" in msg and "notebook-main5" in msg
    assert "other-run" not in msg
    (root / "notebook-main56" / "train").mkdir(parents=True)
    (root / "notebook-main56" / "train" / "notes.txt").write_text("x")
    with pytest.raises(FileNotFoundError) as e2:
        steps.restore(run)
    assert "exists but holds no checkpoint.pt" in str(e2.value) and "notes.txt" in str(e2.value)
    assert "best.pt" in str(e2.value)                                       # says what a run folder holds


def test_restore_weights_is_never_a_path_and_from_dir_reads_another_runs_folder(art, tmp_path):  # noqa: F811
    s1 = _settings(art, tmp_path / "first" / "train")
    trained = steps.run_all(s1, tmp_path / "first" / "out")
    other = steps.build(steps.prepare(_settings(art, tmp_path / "second" / "train")))
    with pytest.raises(ValueError, match="pass from_dir="):
        steps.restore(other, weights=str(tmp_path / "first" / "train"))
    with pytest.raises(ValueError, match="must be one of"):
        steps.restore(other, weights="middle")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        steps.restore(other)                                                # its own folder has nothing
    got = steps.restore(other, from_dir=tmp_path / "first" / "train")
    assert got.result.best_epoch == trained.result.best_epoch and got.result.restored_best
    shutil.rmtree(tmp_path / "second", ignore_errors=True)
