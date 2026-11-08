"""colab/drive_sync.py with a fake Drive (a tmp dir): where a run works, push, restore, and the refusals."""

from pathlib import Path

import pytest

from nig_timenet_torch.colab import drive_sync as ds


def make_run(root: Path, name="r1", ckpt=b"weights-1"):
    files = (("train", "best.pt", ckpt), ("train", "resume.json", b"{}"), ("out", "test_report.json", b"{}"))
    for sub, fn, data in files:
        p = root / name / sub / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root / name


def test_run_paths_follow_the_work_on_switch(tmp_path):
    d = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "drive")
    assert d.run_dir == str(tmp_path / "drive" / "r1" / "train") and d.out_dir.endswith("r1/out")
    loc = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    assert loc.run_dir == str(tmp_path / "local" / "r1" / "train") and loc.drive_dir == str(tmp_path / "drive" / "r1")
    for bad in ("", "a/b", ".."):
        with pytest.raises(ds.SyncError):
            ds.run_paths(bad, "d", "l")
    with pytest.raises(ds.SyncError, match="WORK_ON"):
        ds.run_paths("r1", "d", "l", "cloud")


def test_push_copies_the_working_folder_and_skips_unchanged_files(tmp_path):
    paths = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    make_run(tmp_path / "local")
    res = ds.push_run(paths, echo=lambda m: None)
    assert res["copied"] == 3 and (tmp_path / "drive/r1/train/best.pt").read_bytes() == b"weights-1"
    assert (tmp_path / "drive/r1" / ds.PUSHED).is_file()
    again = ds.push_run(paths, echo=lambda m: None)
    assert again["copied"] == 0 and again["skipped"] == 3
    (tmp_path / "local/r1/train/best.pt").write_bytes(b"weights-2")  # a later epoch replaces the older checkpoint
    assert ds.push_run(paths, echo=lambda m: None)["copied"] == 1
    assert (tmp_path / "drive/r1/train/best.pt").read_bytes() == b"weights-2"
    assert not list((tmp_path / "drive").rglob("*.tmp"))


def test_push_is_a_no_op_when_the_run_works_on_drive_and_refuses_a_missing_folder(tmp_path):
    on_drive = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "drive")
    assert ds.push_run(on_drive, echo=lambda m: None)["already_on_drive"] is True
    local = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    with pytest.raises(ds.SyncError, match="nothing to push"):
        ds.push_run(local, echo=lambda m: None)


def test_restore_brings_back_an_earlier_run_by_name_or_path(tmp_path):
    make_run(tmp_path / "drive", "old", b"old-weights")
    new = ds.run_paths("new", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    res = ds.restore_run("old", new, echo=lambda m: None)
    assert res["copied"] == 3 and (tmp_path / "local/new/train/best.pt").read_bytes() == b"old-weights"
    other = ds.run_paths("other", str(tmp_path / "drive"), str(tmp_path / "local2"), "local")
    ds.restore_run(str(tmp_path / "drive" / "old"), other, echo=lambda m: None)
    assert (tmp_path / "local2/other/out/test_report.json").is_file()
    assert ds.restore_run("", new)["copied"] == 0


def test_restore_refuses_to_replace_different_files_and_non_run_folders(tmp_path):
    make_run(tmp_path / "drive", "old", b"old-weights")
    make_run(tmp_path / "local", "new", b"my newer weights")
    new = ds.run_paths("new", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    with pytest.raises(ds.SyncError, match="different files"):
        ds.restore_run("old", new, echo=lambda m: None)
    assert (tmp_path / "local/new/train/best.pt").read_bytes() == b"my newer weights"  # nothing was written
    ds.restore_run("old", new, overwrite=True, echo=lambda m: None)
    assert (tmp_path / "local/new/train/best.pt").read_bytes() == b"old-weights"
    (tmp_path / "drive" / "empty").mkdir()
    with pytest.raises(ds.SyncError, match="not a run folder"):
        ds.restore_run("empty", new)


def test_restore_of_the_runs_own_folder_does_nothing(tmp_path):
    make_run(tmp_path / "drive", "r1")
    own = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "drive")
    assert ds.restore_run("r1", own, echo=lambda m: None)["copied"] == 0


def test_git_folders_and_temp_files_are_never_copied(tmp_path):
    make_run(tmp_path / "local")
    (tmp_path / "local/r1/.git").mkdir()
    (tmp_path / "local/r1/.git/config").write_text("secret")
    (tmp_path / "local/r1/train/x.pt.tmp").write_text("half")
    paths = ds.run_paths("r1", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    ds.push_run(paths, echo=lambda m: None)
    names = {p.name for p in (tmp_path / "drive").rglob("*") if p.is_file()}
    assert "config" not in names and "x.pt.tmp" not in names
