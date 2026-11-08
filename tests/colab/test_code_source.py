"""colab/bootstrap.py, code source: Drive copy vs GitHub clone, updating the Drive copy without losing local edits,
and the token staying out of every output and out of ``.git/config``. Local ``file://`` remotes only."""

import json
import os

import pytest

from nig_timenet_torch.colab import bootstrap as bs
from tests.colab.helpers import git

TOKEN = "ghp_abcdefghijklmnop1234"


def make_remote(base):
    """-> (file:// url of a bare repo with branch main and tag v1, work dir to add commits from)."""
    work = base / "w"
    work.mkdir()
    git("init", "-q", "-b", "main", cwd=work)
    (work / "pyproject.toml").write_text("[project]\nname='x'\n")
    (work / "a.txt").write_text("one\n")
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "c1", cwd=work)
    git("tag", "v1", cwd=work)
    bare = base / "remote.git"
    git("clone", "-q", "--bare", str(work), str(bare), cwd=base)
    git("remote", "add", "origin", f"file://{bare}", cwd=work)
    return f"file://{bare}", work


def advance(work, text):
    (work / "a.txt").write_text(text)
    git("commit", "-qam", text.strip(), cwd=work)
    git("push", "-q", "origin", "main", cwd=work)
    return git("rev-parse", "HEAD", cwd=work)


def call(url, drive, base, **kw):
    out = []
    args = dict(token=TOKEN, url=url, ref="main", drive_dir=str(drive), workdir=str(base / "wd"), echo=out.append)
    args.update(kw)
    return bs.update_drive_copy(**args), out


def test_resolve_code_source_modes(tmp_path):
    drive, local = tmp_path / "drive", str(tmp_path / "local")
    assert bs.resolve_code_source("auto", str(drive), local) == {"resolved": "github", "dir": local,
                                                                 "has_drive_copy": False}
    with pytest.raises(bs.BootstrapError, match="no project copy"):
        bs.resolve_code_source("drive", str(drive), local)
    with pytest.raises(bs.BootstrapError, match="REPO_SOURCE"):
        bs.resolve_code_source("gihub", str(drive), local)
    drive.mkdir()
    (drive / "pyproject.toml").write_text("x")
    assert bs.resolve_code_source("auto", str(drive), local)["resolved"] == "drive"
    assert bs.resolve_code_source("drive", str(drive), local)["dir"] == str(drive)
    assert bs.resolve_code_source("github", str(drive), local)["dir"] == local


def test_update_drive_copy_clones_then_updates_and_the_token_is_nowhere(tmp_path):
    url, work = make_remote(tmp_path)
    drive = tmp_path / "drive" / "proj"
    info, out = call(url, drive, tmp_path)
    assert info["action"] == "cloned" and (drive / "a.txt").read_text() == "one\n"
    sha2 = advance(work, "two\n")
    info, out = call(url, drive, tmp_path)
    assert info == {"sha": sha2, "action": "updated"} and (drive / "a.txt").read_text() == "two\n"
    assert call(url, drive, tmp_path)[0]["action"] == "unchanged"
    assert json.loads((drive / bs.SYNC_MARK).read_text())["sha"] == sha2
    for f in drive.joinpath(".git").rglob("*"):
        if f.is_file():
            assert TOKEN.encode() not in f.read_bytes(), f
    assert TOKEN not in "\n".join(out)
    info, _ = call(url, drive, tmp_path, ref="v1")  # a tag works too, and going back is allowed
    assert (drive / "a.txt").read_text() == "one\n"


def test_update_refuses_local_edits_and_local_commits_unless_told_to_discard(tmp_path):
    url, work = make_remote(tmp_path)
    drive = tmp_path / "drive"
    call(url, drive, tmp_path)
    advance(work, "two\n")
    (drive / "a.txt").write_text("my edit\n")
    with pytest.raises(bs.LocalEditsError, match="a.txt"):
        call(url, drive, tmp_path)
    assert (drive / "a.txt").read_text() == "my edit\n"  # untouched by the refusal
    info, _ = call(url, drive, tmp_path, discard_local=True)
    assert info["action"] == "updated" and (drive / "a.txt").read_text() == "two\n"
    (drive / "b.txt").write_text("mine")  # a commit made on the Drive copy
    git("add", "b.txt", cwd=drive)
    git("commit", "-qm", "local", cwd=drive)
    with pytest.raises(bs.LocalEditsError, match="not the synced commit"):
        call(url, drive, tmp_path)
    assert (drive / "b.txt").exists()
    call(url, drive, tmp_path, discard_local=True)
    assert not (drive / "b.txt").exists()  # the local commit is gone from the working tree
    assert git("rev-parse", "HEAD", cwd=drive) == json.loads((drive / bs.SYNC_MARK).read_text())["sha"]


def test_update_refuses_a_non_git_folder_without_touching_it(tmp_path):
    url, _ = make_remote(tmp_path)
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "mine.txt").write_text("hand copied")
    with pytest.raises(bs.BootstrapError, match="not a git copy"):
        call(url, drive, tmp_path)
    assert (drive / "mine.txt").read_text() == "hand copied"


def test_load_code_github_drive_and_the_token_rule(tmp_path):
    url, work = make_remote(tmp_path)
    drive, local, out = tmp_path / "drive", tmp_path / "local", []
    kw = dict(repo="o/r", ref="main", drive_dir=str(drive), local_dir=str(local), workdir=str(tmp_path / "wd"),
              remote_url=url, echo=out.append)
    with pytest.raises(bs.BootstrapError, match="token is needed"):
        bs.load_code("github", token="", **kw)
    got = bs.load_code("auto", token=TOKEN, update_drive=True, **kw)  # no Drive copy yet: GitHub + creates the copy
    assert got["resolved"] == "github" and got["dir"] == str(local) and got["drive_copy"]["action"] == "cloned"
    assert (local / "a.txt").exists() and (drive / "a.txt").exists()
    sha2 = advance(work, "two\n")
    got = bs.load_code("drive", token="", **kw)  # runs from the Drive copy with no token and no network
    assert got["resolved"] == "drive" and got["dir"] == str(drive) and got["sha"] != sha2
    assert (drive / "a.txt").read_text() == "one\n"
    got = bs.load_code("drive", token=TOKEN, update_drive=True, **kw)
    assert got["sha"] == sha2 and (drive / "a.txt").read_text() == "two\n"
    assert TOKEN not in "\n".join(out)


def test_find_token_order_and_acquire_token_preset(tmp_path):
    assert bs.find_token("S", {"GITHUB_TOKEN": " t1 "}, lambda n: "t2") == "t1"
    assert bs.find_token("S", {}, lambda n: "t2") == "t2"
    assert bs.find_token("S", {}, lambda n: 1 / 0) == "" and bs.find_token("S", {}) == ""
    p = str(tmp_path / "tok.txt")
    never = lambda: 1 / 0  # noqa: E731
    assert bs.acquire_token(p, "o/r", never, accepted=lambda t, r: True, preset=TOKEN, save=False) == TOKEN
    assert not os.path.exists(p)  # a Colab secret is never copied to Drive
    assert bs.acquire_token(p, "o/r", never, accepted=lambda t, r: True, preset=TOKEN) == TOKEN
    assert bs.read_token(p) == TOKEN  # a typed token is kept for the next session
    os.remove(p)
    ask = lambda: TOKEN  # noqa: E731
    assert bs.acquire_token(p, "o/r", ask, accepted=lambda t, r: True, save=False) == TOKEN
    assert not os.path.exists(p)
