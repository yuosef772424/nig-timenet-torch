"""colab/bootstrap.py: masking, the token file, clean clones, the --no-deps install. Local git only."""

import ast
import os
import stat
import sys
import urllib.error

import pytest

from nig_timenet_torch.colab import bootstrap as bs
from tests.colab.helpers import git, make_origin
from tests.scan import PKG_DIR

TOKEN = "ghp_abcdefghijklmnop1234"


def test_bootstrap_imports_only_the_standard_library():
    """The notebook exec()s this one file before anything else exists on the VM."""
    tree = ast.parse((PKG_DIR / "colab" / "bootstrap.py").read_text(encoding="utf-8"))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names), sorted(mods - set(sys.stdlib_module_names))


def test_mask_hides_every_form_of_a_secret():
    import base64
    import urllib.parse

    s = bs.Secrets(TOKEN)
    b64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    for form in (TOKEN, urllib.parse.quote(TOKEN), b64, b64.rstrip("=")):
        assert TOKEN not in s.mask(f"before {form} after") and "***" in s.mask(f"before {form} after")
    assert s.mask("nothing secret") == "nothing secret"
    assert bs.Secrets("abc").mask("abc") == "abc"  # shorter than 6 chars: ignored, would mangle ordinary text


def test_token_file_roundtrip_and_permissions(tmp_path):
    p = tmp_path / "secrets" / "tok.txt"
    assert bs.read_token(str(p)) == ""
    bs.save_token(str(p), TOKEN)
    assert bs.read_token(str(p)) == TOKEN
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    p.write_text("two\nlines here\n")
    assert bs.read_token(str(p)) == ""
    p.write_text("short")
    assert bs.read_token(str(p)) == ""


def test_acquire_token_uses_saved_then_asks_and_saves(tmp_path):
    p = str(tmp_path / "tok.txt")
    asked = []
    ask = lambda: asked.append(1) or "wrong-token-value"  # noqa: E731
    with pytest.raises(bs.BootstrapError, match="no GitHub token accepted"):
        bs.acquire_token(p, "o/r", ask, accepted=lambda t, r: False, attempts=2)
    assert len(asked) == 2 and not os.path.exists(p)
    assert bs.acquire_token(p, "o/r", lambda: TOKEN, accepted=lambda t, r: t == TOKEN) == TOKEN
    assert bs.read_token(p) == TOKEN
    assert bs.acquire_token(p, "o/r", lambda: 1 / 0, accepted=lambda t, r: True) == TOKEN  # saved + accepted: no prompt
    p2 = str(tmp_path / "tok2.txt")
    bs.save_token(p2, TOKEN)  # saved token rejected (revoked): asks again
    got = bs.acquire_token(p2, "o/r", lambda: "another-good-token", accepted=lambda t, r: t != TOKEN)
    assert got == "another-good-token"


def test_token_accepted_distinguishes_rejection_from_outage():
    def opener(code):
        def f(req, timeout):
            assert TOKEN in req.headers["Authorization"] and TOKEN not in req.full_url
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
        return f

    assert bs.token_accepted(TOKEN, "o/r", opener(401)) is False
    assert bs.token_accepted(TOKEN, "o/r", opener(404)) is False
    with pytest.raises(bs.BootstrapError, match="HTTP 503"):
        bs.token_accepted(TOKEN, "o/r", opener(503))


def test_git_env_writes_the_askpass_script_without_the_token(tmp_path):
    d = tmp_path / "askpass"
    env = bs.git_env(TOKEN, str(d))
    script = d / "askpass.sh"
    assert TOKEN not in script.read_text() and env["COLAB_RUNNER_TOKEN"] == TOKEN
    assert os.access(script, os.X_OK)
    script.unlink()  # vanished (cleaned temp dir, restarted VM): the next call rebuilds it
    bs.git_env(TOKEN, str(d))
    assert script.is_file()


def test_clean_clone_replaces_an_existing_directory_and_prints_the_commit(tmp_path):
    url, sha = make_origin(tmp_path)
    dest = tmp_path / "content" / "tf"
    dest.mkdir(parents=True)
    (dest / "stale.txt").write_text("left over from an earlier run")
    assert bs.clean_clone(url, str(dest), "main") == sha
    assert not (dest / "stale.txt").exists() and (dest / "src" / "nig_timenet_torch").is_dir()
    assert bs.clean_clone(url, str(dest), sha, exact_sha=True) == sha
    with pytest.raises(bs.BootstrapError, match="40-hex"):
        bs.clean_clone(url, str(dest), "main", exact_sha=True)
    with pytest.raises(bs.BootstrapError, match="branch nope not found"):
        bs.clean_clone(url, str(dest), "nope")
    with pytest.raises(bs.BootstrapError, match="not found"):
        bs.clean_clone(url, str(dest), "0" * 40, exact_sha=True)


def test_setup_clones_clones_every_repo_cleanly_and_echoes_commits(tmp_path, monkeypatch):
    url, sha = make_origin(tmp_path)
    monkeypatch.setattr(bs, "clean_clone",
                        lambda u, dest, ref, **kw: git("rev-parse", "HEAD", cwd=_clone(tmp_path, url, dest)))
    out = []
    shas = bs.setup_clones(TOKEN, str(tmp_path / "work"), {"torch": ("o/torch", "main", str(tmp_path / "a")),
                                                          "tf": ("o/tf", "master", str(tmp_path / "b"))}, out.append)
    assert shas == {"torch": sha, "tf": sha}
    assert any("o/torch @ main -> " + sha in line for line in out)
    assert any("o/tf @ master -> " + sha in line for line in out)
    assert (tmp_path / "work" / "git_askpass" / "askpass.sh").is_file()


def _clone(tmp_path, url, dest):
    git("clone", "-q", url, dest, cwd=tmp_path)
    return dest


class Done:
    def __init__(self, code=0, err=""):
        self.returncode, self.stdout, self.stderr = code, "", err


def test_install_package_is_no_deps_and_refuses_a_changed_torch(tmp_path):
    seen = []
    ok = bs.install_package(str(tmp_path), python="py", run=lambda cmd, **k: seen.append(cmd) or Done(),
                            version_of=lambda d: "2.11.0+cu126", modules=())
    assert ok == {"torch_before": "2.11.0+cu126", "torch_after": "2.11.0+cu126",
                  "purged_modules": [], "purged_paths": []}
    assert seen == [["py", "-m", "pip", "install", "-q", "--no-deps", "-e", str(tmp_path)]]
    versions = iter(["2.11.0+cu126", "2.15.0+cpu"])
    with pytest.raises(bs.BootstrapError, match="changed the installed torch from 2.11.0\\+cu126 to 2.15.0\\+cpu"):
        bs.install_package(str(tmp_path), python="py", run=lambda cmd, **k: Done(), version_of=lambda d: next(versions),
                           modules=())
    with pytest.raises(bs.BootstrapError, match="failed"):
        bs.install_package(str(tmp_path), python="py", run=lambda cmd, **k: Done(1, "boom"), version_of=lambda d: "x",
                           modules=())


def test_install_package_installs_only_missing_modules(tmp_path):
    seen = []
    bs.install_package(str(tmp_path), python="py", run=lambda cmd, **k: seen.append(cmd) or Done(),
                       version_of=lambda d: "1", modules=(("json", "json-pip"), ("no_such_module_xyz", "pkg-xyz")))
    assert seen[1:] == [["py", "-m", "pip", "install", "-q", "pkg-xyz"]]


def test_purge_package_drops_stale_modules_and_paths(tmp_path):
    """The reused-Colab-runtime bug: an older checkout's code stays alive in sys.modules (and its src
    in sys.path), so a fresh clone's new names fail to import. purge_package must drop exactly that."""
    import types

    new = tmp_path / "nig-timenet-torch"
    (new / "src" / "pkg1").mkdir(parents=True)
    old = tmp_path / "nig-timenet-torch2"
    old_src = old / "src"
    old_src.mkdir(parents=True)
    sys.modules["pkg1"] = types.ModuleType("pkg1")
    sys.modules["pkg1.mod"] = types.ModuleType("pkg1.mod")
    sys.path.insert(0, str(old_src))
    out = bs.purge_package(str(new))
    assert "pkg1" not in sys.modules and "pkg1.mod" not in sys.modules
    assert str(old_src) not in sys.path
    assert out == {"purged_modules": ["pkg1", "pkg1.mod"], "purged_paths": [str(old_src)]}
