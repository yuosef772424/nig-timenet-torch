"""colab_ops: the recipes are pure text (no secret inside), the CLI parses, the bridge waits, refuses a missing URL."""

import subprocess
import sys

import pytest

from nig_timenet_torch.colab_ops import recipes
from nig_timenet_torch.colab_ops.__main__ import build_parser
from nig_timenet_torch.colab_ops.bridge import Bridge


def test_setup_script_is_valid_bash_idempotent_and_has_no_secret():
    s = recipes.setup_script("ccr-x/y z")
    assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0
    assert "'ccr-x/y z'" in s  # the branch is shell-quoted
    assert recipes.TOKEN_FILE in s and "x-access-token:%s" in s and "$T" in s
    assert "if [ -d" in s and "git fetch" in s and "git clone" in s  # update or clone
    assert "[ -d /content/art1 ] ||" in s  # the artifact copy is skipped when present
    assert "ensure_env('pyproject.toml', '/content/nig_venv', '/content/drive/MyDrive/crypto/nig_timenet_env'" in s
    assert "pip install" not in s  # Colab's torch is never replaced
    assert "ghp_" not in s and "github_pat" not in s


def test_setup_script_options():
    s = recipes.setup_script("b", artifact=False, install=False)
    assert "cp -r" not in s and "ensure_env" not in s


def test_env_script_keys_the_cache_on_the_requested_torch():
    e = recipes.env_script("/content/r", torch_key="2.14.1+cu130")
    assert subprocess.run(["bash", "-n"], input=e, text=True).returncode == 0
    assert "torch_version='2.14.1+cu130'" in e and "nig_env.sh" in e
    assert "torch_version=None" in recipes.env_script("/content/r")


def test_job_lines_load_the_cached_env_first():
    for line in (
        recipes.arms_line("/o", arms="gru:3e-4", rows=1, epochs=1, seed=0, device="cpu"),
        recipes.doctor_line("/r", "/o", encoder="gru", lr=1e-3, rows=1, epochs=1, device="cpu"),
    ):
        assert line.startswith(". /content/nig_env.sh && ")


def test_arms_and_doctor_lines():
    a = recipes.arms_line("/content/o", arms="gru:3e-4", rows=100, epochs=2, seed=1, device="cpu", n_variants=4)
    assert "--battery --n-variants 4" in a and "--device cpu" in a and "--seed 1" in a
    assert "--battery" not in recipes.arms_line(
        "/o", arms="gru:3e-4", rows=1, epochs=1, seed=0, device="cuda", battery=False
    )
    d = recipes.doctor_line("/r", "/o", encoder="transformer", lr=3e-4, rows=10, epochs=6, device="cuda")
    assert "--trained" in d and '"encoder": "transformer"' in d and "0.0003" in d


def test_cli_parses_every_command():
    p = build_parser()
    for argv in (
        ["rt"],
        ["exec", "ls"],
        ["start", "ls", "n"],
        ["wait", "p1", "--limit", "5"],
        ["out", "p1"],
        ["put", "a", "b"],
        ["setup", "br"],
        ["arms", "/o"],
        ["doctor", "/r", "/o"],
    ):
        assert p.parse_args(argv).cmd == argv[0]


def test_bridge_needs_url_and_token_and_waits(monkeypatch):
    with pytest.raises(ValueError):
        Bridge("", "")
    b = Bridge("https://x.trycloudflare.com", "t" * 40)
    seq = iter([{"status": "running"}, {"status": "running"}, {"status": "completed", "duration_seconds": 3}])
    monkeypatch.setattr(b, "status", lambda pid, wait=0.0: next(seq))
    assert b.wait("p", limit=100)["status"] == "completed"
    monkeypatch.setattr(b, "status", lambda pid, wait=0.0: {"status": "running"})
    assert b.wait("p", limit=0)["status"] == "running"  # gives up at the limit


def test_importing_the_package_does_not_need_colab_bridge():
    code = (
        "import sys; sys.modules['colab_bridge'] = None; import nig_timenet_torch.colab_ops.bridge, "
        "nig_timenet_torch.colab_ops.__main__"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
