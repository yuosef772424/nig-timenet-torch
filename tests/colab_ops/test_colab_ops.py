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
    assert "ghp_" not in s and "github_pat" not in s


def test_setup_script_options():
    s = recipes.setup_script("b", artifact=False, install=False)
    assert "art1" not in s.replace("/content/art1", "") and "pip install" not in s


def test_arms_doctor_runs_lines():
    a = recipes.arms_line("/content/o", arms="gru:3e-4", rows=100, epochs=2, seed=1, device="cpu", n_variants=4)
    assert "--battery --n-variants 4" in a and "--device cpu" in a and "--seed 1" in a
    assert "--battery" not in recipes.arms_line(
        "/o", arms="gru:3e-4", rows=1, epochs=1, seed=0, device="cuda", battery=False
    )
    d = recipes.doctor_line("/r", "/o", encoder="transformer", lr=3e-4, rows=10, epochs=6, device="cuda")
    assert "--trained" in d and '"encoder": "transformer"' in d and "0.0003" in d
    r = recipes.runs_line("itransformer,conv1d_net", 1e-4, 1e-6, 10, 320000)
    assert r.endswith("itransformer,conv1d_net 0.0001 1e-06 10 320000 cuda") and "PYTHONPATH=/content/nig_uhub/src" in r


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
        ["runs", "gru"],
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
