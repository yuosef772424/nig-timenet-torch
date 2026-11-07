"""Stdlib-only git and token kit of the Colab notebook (ROADMAP T3): secrets masking, a GitHub token kept on Drive,
a CLEAN clone (``rm -rf`` first, then ``git clone``, then the exact ref) of every repo the run needs, and the
``--no-deps`` install that leaves Colab's own CUDA torch alone.

The notebook's first cell downloads THIS ONE FILE with the token and ``exec``s it, because nothing else of the package
exists on the VM yet; every other module imports it normally. It therefore imports nothing from the package and
nothing outside the standard library (``tests/colab/test_bootstrap.py`` pins both).

Why a clean clone: ``/content/<dir>`` can already exist from an earlier run on the same VM (the owner hit this with
``/content/tf``); ``git clone`` into it fails and a ``git pull`` there could keep stale or edited files.

The token only ever lives in the environment of git child processes: a ``GIT_ASKPASS`` script (rewritten on every
``git_env`` call, in a directory the caller owns, never ``/tmp``: rule E2) prints it from ``$COLAB_RUNNER_TOKEN``.
It is never in a URL, an argv or ``.git/config``, and every printed string goes through ``mask``.
"""

import base64
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

SHA40 = re.compile(r"[0-9a-fA-F]{40}")


class BootstrapError(Exception):
    """Something the bootstrap refuses or cannot do; the message is already safe to print."""


# ------------------------------------------------------------------------------------------------ masking
def _variants(secret: str) -> set[str]:
    forms = {secret, urllib.parse.quote(secret), urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret)}
    for raw in (secret, "x-access-token:" + secret):
        b = base64.b64encode(raw.encode()).decode()
        forms.update({b, b.rstrip("=")})
    return {f for f in forms if f}


class Secrets:
    """The secret values of one run. ``mask`` hides every form of each (plain, url-encoded, basic-auth base64)."""

    def __init__(self, *values: str):
        self._values: list[str] = []
        for v in values:
            self.add(v)

    def add(self, value) -> None:
        if isinstance(value, str) and len(value) >= 6 and value not in self._values:
            self._values.append(value)

    def mask(self, text) -> str:
        forms: set[str] = set()
        for v in self._values:
            forms |= _variants(v)
        out = str(text)
        for f in sorted(forms, key=len, reverse=True):
            out = out.replace(f, "***")
        return out


# ------------------------------------------------------------------------------------------------ the token
def read_token(path: str) -> str:
    """The one-line token file's content. Errors never echo the content; a missing file gives ``""``."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return ""
    ok = len(lines) == 1 and len(lines[0]) >= 8 and not any(c.isspace() for c in lines[0])
    return lines[0] if ok else ""


def save_token(path: str, token: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def token_accepted(token: str, repo: str, urlopen=urllib.request.urlopen) -> bool:
    """True if GitHub accepts the token for ``repo`` (one GET, token in a header only); False on 401/404; anything
    else (network, 5xx) stops the run instead of being read as 'bad token'."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}", headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "nig-colab"})
    try:
        with urlopen(req, timeout=60):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            return False
        raise BootstrapError(f"GitHub answered HTTP {e.code} while checking the token; try again later") from None
    except Exception as e:  # noqa: BLE001 - never echo the request (it holds the token)
        raise BootstrapError(f"cannot reach GitHub to check the token ({type(e).__name__})") from None


def acquire_token(path: str, repo: str, ask, accepted=token_accepted, attempts: int = 3) -> str:
    """The saved token if GitHub still accepts it, else ask (``ask()`` -> str) up to ``attempts`` times and save the
    first accepted one. Raises when none is accepted."""
    saved = read_token(path)
    if saved and accepted(saved, repo):
        return saved
    for _ in range(attempts):
        new = (ask() or "").strip()
        if len(new) >= 8 and not any(c.isspace() for c in new) and accepted(new, repo):
            save_token(path, new)
            return new
    raise BootstrapError(f"no GitHub token accepted for {repo} after {attempts} attempts")


# ------------------------------------------------------------------------------------------------ git
def git_env(token: str, askpass_dir: str) -> dict:
    """Environment for git child processes. The askpass script is (re)written here on every call, so a script that
    vanished (a cleaned temp dir, a restarted VM) is never what a later push depends on."""
    os.makedirs(askpass_dir, exist_ok=True)
    script = os.path.join(askpass_dir, "askpass.sh")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write('#!/bin/sh\ncase "$1" in\n  Username*) printf \'%s\\n\' "x-access-token" ;;\n'
                 '  *) printf \'%s\\n\' "$COLAB_RUNNER_TOKEN" ;;\nesac\n')
    os.chmod(script, 0o700)
    env = dict(os.environ)
    env.update({"GIT_ASKPASS": script, "GIT_TERMINAL_PROMPT": "0", "COLAB_RUNNER_TOKEN": token,
                "GCM_INTERACTIVE": "never"})
    return env


def run_git(args, cwd=None, env=None, secrets: Secrets | None = None, timeout: int = 900) -> str:
    env = env if env is not None else dict(os.environ, GIT_TERMINAL_PROMPT="0")
    i = 0
    while i < len(args) and args[i] == "-c":
        i += 2
    label = args[i] if i < len(args) else "git"
    try:
        r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise BootstrapError(f"git {label} timed out after {timeout} s") from None
    except OSError as e:
        raise BootstrapError(f"git is not usable ({type(e).__name__})") from None
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-600:]
        raise BootstrapError(f"git {label} failed (exit {r.returncode}): {secrets.mask(tail) if secrets else tail}")
    return r.stdout.strip()


def clean_clone(url: str, dest: str, ref: str, env=None, secrets: Secrets | None = None,
                exact_sha: bool = False) -> str:
    """``rm -rf dest``, clone ``url`` and check out ``ref`` (a branch name, or a 40-hex commit); returns HEAD's sha.
    ``exact_sha=True`` refuses anything but a full commit sha and verifies that HEAD landed on it."""
    is_sha = bool(SHA40.fullmatch(ref))
    if exact_sha and not is_sha:
        raise BootstrapError("ref must be a full 40-hex commit sha")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    run_git(["clone", "--quiet", "--no-checkout", url, dest], env=env, secrets=secrets)
    target = ref.lower() if is_sha else f"origin/{ref}"
    try:
        run_git(["checkout", "--quiet", "--detach", target], cwd=dest, env=env, secrets=secrets)
    except BootstrapError:
        if not is_sha:
            raise BootstrapError(f"branch {ref} not found in the repository") from None
        try:  # a commit no branch reaches: ask the server for it directly
            run_git(["fetch", "--quiet", "origin", target], cwd=dest, env=env, secrets=secrets)
            run_git(["checkout", "--quiet", "--detach", target], cwd=dest, env=env, secrets=secrets)
        except BootstrapError:
            raise BootstrapError(f"checkout of {target} failed: commit not found in the repository") from None
    head = run_git(["rev-parse", "HEAD"], cwd=dest, env=env, secrets=secrets)
    if is_sha and head != target:
        raise BootstrapError(f"checkout landed on {head}, expected {target}")
    return head


def setup_clones(token: str, workdir: str, repos: dict, echo=print) -> dict:
    """Clean-clone every repo and print the commit of each. ``repos`` = {name: (``owner/name``, ref, dest dir)};
    returns {name: head sha}. Uses a fresh git environment, so nothing from an earlier run on this VM is reused."""
    secrets = Secrets(token)
    env = git_env(token, os.path.join(workdir, "git_askpass"))
    shas = {}
    for name, (repo, ref, dest) in repos.items():
        shas[name] = clean_clone(f"https://github.com/{repo}.git", dest, ref, env=env, secrets=secrets)
        echo(f"{name}: {repo} @ {ref} -> {shas[name]} ({dest})")
    return shas


def purge_package(repo_dir: str) -> dict:
    """Drop every in-memory module of this repo's packages and their stale ``src`` paths. A reused
    Colab runtime keeps code from an EARLIER checkout alive in ``sys.modules`` - the classic
    'cannot import name ... from' ImportError when a fresh clone adds names - and an old checkout's
    ``src`` entry in ``sys.path`` can shadow the fresh clone. Returns what was dropped."""
    src = os.path.abspath(os.path.join(repo_dir, "src"))
    pkgs = ([d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and not d.endswith(".egg-info")]
            if os.path.isdir(src) else [])
    gone = [m for m in list(sys.modules) if any(m == p or m.startswith(p + ".") for p in pkgs)]
    for m in gone:
        del sys.modules[m]
    repo = os.path.basename(os.path.abspath(repo_dir))
    stale = []
    for p in [q for q in list(sys.path) if q]:
        parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
        if os.path.basename(os.path.abspath(p)) == "src" and parent != repo and parent.startswith(repo):
            stale.append(p)
    for p in stale:
        sys.path.remove(p)
    return {"purged_modules": gone, "purged_paths": stale}


def _installed_version(dist: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def install_package(repo_dir: str, python: str | None = None, run=subprocess.run, version_of=_installed_version,
                    modules: tuple = (("pyarrow", "pyarrow"),)) -> dict:
    """``pip install --no-deps -e <repo_dir>`` (pyproject pins the CI torch, which would replace Colab's CUDA build),
    then install only the third-party ``modules`` ((import name, pip name) pairs) that are missing. Refuses if the
    installed torch version differs from before: returns {"torch_before", "torch_after"} (equal) otherwise."""
    python = python or sys.executable
    before = version_of("torch")
    cmds = [[python, "-m", "pip", "install", "-q", "--no-deps", "-e", repo_dir]]
    cmds += [[python, "-m", "pip", "install", "-q", pip_name] for mod, pip_name in modules
             if importlib.util.find_spec(mod) is None]
    for cmd in cmds:
        r = run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise BootstrapError(f"{' '.join(cmd[2:5])} failed (exit {r.returncode}): {(r.stderr or r.stdout)[-400:]}")
    after = version_of("torch")
    if after != before:
        raise BootstrapError(f"the install changed the installed torch from {before} to {after}: Colab's CUDA build "
                             f"must stay; use a fresh runtime and find what pulled torch in")
    src = os.path.join(repo_dir, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return {"torch_before": before, "torch_after": after, **purge_package(repo_dir)}
