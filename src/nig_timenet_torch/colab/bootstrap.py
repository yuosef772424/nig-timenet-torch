"""PURPOSE: Stdlib-only git and token kit of the Colab notebook: secrets masking, a Drive-kept GitHub token, clean
    clones, the code-source choice (GitHub / Drive copy) and the --no-deps install that leaves Colab's torch alone.
TAGS: colab bootstrap, git clone, github token, code source, drive copy, pip install, purge package, stdlib only
PITFALLS: imports nothing from the package (the notebook execs this one file); the Drive copy is only ever fetched and
    checked out, never pushed (the token is read-only); update_drive_copy refuses local edits unless told to discard.

Stdlib-only git and token kit of the Colab notebook (ROADMAP T3): secrets masking, a GitHub token kept on Drive,
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
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["SHA40", "BootstrapError", "Secrets", "read_token", "save_token", "token_accepted", "acquire_token",
           "git_env", "run_git", "clean_clone", "setup_clones", "purge_package", "install_package",
           "CODE_SOURCES", "LocalEditsError", "find_token", "resolve_code_source", "update_drive_copy", "load_code"]

SHA40 = re.compile(r"[0-9a-fA-F]{40}")


CODE_SOURCES = ("auto", "github", "drive")
SYNC_MARK = ".drive_sync.json"          # in the Drive copy: the commit update_drive_copy last put there


class BootstrapError(Exception):
    """Something the bootstrap refuses or cannot do; the message is already safe to print."""


class LocalEditsError(BootstrapError):
    """The Drive copy holds edits or commits of its own; the update would overwrite them."""


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


def find_token(secret_name: str, environ=None, secret_getter=None) -> str:
    """A token that needs no prompt: ``$GITHUB_TOKEN``, then the Colab secret ``secret_name`` (``secret_getter(name)``;
    the notebook passes ``google.colab.userdata.get``). ``""`` when there is none. Never raises, never prints."""
    tok = (environ if environ is not None else os.environ).get("GITHUB_TOKEN", "")
    if not tok and secret_getter is not None:
        try:
            tok = secret_getter(secret_name) or ""
        except Exception:  # noqa: BLE001 - a missing secret / no notebook access is "no token", not an error
            tok = ""
    return tok.strip()


def acquire_token(path: str, repo: str, ask, accepted=token_accepted, attempts: int = 3, preset: str = "",
                  save: bool = True) -> str:
    """``preset`` (from ``find_token`` or typed) if GitHub accepts it, else the saved token if accepted, else ask
    (``ask()`` -> str) up to ``attempts`` times. A token that was not already the saved one is written to the token
    file only when ``save`` (pass ``save=False`` for a Colab secret: it must not be copied to Drive). Raises when none
    is accepted."""
    if preset and accepted(preset, repo):
        if save and read_token(path) != preset:
            save_token(path, preset)
        return preset
    saved = read_token(path)
    if saved and accepted(saved, repo):
        return saved
    for _ in range(attempts):
        new = (ask() or "").strip()
        if len(new) >= 8 and not any(c.isspace() for c in new) and accepted(new, repo):
            if save:
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


# ------------------------------------------------------------------------------------------------ code source
def resolve_code_source(mode: str, drive_dir: str, local_dir: str) -> dict:
    """Where the code runs from. ``github``: a clean clone in ``local_dir``; ``drive``: the Drive copy as it is;
    ``auto``: the Drive copy if one exists, else GitHub. -> {"resolved", "dir", "has_drive_copy"}."""
    if mode not in CODE_SOURCES:
        raise BootstrapError(f"REPO_SOURCE must be one of {CODE_SOURCES}, not {mode!r}")
    has = os.path.isfile(os.path.join(drive_dir, "pyproject.toml"))
    resolved = "drive" if mode == "drive" or (mode == "auto" and has) else "github"
    if resolved == "drive" and not has:
        raise BootstrapError(f"no project copy at {drive_dir}: run once with REPO_SOURCE='github' and "
                             f"UPDATE_DRIVE_COPY=True to create it")
    return {"resolved": resolved, "dir": drive_dir if resolved == "drive" else local_dir, "has_drive_copy": has}


def update_drive_copy(token: str, url: str, ref: str, drive_dir: str, workdir: str, discard_local: bool = False,
                      echo=print) -> dict:
    """Bring the Drive copy to ``ref`` (branch, tag or sha) of ``url``: clone if absent, else fetch + detached checkout.
    Refuses (LocalEditsError) when tracked files are edited or HEAD is not the commit this function put there last,
    unless ``discard_local``. Fetch/checkout only: nothing is ever pushed. -> {"sha", "action"}."""
    secrets, env = Secrets(token), git_env(token, os.path.join(workdir, "git_askpass"))
    mark = os.path.join(drive_dir, SYNC_MARK)
    if not os.path.isdir(os.path.join(drive_dir, ".git")):
        if os.path.isdir(drive_dir) and os.listdir(drive_dir):
            raise BootstrapError(f"{drive_dir} exists and is not a git copy; move it aside (it is not touched)")
        sha, action = clean_clone(url, drive_dir, ref, env=env, secrets=secrets), "cloned"
    else:
        head = run_git(["rev-parse", "HEAD"], cwd=drive_dir, env=env, secrets=secrets)
        dirty = run_git(["status", "--porcelain", "--untracked-files=no"], cwd=drive_dir, env=env, secrets=secrets)
        last = json.load(open(mark, encoding="utf-8")).get("sha") if os.path.isfile(mark) else None
        if (dirty or (last and last != head)) and not discard_local:
            why = f"edited files: {dirty!r}" if dirty else f"HEAD {head[:10]} is not the synced commit {last[:10]}"
            raise LocalEditsError(f"the Drive copy {drive_dir} has local changes ({why}); keep them "
                                  f"(copy them away) or set DISCARD_LOCAL_CHANGES = True to overwrite them")
        run_git(["fetch", "--quiet", url, ref], cwd=drive_dir, env=env, secrets=secrets)
        sha = run_git(["rev-parse", "FETCH_HEAD^{commit}"], cwd=drive_dir, env=env, secrets=secrets)
        action = "unchanged" if sha == head and not dirty else "updated"
        run_git(["checkout", "--quiet", "-f", "--detach", sha], cwd=drive_dir, env=env, secrets=secrets)
    run_git(["remote", "set-url", "origin", url], cwd=drive_dir, env=env, secrets=secrets)  # never holds the token
    with open(mark, "w", encoding="utf-8") as fh:
        json.dump({"sha": sha, "ref": ref}, fh)
    echo(f"Drive copy {action}: {ref} -> {sha} ({drive_dir})")
    return {"sha": sha, "action": action}


def load_code(mode: str, repo: str, ref: str, drive_dir: str, local_dir: str, workdir: str, token: str = "",
              update_drive: bool = False, discard_local: bool = False, remote_url: str | None = None,
              echo=print) -> dict:
    """The notebook's code-source step: optionally update the Drive copy from GitHub, then make ``dir`` ready to run
    from (a clean clone for ``github``, the Drive copy for ``drive``). A token is needed only when GitHub is touched.
    -> {"dir", "resolved", "sha", "drive_copy"}."""
    src = resolve_code_source(mode, drive_dir, local_dir)
    url = remote_url or f"https://github.com/{repo}.git"
    if (src["resolved"] == "github" or update_drive) and not token:
        raise BootstrapError("a GitHub token is needed to read GitHub (Colab secret GITHUB_TOKEN, the token file, or "
                             "the prompt); with a Drive copy and REPO_SOURCE='drive' no token is needed")
    drive_copy = update_drive_copy(token, url, ref, drive_dir, workdir, discard_local, echo) if update_drive else None
    if src["resolved"] == "github":
        env = git_env(token, os.path.join(workdir, "git_askpass"))
        sha = clean_clone(url, src["dir"], ref, env=env, secrets=Secrets(token))
        echo(f"code from GitHub: {repo} @ {ref} -> {sha} ({src['dir']})")
    else:
        mark = os.path.join(drive_dir, SYNC_MARK)
        sha = json.load(open(mark, encoding="utf-8")).get("sha", "") if os.path.isfile(mark) else ""
        echo(f"code from the Drive copy {drive_dir} (synced commit {sha or 'unknown'})")
    return {"dir": src["dir"], "resolved": src["resolved"], "sha": sha, "drive_copy": drive_copy}


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
