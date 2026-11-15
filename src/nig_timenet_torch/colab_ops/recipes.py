"""PURPOSE: Pure builders of the shell text a Colab runtime runs: setup (code + deps + artifact copy), a job
    start line, the doctor line; no network, no secrets, testable on any machine.
TAGS: colab recipes, setup script, shell builders, git clone private, artifact copy, doctor command
PITFALLS: the GitHub token is read from a Drive file on the runtime (``TOKEN_FILE``) and handed to git through
    environment variables for ONE command (never written to .git/config); every builder is idempotent so a re-run after
    a bridge drop is safe; the torch pin follows pyproject (>=2.15,<2.16) and is installed only when the version
    differs.
"""

import shlex

__all__ = [
    "TOKEN_FILE",
    "ARTIFACT_DRIVE",
    "ARTIFACT_LOCAL",
    "REPO",
    "ENV_CACHE",
    "VENV",
    "ENV_SH",
    "setup_script",
    "env_script",
    "arms_line",
    "doctor_line",
    "runs_line",
]

REPO = "https://github.com/yuosef772424/nig-timenet-torch.git"
TOKEN_FILE = "/content/drive/MyDrive/crypto/secrets/github_token.txt"
ARTIFACT_DRIVE = "/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1"
ARTIFACT_LOCAL = "/content/art1"
ENV_CACHE = "/content/drive/MyDrive/nig_timenet_env"  # the notebook's ENV_CACHE_DIR: one env-<key>.tar lives here
VENV = "/content/nig_venv"
ENV_SH = "/content/nig_env.sh"  # `. /content/nig_env.sh` puts the cached env + the repo on PYTHONPATH
_GIT = (
    "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.https://github.com/.extraheader "
    'GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $B"'
)


def setup_script(
    branch: str,
    dest: str = "/content/nig-timenet-torch",
    *,
    artifact: bool = True,
    install: bool = True,
    torch_key: str | None = None,
) -> str:
    """Clone (or fast-forward) ``branch`` into ``dest``; restore the cached env from Drive (built once, see
    ``env_script``); copy the artifact to local disk (the Drive mmap is slow). Prints the head commit, the env source,
    torch version and the device. Colab's own torch is used (never installed)."""
    b, d = shlex.quote(branch), shlex.quote(dest)
    lines = [
        "set -e",
        f'T=$(tr -d "\\r\\n " < {TOKEN_FILE}); B=$(printf "x-access-token:%s" "$T" | base64 -w0)',
        f"if [ -d {d}/.git ]; then",
        f"  cd {d} && {_GIT} git fetch -q origin {b} && git checkout -q -B {b} FETCH_HEAD",
        "else",
        f"  {_GIT} git clone -q --branch {b} {REPO} {d}",
        "fi",
        f"git -C {d} log --oneline | head -1",
    ]
    if install:
        lines += [env_script(dest, torch_key=torch_key)]
    if artifact:
        lines += [f"[ -d {ARTIFACT_LOCAL} ] || cp -r {ARTIFACT_DRIVE} {ARTIFACT_LOCAL}"]
    lines += ['python -c "import torch;print(torch.__version__, torch.cuda.is_available())"']
    return "\n".join(lines)


def env_script(repo: str = "/content/nig-timenet-torch", *, torch_key: str | None = None) -> str:
    """Restore the project's env from the Drive cache (``ENV_CACHE``/env-<key>.tar) or build and cache it, then write
    ``ENV_SH``. ``torch_key``: the torch version the cache key is computed for (default: the runtime's torch); build
    with the version a FRESH Colab has (e.g. ``2.14.1+cu130``) so later sessions hit the cache."""
    tv = repr(torch_key)
    return "\n".join(
        [
            f"cd {shlex.quote(repo)} && PYTHONPATH={shlex.quote(repo)}/src python - <<'PY'",
            "from nig_timenet_torch.colab.env_cache import ensure_env",
            f"r = ensure_env('pyproject.toml', '{VENV}', '{ENV_CACHE}', torch_version={tv})",
            "print('env', r['source'], r['key'])",
            "PY",
            f"SP=$(ls -d {VENV}/lib/python*/site-packages)",
            f"printf 'export PYTHONPATH=%s:%s/src\\n' \"$SP\" {shlex.quote(repo)} > {ENV_SH}",
        ]
    )


def arms_line(
    out: str,
    *,
    arms: str,
    rows: int,
    epochs: int,
    seed: int,
    device: str,
    battery: bool = True,
    n_variants: int = 1,
    repo: str = "/content/nig-timenet-torch",
) -> str:
    """The small-sample encoder x lr arms (docs/audit/2026-11-14-model-first/quick_arms.py)."""
    flags = f" --battery --n-variants {int(n_variants)}" if battery else ""
    return (
        f". {ENV_SH} && cd {shlex.quote(repo)} && "
        f"python -u docs/audit/2026-11-14-model-first/quick_arms.py {ARTIFACT_LOCAL} "
        f"{shlex.quote(out)} --rows {int(rows)} --epochs {int(epochs)} --seed {int(seed)} "
        f"--arms {shlex.quote(arms)} --device {shlex.quote(device)}{flags}"
    )


def doctor_line(
    run_dir: str,
    out: str,
    *,
    encoder: str,
    lr: float,
    rows: int,
    epochs: int,
    device: str,
    batch_size: int = 128,
    seed: int = 0,
    repo: str = "/content/nig-timenet-torch",
) -> str:
    """The model doctor on a FINISHED run's best weights (settings must equal the run's)."""
    model = f'{{"encoder": "{encoder}"}}'
    train = f'{{"lr": {lr}, "epochs": {int(epochs)}, "device": "{device}"}}'
    return (
        f". {ENV_SH} && cd {shlex.quote(repo)} && python -u -m nig_timenet_torch.doctor {ARTIFACT_LOCAL} --run-dir "
        f"{shlex.quote(run_dir)} --trained --max-rows {int(rows)} --batch-size {int(batch_size)} "
        f"--seed {int(seed)} --model {shlex.quote(model)} --train {shlex.quote(train)} --out {shlex.quote(out)}"
    )


def runs_line(
    models: str,
    lr: float,
    lr_min: float,
    epochs: int,
    rows: int,
    *,
    device: str = "cuda",
    code: str = "/content/nig_uhub/src",
    script: str = "/content/run_models.py",
) -> str:
    """The single-model runs of the sequential-gradient branch (scripts/run_models.py) with that branch's code."""
    return (
        f". {ENV_SH} && cd /content && PYTHONPATH={shlex.quote(code)}:$PYTHONPATH python -u {shlex.quote(script)} "
        f"{shlex.quote(models)} {lr:g} {lr_min:g} {int(epochs)} {int(rows)} {shlex.quote(device)}"
    )
