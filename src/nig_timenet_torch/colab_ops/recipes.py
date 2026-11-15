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
    "setup_script",
    "arms_line",
    "doctor_line",
    "runs_line",
]

REPO = "https://github.com/yuosef772424/nig-timenet-torch.git"
TOKEN_FILE = "/content/drive/MyDrive/crypto/secrets/github_token.txt"
ARTIFACT_DRIVE = "/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1"
ARTIFACT_LOCAL = "/content/art1"
_GIT = (
    "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.https://github.com/.extraheader "
    'GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $B"'
)


def setup_script(
    branch: str, dest: str = "/content/nig-timenet-torch", *, artifact: bool = True, install: bool = True
) -> str:
    """Clone (or fast-forward) ``branch`` into ``dest``; install torch 2.15 + the package; copy the artifact to local
    disk (the Drive mmap is slow). Prints the head commit, torch version and the device."""
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
        lines += [
            "python - <<'PY' || pip install -q 'torch>=2.15,<2.16' 2>&1 | tail -1",
            "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.15') else 1)",
            "PY",
            f"pip install -q -e {d}'[data]' 2>&1 | tail -1",
        ]
    if artifact:
        lines += [f"[ -d {ARTIFACT_LOCAL} ] || cp -r {ARTIFACT_DRIVE} {ARTIFACT_LOCAL}"]
    lines += ['python -c "import torch;print(torch.__version__, torch.cuda.is_available())"']
    return "\n".join(lines)


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
        f"cd {shlex.quote(repo)} && python -u docs/audit/2026-11-14-model-first/quick_arms.py {ARTIFACT_LOCAL} "
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
        f"cd {shlex.quote(repo)} && python -u -m nig_timenet_torch.doctor {ARTIFACT_LOCAL} --run-dir "
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
        f"cd /content && PYTHONPATH={shlex.quote(code)} python -u {shlex.quote(script)} "
        f"{shlex.quote(models)} {lr:g} {lr_min:g} {int(epochs)} {int(rows)} {shlex.quote(device)}"
    )
