# Colab ops: the steps of every session (2026-11-15)

Code: `src/nig_timenet_torch/colab_ops/` (`python -m nig_timenet_torch.colab_ops --help`). Needs the optional package
`pip install git+https://github.com/yuosef772424/colab-mcp-bridge` and, from the Colab bridge cell, two environment
variables (never written to a file in the repository):

```bash
export CB_URL=https://<tunnel>.trycloudflare.com      # changes every Colab runtime
export COLAB_BRIDGE_TOKEN=<BRIDGE_TOKEN of the cell>
```

| step | command |
|---|---|
| what runtime is this (GPU, RAM, id) | `colab_ops rt` |
| code + cached env + artifact copy (idempotent) | `colab_ops setup <branch> [--torch-key 2.14.1+cu130]` |
| a second branch's code only | `colab_ops setup <branch> --dest /content/nig_uhub --no-artifact --no-install` |
| small-sample encoder x lr arms (+ battery) | `colab_ops arms /content/out --arms gru:3e-4,transformer:1e-4 --rows 20000 --epochs 6` |
| doctor on a finished run's best weights | `colab_ops doctor <run_dir>/ckpt /content/doc --encoder gru --lr 3e-4` |
| run one command and wait (default timeout 120 s) | `colab_ops exec '<command>' [--timeout N]` |
| start a background process, get its pid | `colab_ops start '<command>' <name>` |
| follow a started process | `colab_ops wait <pid> --limit 400` / `colab_ops out <pid>` |
| copy a small file to the runtime | `colab_ops put <local> <remote>` |

(`colab_ops runs` also exists, for single-model runs of the `claude/sequential-gradient-model-uhub56` branch, which is
not on main: it needs `setup` of that branch and is not part of the steps of a session.)

## Queue route (the owner presses Run all; summary of `docs/PLANNER_HANDOFF.md` "How to queue a Colab job")

The alternative to the bridge: a job file `docs/jobs/<job_id>.json` (`repo_ref` there is 40 zeros) is copied to
`job.json` at the root of branch `colab-queue` with `repo_ref` replaced by the 40-hex merge commit to run (`rerun: true`
only to repeat an existing `job_id`) and checked first with `python -m nig_timenet_torch.colab check-job job.json`; the
owner runs `notebooks/colab_nig_timenet_torch.ipynb`; results appear on branch `colab-results` under `runs/<job_id>/`
(`bundle_summary.md` first); a failed publish is retried with `python -m nig_timenet_torch.colab republish <bundle_dir>`.
Smoke job first (rule E5); a code change means a new `run_dir` and `job_id`. Full steps: `docs/PLANNER_HANDOFF.md`.

## The environment cache (nobody installs per conversation)

`setup` restores `/content/drive/MyDrive/crypto/nig_timenet_env/env-<key>.tar` (the shared project folder `crypto`, so every account that shares it uses the same env; before 2026-11-15 it lived in `MyDrive/nig_timenet_env`, the notebook's `ENV_CACHE_DIR`: set it to the new path to share it) (316 MB: numpy, pandas, pyarrow,
pandas-ta-classic, requests and their dependencies, as a venv with `--system-site-packages`) into `/content/nig_venv`
in ~10 s, and writes `/content/nig_env.sh` (`PYTHONPATH` = the venv + the repo's `src`). Every job line starts with
`. /content/nig_env.sh`. Colab's own torch is used and never installed.

- The key is Python + torch + the requirement lines (D-016). A different Colab torch = a new key: the first `setup`
  builds (~1 min) and replaces the old tar. The tar was built with `--torch-key 2.14.1+cu130` (the torch a fresh GPU
  runtime had on 2026-11-14): a fresh runtime with that torch hits the cache; pass the new version in `--torch-key`
  when Colab moves on.
- A runtime change (CPU -> GPU, restart) is a new empty machine: run `setup` again (artifact copy ~40 s).
- Not verified: this branch's tests on Colab's torch 2.14.1 (pyproject pins 2.15; the cached-env route never installs
  torch). The runs of 2026-11-15 used torch 2.15.0+cu132 installed into the system python.

## Starting a job entry by hand (as run 2026-11-16; the queue runner does the same from `colab-queue`)

```bash
# once per runtime: the config of the job file, pointed at the local artifact copy
colab_ops exec 'mkdir -p /content/cfg && cd /content/nig-timenet-torch && python -c "
import json; j = json.load(open(\"docs/jobs/<job>.json\")); c = j[\"config\"]; c[\"artifact\"] = \"/content/art1\"
json.dump(c, open(\"/content/cfg/<job>.json\", \"w\"))"'
colab_ops start '. /content/nig_env.sh && cd /content/nig-timenet-torch && NIG_REPO_DIR=/content/nig-timenet-torch python -u -m nig_timenet_torch.jobs.<entry> --config /content/cfg/<job>.json --out /content/drive/MyDrive/nig_timenet_runs/<job>/out --max-minutes <N>' <job>
colab_ops wait <pid> --limit 400      # or: colab_ops out <pid>
```

`--out` on Drive keeps the reports when the runtime is recycled (`run_dir` holds only the checkpoints). The smoke job
of the entry runs first, on the same commit (rule E5).

## Traps (each cost a session time)

- A recycled runtime is an empty machine with a NEW tunnel URL: everything under `/content` that is not Drive is gone
  (2026-11-16: the look outputs under `/content/look` and the job `--out` folders under `/content/out_*` were lost;
  only the `run_dir` checkpoints on Drive survived). Write `--out` and look outputs to Drive.
- Never move the artifact through the bridge (about 80 kB/s): copy it inside Colab (`setup`, ~40 s); build slices
  (`nig-envs e3p`, E4) on Colab.
- Do not update the checkout a running job imports from: use a second checkout for other work
  (`setup <branch> --dest /content/nig2 --no-artifact --no-install`, then an env file whose `PYTHONPATH` ends in
  `/content/nig2/src`), otherwise the job's manifest records a commit it did not run.
- Several processes share one GPU well while it has room (`nvidia-smi`: a single GRU job used ~15 % of an L4); ten
  processes made every epoch 5 x slower (25 s -> 139 s). Check utilisation after starting each one.
- A tool call that waits longer than 600 s is moved to the background: keep `wait --limit` at 400.
- Colab's torch (2.14.1+cu130) is older than the pyproject pin; the cached env never installs torch.
- The `panel` entry stops at `--max-minutes` and exits 0 with `finished: false`; the `rolling` entry does not read
  the limit. Both resume from `run_dir`: start the SAME command again after a cut.
- A cut job resumes ONLY on the commit it started on: the run identity in its checkpoints holds the code commit
  (`trainer/identity.py:check_resume` raises `ResumeMismatchError` on any other). After pushing new commits, resume
  from a second checkout pinned at the job's commit (`git_sha` in its `out/manifest.json`, or the commit of the
  first run): `git -C /content/nig-timenet-torch worktree add -f /content/nig_old <sha>`, then start the job with
  `NIG_REPO_DIR=/content/nig_old` and `PYTHONPATH` pointing to `/content/nig_old/src` instead of the main checkout's
  `src` (`two_frames_chain.sh` does this for `panel-rolling-ls-001`, 2026-11-18).
