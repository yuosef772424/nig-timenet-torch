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
| single-model runs of `claude/sequential-gradient-model-uhub56` | `colab_ops runs itransformer,conv1d_net --rows 320000` |
| follow a started process | `colab_ops wait <pid> --limit 400` / `colab_ops out <pid>` |

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
