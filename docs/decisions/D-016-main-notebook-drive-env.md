# D-016: main notebook, Drive sync and the cached environment (ROADMAP T3b)

Date: 2026-11-08
Owner decision: step T3b and the owner's message of 2026-11-08 (translated): the torch `notebooks/main.ipynb` must be as detailed as the first
project's main notebook, in Arabic, and must load the project from and push the project to Google Drive. Nothing that already ran changes.

New defaults (none changes an existing behaviour; each is a new option whose default reproduces what the notebook did before):

- `colab.bootstrap.load_code`: `update_drive=False` and `discard_local=False` (the Drive copy is only touched when asked; local edits are never discarded silently).
  `acquire_token(preset="", save=True)` (a token typed by the owner is kept on Drive as before; a Colab secret passes `save=False`).
- `colab.drive_sync`: `run_paths(work_on="drive")` = the run writes on Drive exactly as the notebook did (`DRIVE_RUNS/notebook-main/{train,out}`);
  `push_run(overwrite=True)` replaces older files of the same run by newer ones; `restore_run(overwrite=False)` refuses to replace a differing file.
- `colab.env_cache`: the cached env keeps the `data` extra of `pyproject.toml`, the key is Python + torch + requirement lines; torch is never installed.
- Notebook settings cell (not pinned code, listed here so a changed value is a visible decision): `REPO_SOURCE="auto"`, `REPO_DIR=/content/drive/MyDrive/nig_timenet_torch`,
  `UPDATE_DRIVE_COPY=False`, `USE_ENV_CACHE=True`, `ENV_CACHE_DIR=/content/drive/MyDrive/nig_timenet_env`, `VENV_DIR=/content/nig_venv`, `WORK_ON="drive"`,
  `LOCAL_RUNS=/content/nig_runs`, `RESTORE_FROM=""`, `PUSH_TO_DRIVE=True`. `CHANGES` holds every `RunSettings` field at its current default
  (a test compares it with the code), so a run with no edit equals `jobs/baseline.py` on the same settings.
- Behaviour that does change for an owner who opens the notebook: it now builds a venv on Drive on the first run (a few minutes, then seconds);
  `USE_ENV_CACHE=False` restores the previous install path.

- `nig_timenet_torch.colab.bootstrap.acquire_token(preset)`: `<new>` -> `''`
- `nig_timenet_torch.colab.bootstrap.acquire_token(save)`: `<new>` -> `True`
- `nig_timenet_torch.colab.bootstrap.find_token(secret_getter)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.load_code(discard_local)`: `<new>` -> `False`
- `nig_timenet_torch.colab.bootstrap.load_code(remote_url)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.load_code(token)`: `<new>` -> `''`
- `nig_timenet_torch.colab.bootstrap.load_code(update_drive)`: `<new>` -> `False`
- `nig_timenet_torch.colab.bootstrap.update_drive_copy(discard_local)`: `<new>` -> `False`
- `nig_timenet_torch.colab.drive_sync.copy_tree(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.colab.drive_sync.push_run(overwrite)`: `<new>` -> `True`
- `nig_timenet_torch.colab.drive_sync.restore_run(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.colab.drive_sync.run_paths(work_on)`: `<new>` -> `'drive'`
- `nig_timenet_torch.colab.env_cache.activate_env(names)`: `<new>` -> `('numpy', 'pandas', 'pyarrow')`
- `nig_timenet_torch.colab.env_cache.ensure_env(extras)`: `<new>` -> `('data')`
- `nig_timenet_torch.colab.env_cache.ensure_env(python)`: `<new>` -> `None`
- `nig_timenet_torch.colab.env_cache.ensure_env(python_version)`: `<new>` -> `None`
- `nig_timenet_torch.colab.env_cache.ensure_env(torch_version)`: `<new>` -> `None`
- `nig_timenet_torch.colab.env_cache.environment_key(extras)`: `<new>` -> `('data')`
- `nig_timenet_torch.colab.env_cache.environment_key(python_version)`: `<new>` -> `None`
- `nig_timenet_torch.colab.env_cache.environment_key(torch_version)`: `<new>` -> `None`
- `nig_timenet_torch.colab.env_cache.requirements_of(extras)`: `<new>` -> `('data')`
