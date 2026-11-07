# D-007: dataset loading defaults (package I1)

Date: 2026-11-07
Owner decision: package I1 of `docs/hardening/PLAN.md` ("Final campaign order"); the loader is ported from the owner's
`feat/ar-stream` (`data_io/loading.py`, source commit 2de2846), with one change required by H0b
(`docs/hardening/BRANCH_AR_STREAM.md`, C4): rule D1 is mandatory by default.
Why: the branch applied the D1 checks only when a folder manifest had an `export` block, so a `.pkl.gz` or an old folder
loaded unchecked (TF incident behind D1: 718 coins in the dataset vs 677 allowed). Now an unchecked source is refused
unless the caller says `allow_unchecked=True`. All other values are the branch's, new here (nothing existed before).
(D-006 is taken on the unmerged branch of PR #25, hence D-007.)

- `nig_timenet_torch.data_io.loading.LoaderSettings(allow_unchecked)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.loading.LoaderSettings(directory)`: `<new>` -> `'.'`
- `nig_timenet_torch.data_io.loading.LoaderSettings(format)`: `<new>` -> `'auto'`
- `nig_timenet_torch.data_io.loading.LoaderSettings(local_dir)`: `<new>` -> `'auto'`
- `nig_timenet_torch.data_io.loading.LoaderSettings(mmap)`: `<new>` -> `True`
- `nig_timenet_torch.data_io.loading.LoaderSettings(name)`: `<new>` -> `'preprocessing_output'`
- `nig_timenet_torch.data_io.loading.LoaderSettings(verify_hashes)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.loading.LoadedData(manifest)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.loading.load_dataset(settings)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.loading.resolve_dataset_path(fmt)`: `<new>` -> `'auto'`
- `nig_timenet_torch.data_io.npy_dir.load(mmap)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.npy_dir.load(small_bytes)`: `<new>` -> `67108864`
- `nig_timenet_torch.data_io.artifact.load_artifact(mmap)`: `<new>` -> `False`
- `nig_timenet_torch.core.config.updated(patch)`: `<new>` -> `None`

Notes: `mmap=False` on `npy_dir.load` / `load_artifact` keeps the old behaviour byte for byte; `LoaderSettings.mmap=True` is
the loader's own default (big arrays stay on disk), a different object from the two functions. `small_bytes` (64 MiB) is the
TF project's threshold. `local_dir='auto'` copies only when the path is under `/content/drive`.
