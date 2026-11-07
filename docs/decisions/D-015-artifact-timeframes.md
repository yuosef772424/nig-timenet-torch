# D-015: the artifact manifest declares its interval and timeframes; a job can refuse another interval (package H5)

Date: 2026-11-07
Owner decision: hardening package H5 (planner addition, data contract). Old artifacts load unchanged.

- `export.base_tf` / `export.timeframes` are optional manifest fields written by `data/artifact_writer.py`. Declared together or
  not at all; when declared, `load_artifact` refuses an artifact whose arrays say otherwise. An artifact without them still
  loads and is flagged (`ArtifactManifest.declares_timeframes` is False; the `data_io.artifact` CLI prints it).
- `LoaderSettings.expect_base_tf` (default None = accept any, behaviour unchanged) and the baseline job key `base_tf` (default
  None) make a job refuse an artifact of another interval at load time (a 1h artifact fed to a 4h job).
- The `RunSettings(data)` line only records that `LoaderSettings`' repr gained the new field (same behaviour).

- `nig_timenet_torch.colab.job.check_entry_config(device)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.artifact.ArtifactManifest(base_tf)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.artifact.ArtifactManifest(timeframes)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.loading.LoaderSettings(expect_base_tf)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.baseline.BaselineConfig(base_tf)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.settings.RunSettings(data)`: `LoaderSettings(directory='.', name='preprocessing_output', format='auto', mmap=True, local_dir='auto', allow_unchecked=False, verify_hashes=False)` -> `LoaderSettings(directory='.', name='preprocessing_output', format='auto', mmap=True, local_dir='auto', allow_unchecked=False, verify_hashes=False, expect_base_tf=None)`
