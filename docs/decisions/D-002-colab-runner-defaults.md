# D-002 — Colab runner and notebook support: new defaults (ROADMAP T3)

Date: 2026-11-03
Owner decision: T3 session prompt (2026-11-03) asks for a Colab notebook with a queue -> run -> results runner, a
`republish` command and a recorded device; the T1 rule applies ("any new default needs a docs/decisions/D-nnn file and a
pin in tests/pins/defaults.json via `python -m tests.defaults_pin --add`").
Why: nothing here changes an existing default: every key below is NEW (the `<new>` format of
docs/decisions/README.md), in the new packages `nig_timenet_torch.colab` and `nig_timenet_torch.jobs`.

What the values mean (so a reader does not have to infer intent):
- `None` for `env`, `secrets`, `cwd`, `python`, `torch_mod`, `argv`, `started`, `record_dir`, `expected`: "use the
  process environment / no masking / the running interpreter / the real `torch` / `sys.argv`"; they exist so the tests
  can inject fakes without a network, a GPU or Colab.
- `echo=print`, `run=subprocess.run`, `urlopen=urllib.request.urlopen`, `version_of`, `device_check`, `accepted`: the same
  injection points, defaulting to the real thing.
- `DEFAULT_WORKDIR` `/content/nig_runner`, `DEFAULT_DRIVE_ROOT` `/content/drive/MyDrive`, `DEFAULT_DRIVE_RUNS`
  `.../nig_timenet_runs`, `DEFAULT_REPO`, `DEFAULT_TOKEN_PATH`: Colab layout; the work dir is deliberately NOT under
  `/tmp` (rule E2). Branch names `colab-queue` / `colab-results` are the TF runner's.
- `grace_seconds=120`, `push_attempts=4`, `attempts=4/3`, `run_git timeout=900`, `echo_lines=400`, `tail_text
  max_bytes=204800`: copied from the TF runner (`tools/colab_runner.py`: grace 120 s, push retried 4 times, git timeout
  900 s, 400 echoed lines, 200 KiB log tail).
- `exact_sha=False`, `force=False`, `adopt_existing=False`: the safe choice (no extra trust, no overwrite).
- `modules=(('pyarrow', 'pyarrow'),)`: the only third-party module the package needs that Colab may lack; torch, numpy and
  pandas are never installed (Colab's own CUDA torch must stay).
- `package_versions(names)`: the packages recorded in `env.json`.

New defaults:

- `nig_timenet_torch.colab.__main__.DEFAULT_REPO`: `<new>` -> `'yuosef772424/nig-timenet-torch'`
- `nig_timenet_torch.colab.__main__.DEFAULT_TOKEN_PATH`: `<new>` -> `'/content/drive/MyDrive/crypto/secrets/github_token.txt'`
- `nig_timenet_torch.colab.__main__.main(argv)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.acquire_token(accepted)`: `<new>` -> `<callable nig_timenet_torch.colab.bootstrap.token_accepted>`
- `nig_timenet_torch.colab.bootstrap.acquire_token(attempts)`: `<new>` -> `3`
- `nig_timenet_torch.colab.bootstrap.clean_clone(env)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.clean_clone(exact_sha)`: `<new>` -> `False`
- `nig_timenet_torch.colab.bootstrap.clean_clone(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.install_package(modules)`: `<new>` -> `(('pyarrow', 'pyarrow'))`
- `nig_timenet_torch.colab.bootstrap.install_package(python)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.install_package(run)`: `<new>` -> `<callable subprocess.run>`
- `nig_timenet_torch.colab.bootstrap.install_package(version_of)`: `<new>` -> `<callable nig_timenet_torch.colab.bootstrap._installed_version>`
- `nig_timenet_torch.colab.bootstrap.run_git(cwd)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.run_git(env)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.run_git(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.bootstrap.run_git(timeout)`: `<new>` -> `900`
- `nig_timenet_torch.colab.bootstrap.setup_clones(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.colab.bootstrap.token_accepted(urlopen)`: `<new>` -> `<callable urllib.request.urlopen>`
- `nig_timenet_torch.colab.contract.entry_args(argv)`: `<new>` -> `None`
- `nig_timenet_torch.colab.contract.write_device_json(torch_mod)`: `<new>` -> `None`
- `nig_timenet_torch.colab.contract.write_manifest(started)`: `<new>` -> `None`
- `nig_timenet_torch.colab.device.assert_device(torch_mod)`: `<new>` -> `None`
- `nig_timenet_torch.colab.device.device_report(expected)`: `<new>` -> `None`
- `nig_timenet_torch.colab.device.device_report(torch_mod)`: `<new>` -> `None`
- `nig_timenet_torch.colab.device.smoke_compute(torch_mod)`: `<new>` -> `None`
- `nig_timenet_torch.colab.prepare.prepare_artifact(adopt_existing)`: `<new>` -> `False`
- `nig_timenet_torch.colab.prepare.prepare_artifact(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.colab.prepare.prepare_artifact(python)`: `<new>` -> `None`
- `nig_timenet_torch.colab.prepare.prepare_artifact(run)`: `<new>` -> `<callable subprocess.run>`
- `nig_timenet_torch.colab.publish.build_bundle(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.clone_branch(env)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.clone_branch(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.fetch_queue(env)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.fetch_queue(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.push_results(attempts)`: `<new>` -> `4`
- `nig_timenet_torch.colab.publish.push_results(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.republish(attempts)`: `<new>` -> `4`
- `nig_timenet_torch.colab.publish.republish(branch)`: `<new>` -> `'colab-results'`
- `nig_timenet_torch.colab.publish.republish(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.colab.publish.republish(force)`: `<new>` -> `False`
- `nig_timenet_torch.colab.publish.results_has_job(env)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.results_has_job(secrets)`: `<new>` -> `None`
- `nig_timenet_torch.colab.publish.tail_text(max_bytes)`: `<new>` -> `204800`
- `nig_timenet_torch.colab.runner.DEFAULT_DRIVE_ROOT`: `<new>` -> `'/content/drive/MyDrive'`
- `nig_timenet_torch.colab.runner.DEFAULT_DRIVE_RUNS`: `<new>` -> `'/content/drive/MyDrive/nig_timenet_runs'`
- `nig_timenet_torch.colab.runner.DEFAULT_WORKDIR`: `<new>` -> `'/content/nig_runner'`
- `nig_timenet_torch.colab.runner.Runner(device_check)`: `<new>` -> `<callable nig_timenet_torch.colab.device.assert_device>`
- `nig_timenet_torch.colab.runner.Runner(drive_root)`: `<new>` -> `'/content/drive/MyDrive'`
- `nig_timenet_torch.colab.runner.Runner(drive_runs)`: `<new>` -> `'/content/drive/MyDrive/nig_timenet_runs'`
- `nig_timenet_torch.colab.runner.Runner(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.colab.runner.Runner(grace_seconds)`: `<new>` -> `120`
- `nig_timenet_torch.colab.runner.Runner(push_attempts)`: `<new>` -> `4`
- `nig_timenet_torch.colab.runner.Runner(python)`: `<new>` -> `None`
- `nig_timenet_torch.colab.runner.Runner(queue_branch)`: `<new>` -> `'colab-queue'`
- `nig_timenet_torch.colab.runner.Runner(results_branch)`: `<new>` -> `'colab-results'`
- `nig_timenet_torch.colab.runner.Runner(runner_sha)`: `<new>` -> `''`
- `nig_timenet_torch.colab.runner.Runner(workdir)`: `<new>` -> `'/content/nig_runner'`
- `nig_timenet_torch.colab.runner.package_versions(names)`: `<new>` -> `('torch', 'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy')`
- `nig_timenet_torch.colab.runner.run_logged(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.colab.runner.run_logged(echo_lines)`: `<new>` -> `400`
- `nig_timenet_torch.colab.session.check_environment(device_check)`: `<new>` -> `<callable nig_timenet_torch.colab.device.assert_device>`
- `nig_timenet_torch.colab.session.check_environment(echo)`: `<new>` -> `<callable builtins.print>`
- `nig_timenet_torch.colab.session.check_environment(record_dir)`: `<new>` -> `None`
- `nig_timenet_torch.jobs.device_smoke.main(argv)`: `<new>` -> `None`
