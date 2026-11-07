# D-014: the defaults pin covers only what the module itself defines (package H5, finding HF-30)

Date: 2026-11-07
Owner decision: hardening package H5 of `docs/hardening/PLAN.md` (owner guidance: rules exist to improve the project; the pin
should flag behaviour defaults, not noise). No value in `src/` changes: `tests/defaults_pin.py` stops collecting two kinds of key.

- Re-exports: a `*DEFAULT*` / `*CONFIG*` constant that is only imported into a module is pinned once, in the module that
  assigns it (`workflow.settings.ANTI_MEMORIZATION_CONFIG` is an import of `model.architectures`; its 13 keys are
  `model.architectures.ANTI_MEMORIZATION_CONFIG.*`, still pinned).
- Injection points: a parameter named in `tests/defaults_pin.py::INJECTION_POINTS` (`echo`, `run`, `env`, `secrets`, `argv`,
  `torch_mod`, `urlopen`, `version_of`, `accepted`, `ask`, `device_check`, `out`, ...) replaces a collaborator in tests; its default
  is the real collaborator or None, not a behaviour setting.
- The pin file keeps the old keys; the lines below record that they are no longer defaults.

Re-exports (13):

- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.class_head_hidden`: `32` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.d_model`: `64` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.dropout`: `0.25` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.encoder`: `'gru'` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.feature_dropout`: `0.1` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.head_hidden`: `64` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.input_clip`: `4.0` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.input_noise_std`: `0.1` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.level_norm`: `'batch'` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.level_passthrough`: `True` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.linear_path_l2`: `0.001` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.num_layers`: `2` -> `<removed>`
- `nig_timenet_torch.workflow.settings.ANTI_MEMORIZATION_CONFIG.stats_mode`: `'symlog'` -> `<removed>`

Injection points (38):

- `nig_timenet_torch.colab.__main__.main(argv)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.acquire_token(accepted)`: `<callable nig_timenet_torch.colab.bootstrap.token_accepted>` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.clean_clone(env)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.clean_clone(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.install_package(run)`: `<callable subprocess.run>` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.install_package(version_of)`: `<callable nig_timenet_torch.colab.bootstrap._installed_version>` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.run_git(env)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.run_git(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.setup_clones(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.colab.bootstrap.token_accepted(urlopen)`: `<callable urllib.request.urlopen>` -> `<removed>`
- `nig_timenet_torch.colab.contract.entry_args(argv)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.contract.write_device_json(torch_mod)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.device.assert_device(torch_mod)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.device.device_report(torch_mod)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.device.smoke_compute(torch_mod)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.prepare.prepare_artifact(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.colab.prepare.prepare_artifact(run)`: `<callable subprocess.run>` -> `<removed>`
- `nig_timenet_torch.colab.publish.build_bundle(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.clone_branch(env)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.clone_branch(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.fetch_queue(env)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.fetch_queue(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.push_results(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.republish(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.colab.publish.results_has_job(env)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.publish.results_has_job(secrets)`: `None` -> `<removed>`
- `nig_timenet_torch.colab.runner.Runner(device_check)`: `<callable nig_timenet_torch.colab.device.assert_device>` -> `<removed>`
- `nig_timenet_torch.colab.runner.Runner(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.colab.runner.run_logged(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.colab.session.check_environment(device_check)`: `<callable nig_timenet_torch.colab.device.assert_device>` -> `<removed>`
- `nig_timenet_torch.colab.session.check_environment(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.data.cli.main(argv)`: `None` -> `<removed>`
- `nig_timenet_torch.eval.battery.show(echo)`: `<callable builtins.print>` -> `<removed>`
- `nig_timenet_torch.jobs.baseline.main(argv)`: `None` -> `<removed>`
- `nig_timenet_torch.jobs.device_smoke.main(argv)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.Run(out)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.prepare(out)`: `None` -> `<removed>`
- `nig_timenet_torch.workflow.run.run_all(out)`: `None` -> `<removed>`
