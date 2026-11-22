"""Colab runner and notebook support (ROADMAP T3). The notebook ``notebooks/colab_nig_timenet_torch.ipynb`` is a thin
caller of this package; every behaviour has a test in ``tests/colab/``.

    bootstrap  stdlib-only: token, clean clones, code source (GitHub or the Drive copy, updating the copy without losing
               local edits), --no-deps install (the notebook's first cell downloads or reads this one file)
    drive_sync run folders between the VM disk and Drive: where a run works, push, restore (main.ipynb)
    env_cache  Python env on Drive: one tar keyed by Python + torch + requirements (main.ipynb)
    paths      the repository, Drive token file, artifact, work dir and Drive runs root (one place)
    device     device report / assertion / smoke computation (rule E1)
    identity   the one place a run's RunIdentity is built (colab-side home of ``identity_for_run``)
    session    notebook section 1: device check and the setup record
    prepare    notebook section 2: TF export -> contract artifact, skipped when already sound for the same source
    job        job.json schema
    contract   run contract between runner and entry (device.json, manifest.json, output validation)
    publish    bundle, Drive copy, push to colab-results, republish (rule E2)
    runner     queue -> run -> results (notebook section 3)
    __main__   command line: republish a bundle, check a job.json

``colab_ops`` (a sibling package, not part of this one) drives a runtime from outside through the bridge.
"""
