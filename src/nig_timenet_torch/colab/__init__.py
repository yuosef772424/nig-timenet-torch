"""Colab runner and notebook support (ROADMAP T3). The notebook ``notebooks/colab_nig_timenet_torch.ipynb`` is a thin
caller of this package; every behaviour has a test in ``tests/colab/``.

    bootstrap  stdlib-only: token, clean clones, --no-deps install (the notebook's first cell downloads this one file)
    device     device report / assertion / smoke computation (rule E1)
    session    notebook section 1: device check and the setup record
    prepare    notebook section 2: TF export -> contract artifact, skipped when already sound for the same source
    job        job.json schema
    contract   run contract between runner and entry (device.json, manifest.json, output validation)
    publish    bundle, Drive copy, push to colab-results, republish (rule E2)
    runner     queue -> run -> results (notebook section 3)
"""
