"""PURPOSE: The fixed places of the Colab workflow in one module: the repository, the Drive token file, the data
    artifact, the runner work dir and the Drive runs root, read by colab, colab_ops and the job entries.
TAGS: colab paths, repository name, drive token path, artifact path, work dir, drive runs root, constants
PITFALLS: stdlib-only constants and nothing else; ``colab/bootstrap.py`` stays standalone (it is pasted into a fresh
    runtime) and does not import this module; a changed value here changes every command that reads it.
"""

REPO_SLUG = "yuosef772424/nig-timenet-torch"
REPO_URL = f"https://github.com/{REPO_SLUG}.git"
TOKEN_FILE = "/content/drive/MyDrive/crypto/secrets/github_token.txt"
ARTIFACT_DRIVE = "/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1"
WORKDIR = "/content/nig_runner"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_RUNS = DRIVE_ROOT + "/nig_timenet_runs"
