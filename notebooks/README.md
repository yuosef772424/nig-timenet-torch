# Notebooks

[`colab_nig_timenet_torch.ipynb`](colab_nig_timenet_torch.ipynb) is the ONE notebook every Colab run starts from
(ROADMAP T3; owner, 2026-11-03: no cells pasted by hand). It is a thin caller of `src/nig_timenet_torch/colab/`; the
logic and its tests (`tests/colab/`) live there, and `tests/notebooks/test_colab_notebook.py` checks the notebook
statically (no outputs, no `def`, short cells, every call matches a real signature, every name is defined earlier).

Open it with [Open in Colab](https://colab.research.google.com/github/yuosef772424/nig-timenet-torch/blob/main/notebooks/colab_nig_timenet_torch.ipynb),
choose a T4 GPU, then Runtime, then Run all. The first run asks for a GitHub token once (kept on Drive, never printed).

| Section | What it does | Code |
|---|---|---|
| 1. Setup | mounts Drive; clean-clones this repo and the TF repo (`rm -rf` first) and prints both commits; `pip install --no-deps` (Colab's CUDA torch is never replaced; the install is refused if torch's version changes); asserts and records the device (`setup.json`, rule E1) | `colab/bootstrap.py`, `colab/session.py`, `colab/device.py` |
| 2. Data | exports `preprocessing_output_latest.pkl.gz` as a contract artifact with the TF tool and checks it with the loader (`--verify-hashes`); skips the export when a sound artifact from the same source exists | `colab/prepare.py` |
| 3. Runner | queue -> run -> results, then republish if the push failed | `colab/runner.py`, `colab/publish.py`, `colab/contract.py` |

## The runner

1. **Queue.** A job is a `job.json` at the root of branch `colab-queue` (schema and field meanings in
   `src/nig_timenet_torch/colab/job.py`; `python -m nig_timenet_torch.colab check-job job.json` validates one before
   you push it). `device` (`cuda` or `cpu`) is required. `repo_ref` pins the commit the job's code runs at; the runner
   itself comes from the notebook's clone of `TORCH_REF`.
2. **Run.** A finished job id (a `runs/<job_id>/` on `colab-results`) is refused unless the job sets `"rerun": true`. The
   runner asserts the device before starting the entry (a `cuda` job on a CPU VM fails without running), runs
   `python -m nig_timenet_torch.jobs.<entry>` under a wall-clock guard with `TMPDIR` inside the work dir, and validates the
   outputs, including the `device.json` the entry wrote (a `cuda` job whose record says `cpu` fails, rule E1).
3. **Results.** The bundle (`job.json`, `env.json` with device and package versions, `run_log.txt`, `bundle_summary.md`,
   `manifest.json`, `device.json`, the outputs) is written to Drive FIRST (`MyDrive/nig_timenet_runs/<job_id>/`; the work
   dir if Drive is not mounted), then pushed to branch `colab-results` under `runs/<job_id>/`. Outputs above 1 MiB stay on
   Drive. A later session reads `runs/<job_id>/` from GitHub; the owner copies nothing.
4. **Republish (rule E2).** If the push failed, the bundle is already safe on Drive. The notebook's last cell retries it, or in a fresh
   runtime: `python -m nig_timenet_torch.colab republish /content/drive/MyDrive/nig_timenet_runs/<job_id>`. It needs only
   that folder and the token file; nothing under `/tmp`, no earlier clone and no runner object.

Entries are modules of `nig_timenet_torch.jobs`. `device_smoke` proves the whole path on a new VM (queue it first with
`"device": "cuda"` and `"config": {"device": "cuda"}`). Training entries (T5r: `baseline`) use
`colab/identity.py::identity_for_run` for the run identity.

The notebook is committed without outputs. First Colab runs on a T4 (2026-11-03, `docs/STATUS.md` T3 row): setup, export,
device and publish worked; the baseline job has not yet finished on the GPU (`docs/bottlenecks.csv` B-024). The previous demo notebook (parity
table, all model variants, a small training run on the fixture) is in git history at commit `b34579c`.

`baseline` (T5r) trains one seed of the default model; the job files `docs/jobs/baseline-001-s{0,1}.json` carry the
placeholder `repo_ref` of 40 zeros: the planner replaces it with the merge commit at queue time. Running the same job
again with the same `run_dir` and `"rerun": true` resumes from the Drive checkpoint.
