"""PURPOSE: The contract-artifact layer below data: the npy_dir format (reader and writer), the artifact loader and
    contract writer, the holdout lock, split plan, target codecs and the synthetic test environments E1 .. E4.
TAGS: data io package, dataset readers, artifact writer, artifact contract, npy dir, synthetic environments
PITFALLS: torch-free (core / stats only; environments.main lazily reaches workflow for the nig-envs script); never
    imports ``data`` (the build pipeline, fetch and the derived-artifact tools sit ABOVE it, G7); nothing is
    re-exported here, import e.g. nig_timenet_torch.data_io.loading.
"""
