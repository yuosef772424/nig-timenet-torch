# D-005: test-split evaluation in the baseline job

Owner request 2026-11-03: after training, the baseline job evaluates the model (best-epoch weights) on ALL rows of
the test split with `eval/target_report.target_report` and writes `test_report.json`; a short summary is printed in
the Colab output. The test split is never used for training or for choosing the best epoch.

- `nig_timenet_torch.jobs.baseline.evaluate_test(chunk)`: `<new>` -> `20000`
