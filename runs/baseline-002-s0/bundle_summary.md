# baseline-002-s0: OK

- repo_ref: `8451a62938697a5eab3d0999c2e466422072f601`
- device: cuda (used cuda, Tesla T4)
- note: T5r baseline-002 seed 0: config of baseline-001-s0 with 5 epochs (as baseline-001-s0-r2, owner), at current main, first run with the evaluation battery (first profit numbers). New run_dir (resume across a code change is refused). r2 took 972 s on a T4; 60 minutes is the cut. Queue only after baseline-smoke-002 passed on the same commit (rule E5). repo_ref is a placeholder: the planner fills the merge sha at queue time.

Kept on Drive only (above the git size limit): battery_table_test.parquet, battery_table_val.parquet
