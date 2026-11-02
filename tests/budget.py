"""Rule E3: the fast suite stays fast. Pure budget logic, used by tests/conftest.py and tested in
tests/test_time_budget.py.

A run is a list of ``(nodeid, seconds, is_slow)``. Tests marked ``slow`` are exempt (they run in CI only, see
pyproject.toml); the rest must fit the total budget, and no single fast test may eat most of it.
"""

import os

TOTAL_BUDGET_S = float(os.environ.get("NIG_FAST_SUITE_BUDGET_S", 300))   # rule E3: < 5 minutes on CPU
PER_TEST_BUDGET_S = float(os.environ.get("NIG_FAST_TEST_BUDGET_S", 60))  # one fast test: mark it `slow` beyond this


def violations(records, total_budget: float = TOTAL_BUDGET_S, per_test_budget: float = PER_TEST_BUDGET_S) -> list[str]:
    fast = [(nodeid, secs) for nodeid, secs, is_slow in records if not is_slow]
    out = [f"{nodeid}: {secs:.1f}s > {per_test_budget:.0f}s per fast test; mark it @pytest.mark.slow"
           for nodeid, secs in fast if secs > per_test_budget]
    total = sum(secs for _, secs in fast)
    if total > total_budget:
        slowest = ", ".join(f"{n} {s:.1f}s" for n, s in sorted(fast, key=lambda x: -x[1])[:5])
        out.append(f"fast suite took {total:.1f}s > {total_budget:g}s budget (rule E3). Slowest: {slowest}")
    return out
