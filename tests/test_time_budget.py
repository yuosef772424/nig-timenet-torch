"""Rule E3: the time-budget check (tests/budget.py + tests/conftest.py) flags what it should."""

from tests.budget import PER_TEST_BUDGET_S, TOTAL_BUDGET_S, violations


def test_budgets_match_rule_e3():
    assert TOTAL_BUDGET_S <= 300, "E3: the fast suite runs in < 5 minutes"
    assert PER_TEST_BUDGET_S < TOTAL_BUDGET_S


def test_fast_suite_within_budget_passes():
    assert violations([("a", 10.0, False), ("b", 20.0, False)], total_budget=60, per_test_budget=30) == []


def test_one_slow_fast_test_is_flagged():
    out = violations([("a", 31.0, False)], total_budget=600, per_test_budget=30)
    assert len(out) == 1 and "a: 31.0s" in out[0] and "slow" in out[0]


def test_total_over_budget_is_flagged_with_the_slowest_tests():
    out = violations([("a", 25.0, False), ("b", 25.0, False), ("c", 25.0, False)], total_budget=60, per_test_budget=30)
    assert len(out) == 1 and "75.0s > 60s" in out[0] and "a 25.0s" in out[0]


def test_slow_marked_tests_are_exempt():
    assert violations([("slow_one", 900.0, True), ("a", 1.0, False)], total_budget=60, per_test_budget=30) == []
