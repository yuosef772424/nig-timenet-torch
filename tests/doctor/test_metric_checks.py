"""doctor/metric_checks.py (card D): the Student-t score, the profit rule and the GOAL row."""

import numpy as np
import torch

from nig_timenet_torch.doctor.metric_checks import metric_checks, profit_rows, student_nll


def test_student_nll_matches_torch_student_t():
    rng = np.random.default_rng(0)
    y, mu = rng.normal(size=50) * 0.03, rng.normal(size=50) * 0.01
    nu, alpha, beta = rng.uniform(0.2, 3, 50), rng.uniform(1.5, 5, 50), rng.uniform(1e-4, 1e-2, 50)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    scale = torch.sqrt(t(beta) * (1 + t(nu)) / (t(nu) * t(alpha)))
    dist = torch.distributions.StudentT(df=2 * t(alpha), loc=t(mu), scale=scale)
    want = -dist.log_prob(t(y)).numpy()
    np.testing.assert_allclose(student_nll(y, mu, nu, alpha, beta), want, rtol=1e-9)


def _stat(avg, ci_lo):
    return {"avg": avg, "ci_lo": ci_lo, "ci_hi": avg + 0.01, "n": 100}


def test_profit_rows_flag_profit_only_with_ci_above_zero_and_beating_the_baseline():
    sections = {
        "S1": {"stats": {"model": _stat(0.02, 0.005), "hold": _stat(0.01, -0.01), "flip": _stat(0.005, -0.01)}},
        "S2": {"stats": {"model": _stat(0.02, -0.001), "hold": _stat(0.001, -0.01)}},      # CI contains 0
        "S3": {"stats": {"model": _stat(0.02, 0.005), "hold": _stat(0.03, 0.01)}},          # a baseline does better
        "S4": {"stats": {"rule - model": _stat(0.02, 0.005), "rule - hold": _stat(0.0, -0.01), "other": {}}},
        "S5": {"stats": {"hold": _stat(0.01, 0.0)}},                                         # no model line: no row
    }
    rows = {r["family"]: r for r in profit_rows(sections)}
    assert set(rows) == {"S1", "S2", "S3", "S4: rule"}
    assert rows["S1"]["profit"] and rows["S1"]["best_baseline"] == "hold"
    assert not rows["S2"]["profit"] and not rows["S3"]["profit"]
    assert rows["S4: rule"]["profit"]


def test_metric_checks_has_a_goal_row(ctx):
    checks = metric_checks(ctx)
    goal = [c for c in checks if c.check.startswith("GOAL")]
    assert len(goal) == 1 and goal[0].card == "D" and goal[0].verdict in ("pass", "fail")
    assert any(c.stage == "metrics/val/close" for c in checks)
