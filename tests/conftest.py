"""Rule E3 enforcement: fail the run when the fast (not `slow`) tests exceed the time budget in tests/budget.py."""

from tests.budget import violations

_records: dict[str, list] = {}


def pytest_runtest_logreport(report):
    rec = _records.setdefault(report.nodeid, [0.0, False])
    rec[0] += report.duration  # setup + call + teardown
    rec[1] = rec[1] or "slow" in report.keywords


def pytest_sessionfinish(session, exitstatus):
    problems = violations([(nodeid, secs, slow) for nodeid, (secs, slow) in _records.items()])
    session.config._time_budget_problems = problems
    if problems and exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    problems = getattr(terminalreporter.config, "_time_budget_problems", [])
    if problems:
        terminalreporter.section("time budget (rule E3)", red=True)
        for line in problems:
            terminalreporter.line(line, red=True)


from tests.data_io.helpers import art  # noqa: E402,F401  (fixture shared by data_io, jobs, workflow, trainer, notebooks, colab)
