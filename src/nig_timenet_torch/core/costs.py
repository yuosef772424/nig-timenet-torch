"""PURPOSE: The one default trading cost of the project (cost per side of a trade, 0.08 %), read by the trainer's
    side-payoff loss, the run settings, the battery and every report (audit 2026-11b B-34).
TAGS: trading cost, cost per side, round trip, default cost, battery, run settings
PITFALLS: the cost is PER SIDE: a round trip pays twice it (``eval.battery_lib.round_trip``); it lives in ``core`` so
    the trainer and the workflow read it without importing ``eval``; ``tests/eval/test_battery_lib.py`` refuses a
    typed ``0.0008`` anywhere else in the package.
"""

__all__ = ["DEFAULT_COST_PER_SIDE"]

#: the cost per side every default of the battery, the reports, the trainer and the run settings starts from
DEFAULT_COST_PER_SIDE = 0.0008
