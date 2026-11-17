"""PURPOSE: The bridges of the backtest: one module per host project, the ONLY code that knows a project's formats
    (model outputs and their times, how its data load); everything else in nig_backtest reads the signal contract.
TAGS: bridge, host project, adapter, model outputs, data loading, portability, signal contract
PITFALLS: a bridge imports its host project lazily, inside its functions (checked by tests/nig_backtest), so
    ``import nig_backtest`` works where the host is not installed; moving the backtest = copy the folder and write a
    bridge for the new project.
"""
