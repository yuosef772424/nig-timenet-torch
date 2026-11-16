"""Repo tooling (not part of the installed package): the step-closing tool and the register collector.

PURPOSE:  Scripts a session or CI runs from the repo root as ``python -m tools.<name>``: ``close_step`` (end-of-step
  bookkeeping) and ``collect_registers`` (PR-description friction and suggestions into the CSV registers).
TAGS:     tools, close, step, bookkeeping, registers, decisions, status, pins, map
PITFALLS: Standard library plus ``tests.*`` helpers only, so the collector runs in CI without torch; run from the repo
  root (the ``tests`` package is imported from there).
"""
