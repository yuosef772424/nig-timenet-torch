# Handoff 2026-11-23: old-project reports and charts, all other charts, data pipeline settings, run names

Status: open (plan written 2026-11-23 before the work starts; the owner is asleep, the lead runs it alone)

## What was asked (owner, 2026-11-23, in order)
1. Merge the branches (state found: the session branch of all three repos equals `main` / `master`; the only open PR is
   #79 (DY1, a draft with arms still training, another session's work) and is NOT merged here; every other unmerged
   branch is 160-600 commits behind `main`).
2. Put the reports of the old project (TF, `yuosef772424/crypto-signal-prediction`) into this one: the function
   `run_chicks` (`workflow/run.py` there, which calls `evaluation.full_analysis.run_full_analysis`). Results must
   MATCH the old ones: the figures, the reports, the printed text.
3. Also wire every figure and report that exists here and is not used by any run path (e.g. the top-10 return chart,
   the chart of the return after the market is subtracted / the target), and ADD the charts a reader needs
   (a person understands a picture faster than a table).
4. Update the data pipeline so that EVERY setting of the old data-preparation notebook can be set here, in the
   data notebook; and make this project accept the data the old code prepares.
5. A run name that is already taken by a run with other settings must not raise: the run takes a new name.
6. Merge the work into `main` and update the `main` notebook.

## Work packages (three Sonnet agents in their own worktrees, no push; the lead checks, merges, pushes)
| Pkg | Owner of | Definition of done |
|---|---|---|
| W1 chicks | new `eval/chicks_*.py`, `workflow/chicks_run.py` | `run_chicks(...)` on a finished run writes the same files / figures / printed text as `run_full_analysis`; parity tests against golden outputs of the old code |
| W2 charts | new `eval/charts_*.py`, `workflow/charts_run.py` | every chart that exists but is unused is reachable from one `charts` step; new necessary charts added; an index page of the figures |
| W3 pipeline | `data/` settings + `notebooks/data_pipeline.ipynb`, `data_io/loading.py` | every old `CONFIG` key is a setting here (or listed as intentionally dropped with the reason); old-code data loads and gives the same arrays |
| Lead | run-name rule (W4), `notebooks/main.ipynb`, merge, PR, close_step | see "Order" |

Shared rules for the agents: do not edit `notebooks/main.ipynb`, `docs/STATUS.md`, `docs/MAP.md`, `docs/INDEX.md`;
do not run `close_step`; list any new default (it needs a pin and a decision, rule B1) and any new setting in the
report; every new module gets a card (rule B7); tests of what you touched must pass; no push.

## Order
1. This note + install of the test environment. 2. W1 / W2 / W3 in parallel; the lead does W4 meanwhile.
3. The lead checks each branch against its source, merges them into the session branch, regenerates MAP / INDEX,
   runs the fast suite and ruff. 4. Main notebook: new sections for the chicks report and the charts step, the
   settings listing refreshed. 5. `close_step`, decision records, draft PR on each repo, CI green, merge into `main`
   (the TF project and the bridge only if they changed).

## Left / not done in this session (decided by the lead, owner may reverse)
- PR #79 (DY1) and the old unmerged branches stay as they are (work in progress / stale).
