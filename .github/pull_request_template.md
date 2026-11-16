## What was built
<!-- One short paragraph: the step id, what changed, which modules. -->

## Behaviour changes (rule B1)
<!-- Paste the section that `python -m tools.close_step --step <id>` printed (also in .close_step/behaviour_changes.md):
     the generated table of changed or new defaults with their decision ids, plus ONE free-text line for any other
     behaviour change (write `none` if there is none). A PR that changes no behaviour says so here. -->

## Process change for sessions
<!-- Only when the PR changes how sessions or CI work; otherwise write `none`. -->

## Checks
<!-- Commands run and results (test counts), and what was NOT verified. -->

## Friction
<!-- Machine-read by `python -m tools.collect_registers` after the PR is merged (rule F1). Do NOT edit
     docs/friction_log.csv in the PR. One bullet per obstacle that cost real time, written as `field: value` lines,
     field names = the CSV columns (one line per field, no line breaks inside a value):

- step: TL1
  what_slowed_or_broke: what slowed you down or broke
  where: path/to/file.py:123
  minutes_lost: 10
  root_cause: why it happened
  fix_idea: what would prevent it

     `date` is optional (default: the merge date). `minutes_lost` is a whole number. If there was no obstacle, write
     exactly one bullet: `- none: <why>`. Replace this comment's example with your rows (or delete the comment). -->
- TODO: replace this line with your rows (the collector rejects a body that still has it)

## Suggestions
<!-- Machine-read like Friction; do NOT edit docs/suggestions.csv in the PR (rule F5). One bullet per row. Required:
     (a) at least one concrete improvement suggestion (kind: suggestion) learned during the step, and
     (b) every UNCLEAR or STALE place you met (kind: unclear / stale). Columns, as `field: value` lines:

- step: TL1
  kind: suggestion
  where: path/to/file.py:123
  what: what you suggest, or what was unclear or stale
  minutes_lost: 0
  proposed_fix: the concrete fix
  status: open

     `kind` is one of suggestion / unclear / stale. `status` is optional (default `open`; a stale one-line fix made in
     this PR: `fixed`). `date` is optional. Say `- none: <why>` explicitly if you really have no suggestion. -->
- TODO: replace this line with your rows (the collector rejects a body that still has it)
