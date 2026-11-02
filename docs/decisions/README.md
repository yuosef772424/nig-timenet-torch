# Decisions

One file per owner decision that changes a default (rule B1): `D-nnn-short-name.md`, numbered from `D-001`, in the
SAME PR as the change. A new option whose default reproduces the old behaviour needs no decision file, only a line in
`tests/pins/defaults.json` (see below).

`tests/test_decisions.py` compares every default in the package (dataclass field defaults, function and `__init__`
parameter defaults, registry `default_config` values, module constants listed there) with `tests/pins/defaults.json`:

- a default that is **new** is not allowed to stay unpinned: add it to the pin file (its value must reproduce the old
  behaviour, B1);
- a default that **changed or disappeared** fails unless a decision file in this folder lists it on a line of the
  exact form

      - `module.path.name`: `old repr` -> `new repr`

  The check applies the decision files in filename order on top of the pin file, so the old value written in a
  decision must be the value in force before it.

Template:

```markdown
# D-001 — what changed

Date: YYYY-MM-DD
Owner decision: <who / where it was decided>
Why: <the incident or measurement that justifies it>

- `module.path.name`: `old` -> `new`
```
