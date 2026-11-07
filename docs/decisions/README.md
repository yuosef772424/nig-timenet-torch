# Decisions

One file per owner decision that changes a default (rule B1): `D-nnn-short-name.md`, numbered from `D-001`, in the
SAME PR as the change. The session that makes the change writes the file and cites the owner's instruction (the step
prompt or message) as the decision; the owner confirms by merging the PR. Since T1 (owner, 2026-11-02) a NEW default needs one too (it is documented, not judged).

`tests/test_decisions.py` compares every default in the package (dataclass field defaults, function and `__init__`
parameter defaults, registry `default_config` values, module constants listed there) with `tests/pins/defaults.json`:

- a default that is **new** is not allowed to stay unpinned: add it to the pin file (its value must reproduce the old
  behaviour, B1);
- a default that **changed or disappeared** fails unless a decision file in this folder lists it on a line of the
  exact form

      - `module.path.name`: `old repr` -> `new repr`

  The check applies the decision files in filename order on top of the pin file, so the old value written in a
  decision must be the value in force before it;
- a **new** default is documented with `<new>` as the old value, `- `key`: `<new>` -> `value``; the value must agree
  with the pin file (`python -m tests.defaults_pin --add` writes additions only and never touches an existing key).
  Rule from the owner (T1 session, 2026-11-02): a new or changed default needs a decision file AND a pin.

What is pinned: constants assigned in the module itself (a re-export is pinned where it is defined) and parameter defaults
except the injection points listed in `tests/defaults_pin.py::INJECTION_POINTS` (`run`, `env`, `echo`, ...). CI runs
`python -m tests.pin_diff origin/<base>` on every PR: each key added, changed or removed in `tests/pins/defaults.json`
versus the base must appear on a line of a decision file ADDED by the PR.

Template:

```markdown
# D-001 — what changed

Date: YYYY-MM-DD
Owner decision: <who / where it was decided>
Why: <the incident or measurement that justifies it>

- `module.path.name`: `old` -> `new`
```
