"""Rule B1 (docs/RULES.md): a default never changes silently.

``tests/pins/defaults.json`` pins every default of the package (see tests/defaults_pin.py for what counts). A default
that changed or disappeared fails here unless a decision file ``docs/decisions/D-nnn-*.md`` lists it as
``- `key`: `old` -> `new```; a new default must be pinned (its value reproduces the old behaviour). The tests below
the first two prove the guard itself works, on throw-away decision folders.
"""

import json

import pytest

from tests import defaults_pin as dp


def _write(folder, name, *lines):
    (folder / name).write_text("# decision\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def test_pin_file_is_consistent():
    data = json.loads(dp.PIN_FILE.read_text(encoding="utf-8"))
    assert data["fingerprint"] == dp.fingerprint(data["defaults"]), (
        "tests/pins/defaults.json: `fingerprint` does not match `defaults`; regenerate with "
        "`python -m tests.defaults_pin --add` (additions only)")


def test_defaults_match_pin_and_decisions():
    current = dp.collect_defaults()
    effective = dp.apply_decisions(dp.load_pin())
    changed, added = dp.compare(current, effective)
    msg = []
    if changed:
        msg.append("DEFAULTS CHANGED without a decision (rule B1). Write docs/decisions/D-nnn-<name>.md in this PR "
                   "with one line per key:\n" + "\n".join(
                       f"- `{k}`: `{old}` -> `{'<removed>' if new is None else new}`" for k, (old, new) in
                       sorted(changed.items())))
    if added:
        msg.append("NEW defaults are not pinned (B1: a new option's default must reproduce the old behaviour). Run "
                   "`python -m tests.defaults_pin --add` and commit tests/pins/defaults.json:\n" + "\n".join(added))
    assert not msg, "\n\n".join(msg)


def test_decision_files_are_well_formed():
    files = dp.decision_files()
    numbers = []
    for path in files:
        m = dp.DECISION_NAME.match(path.name)
        assert m, f"{path.name}: decision files are named D-nnn-short-name.md"
        numbers.append(int(m.group(1)))
        assert dp.parse_decision(path), f"{path.name}: no line of the form - `key`: `old` -> `new`"
    assert numbers == list(range(1, len(numbers) + 1)), f"decision numbers must be 001, 002, ...: got {numbers}"
    dp.apply_decisions(dp.load_pin())  # raises if a decision's old value is not the value in force


def test_guard_blocks_a_changed_default(tmp_path):
    pinned = {"m.f(x)": "1", "m.f(y)": "'a'"}
    current = {"m.f(x)": "2", "m.f(y)": "'a'"}
    changed, added = dp.compare(current, dp.apply_decisions(pinned, tmp_path))
    assert changed == {"m.f(x)": ("1", "2")} and not added


def test_guard_accepts_a_decided_change(tmp_path):
    _write(tmp_path, "D-001-x.md", "- `m.f(x)`: `1` -> `2`")
    changed, added = dp.compare({"m.f(x)": "2"}, dp.apply_decisions({"m.f(x)": "1"}, tmp_path))
    assert not changed and not added


def test_guard_chains_decisions_and_rejects_a_wrong_old_value(tmp_path):
    _write(tmp_path, "D-001-x.md", "- `m.f(x)`: `1` -> `2`")
    _write(tmp_path, "D-002-y.md", "- `m.f(x)`: `2` -> `3`")
    assert dp.apply_decisions({"m.f(x)": "1"}, tmp_path) == {"m.f(x)": "3"}
    _write(tmp_path, "D-003-z.md", "- `m.f(x)`: `1` -> `9`")  # 1 is no longer in force
    with pytest.raises(ValueError, match="old value"):
        dp.apply_decisions({"m.f(x)": "1"}, tmp_path)


def test_guard_blocks_a_removed_default_and_flags_unpinned_ones(tmp_path):
    changed, added = dp.compare({"m.g(z)": "0"}, dp.apply_decisions({"m.f(x)": "1"}, tmp_path))
    assert changed == {"m.f(x)": ("1", None)} and added == ["m.g(z)"]
    _write(tmp_path, "D-001-x.md", "- `m.f(x)`: `1` -> `<removed>`")
    assert dp.apply_decisions({"m.f(x)": "1"}, tmp_path) == {}


def test_a_real_default_change_is_detected(monkeypatch):
    """End to end on the real package: pretend one pinned default had another value."""
    pinned = dp.load_pin()
    key = "nig_timenet_torch.core.seeds.set_determinism(seed)"
    assert key in pinned
    tampered = {**pinned, key: "12345"}
    changed, _ = dp.compare(dp.collect_defaults(), dp.apply_decisions(tampered))
    assert key in changed


def test_guard_accepts_a_documented_new_default_and_checks_it_against_the_pin(tmp_path):
    _write(tmp_path, "D-001-x.md", "- `m.f(x)`: `<new>` -> `7`")
    assert dp.apply_decisions({}, tmp_path) == {"m.f(x)": "7"}            # pin file not yet updated: fine
    assert dp.apply_decisions({"m.f(x)": "7"}, tmp_path) == {"m.f(x)": "7"}  # pinned with the same value: fine
    with pytest.raises(ValueError, match="pin file says"):
        dp.apply_decisions({"m.f(x)": "8"}, tmp_path)

