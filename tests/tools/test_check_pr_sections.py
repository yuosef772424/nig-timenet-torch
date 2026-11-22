"""tools.check_pr_sections: the CI check of a PR description's Friction / Suggestions sections."""

from pathlib import Path

from tools import check_pr_sections as cps

GOOD = """## Behaviour changes (rule B1)
none

## Friction
- none: nothing slowed this PR

## Suggestions
- kind: suggestion
  step: HB8
  where: tools/x.py
  what: a thing
  minutes_lost: 5
  proposed_fix: do it
"""


def test_a_good_description_passes():
    assert cps.problems(GOOD) == []
    assert cps.main({"PR_BODY": GOOD}) == 0


def test_missing_bad_and_empty_sections_fail(capsys):
    assert any("`## Suggestions` section is missing" in p for p in cps.problems("## Friction\n- none: ok fine\n"))
    bad = GOOD.replace("kind: suggestion", "kind: idea")
    assert any("kind must be one of" in p for p in cps.problems(bad))
    assert cps.main({}) == 1 and "empty" in capsys.readouterr().err


def test_the_workflow_passes_the_body_through_an_env_var_only_on_pull_requests():
    text = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    step = text.split("name: PR description has valid Friction and Suggestions sections", 1)[1].split("- name:", 1)[0]
    assert "if: github.event_name == 'pull_request'" in step
    assert "PR_BODY: ${{ github.event.pull_request.body }}" in step
    run = step.split("run:", 1)[1]
    assert "github.event" not in run, "the PR body must never be interpolated into the shell command"
