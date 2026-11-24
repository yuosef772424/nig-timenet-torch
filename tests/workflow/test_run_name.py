"""Owner 2026-11-23: a run name taken by a run with other settings gives the run a new name instead of an error; the
same settings resume under the same name."""

from pathlib import Path

import pytest

from nig_timenet_torch.colab.drive_sync import run_paths
from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.trainer.identity import ResumeMismatchError
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.run_name import ON_TAKEN, candidate_name, claim_run_name
from nig_timenet_torch.workflow.settings import RunSettings
from tests.data_io.helpers import art  # noqa: F401  (fixture)

SMALL = {"d_model": 16, "num_layers": 1, "head_hidden": 16, "class_head_hidden": 8}


def _built(data, paths, **over):
    base = RunSettings(run_dir=paths.run_dir, data=LoaderSettings(name=str(data)))
    s = updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "split.max_rows": 60,
                       "model.overrides": SMALL, **over})
    return steps.build(steps.prepare(s, paths.out_dir))


def test_candidate_names_count_from_the_plain_name():
    assert [candidate_name("a", n) for n in (1, 2, 3)] == ["a", "a-2", "a-3"] and ON_TAKEN[0] == "new_name"


def test_a_taken_name_gets_the_next_free_one_and_the_same_settings_resume(art, tmp_path):  # noqa: F811
    paths = run_paths("nm", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    first = _built(art, paths)
    assert claim_run_name(first, paths).name == "nm"                      # nothing there yet: the plain name
    steps.fit(first)
    notes = []
    same = _built(art, paths)
    assert claim_run_name(same, paths, echo=notes.append).name == "nm" and not notes     # same identity: resume
    other = _built(art, paths, **{"train.seed": 7})
    with pytest.raises(ResumeMismatchError):                              # what happened before this change
        steps.fit(other)
    got = claim_run_name(other, paths, echo=notes.append)
    assert got.name == "nm-2" and got.run_dir == str(Path(tmp_path / "local" / "nm-2" / "train"))
    assert other.settings.run_dir == got.run_dir and other.out == Path(got.out_dir) and Path(got.out_dir).is_dir()
    assert "nm" in notes[0] and "nm-2" in notes[0]
    steps.fit(other)                                                      # trains in the new folder
    third = _built(art, paths, **{"train.seed": 8})
    assert claim_run_name(third, paths, echo=notes.append).name == "nm-3"      # nm and nm-2 are both taken
    again = _built(art, paths, **{"train.seed": 7})
    assert claim_run_name(again, paths, echo=notes.append).name == "nm-2"      # the seed-7 run resumes nm-2


def test_the_drive_copy_counts_as_taken_and_error_mode_keeps_the_old_behaviour(art, tmp_path):  # noqa: F811
    paths = run_paths("dr", str(tmp_path / "drive"), str(tmp_path / "local"), "local")
    drive_side = run_paths("dr", str(tmp_path / "drive"), str(tmp_path / "drive"), "drive")
    steps.fit(_built(art, drive_side))                                    # a run that lives only on Drive
    fresh = _built(art, paths, **{"train.seed": 3})                       # a new VM works locally, other seed
    assert claim_run_name(fresh, paths, echo=lambda *_: None).name == "dr-2"
    again = _built(art, paths, **{"train.seed": 3})
    assert claim_run_name(again, drive_side, on_taken="error") is drive_side
    with pytest.raises(ValueError, match="on_taken"):
        claim_run_name(again, paths, on_taken="overwrite")
