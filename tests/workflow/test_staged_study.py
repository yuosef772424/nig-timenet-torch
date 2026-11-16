"""ST1: the staged study on the fixture artifact (CPU): the staged arm and B1 / B2 / B3 per seed at equal steps,
the tables, the accept rule."""

import json

import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.trainer.staged import StageSettings
from nig_timenet_torch.workflow.settings import RunSettings
from nig_timenet_torch.workflow.staged_report import comparison, stage_table
from nig_timenet_torch.workflow.staged_study import run_staged_study, verdict


def settings(art, tmp_path):  # noqa: F811
    st = StageSettings(start_share=0.5, eval_every=2, step_cap_epochs=2, warmup_steps=2, bn_refresh_steps=2, k=2,
                       probe_rows=16)
    base = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art)))
    return updated(base, {"train.batch_size": 64, "train.device": "cpu", "train.staged": st,
                          "evaluation.battery.enabled": False,
                          "split.max_rows": 96})


def test_study_runs_every_arm_at_the_staged_runs_number_of_steps(art, tmp_path):  # noqa: F811
    out = tmp_path / "study"
    study = run_staged_study(settings(art, tmp_path), (0, 1), out)
    assert set(study["arms"]) == {"staged", "b1", "b2", "b3"}
    for i in range(2):
        steps = {a: study["arms"][a][i]["steps"] for a in study["arms"]}
        assert len(set(steps.values())) == 1, steps
        assert study["arms"]["b1"][i]["stages"][0]["rows"] == study["arms"]["staged"][i]["stages"][-1]["rows"]
        assert study["arms"]["b3"][i]["stages"][0]["rows"] == study["arms"]["staged"][i]["stages"][0]["rows"]
    assert "test_rank_ic" in study["arms"]["staged"][0]
    text = (out / "study.md").read_text()
    assert "Verdict:" in text and "| stage |" in text and "WARNING" not in text
    assert json.loads((out / "study.json").read_text())["verdict"] == study["verdict"]
    assert study["verdict"].startswith(("no gain", "staged training helps"))


def test_verdict_follows_the_accept_rule():
    def row(seed, v, ic):
        return {"seed": seed, "steps": 10, "best_value": v, "test_rank_ic": {"high.rank_ic_by_timestamp": ic}}

    arms = {"staged": [row(0, -6.9, 0.3), row(1, -6.9, 0.3)], "b1": [row(0, -6.8, 0.3), row(1, -6.81, 0.29)],
            "b2": [row(0, -6.8, 0.3), row(1, -6.8, 0.3)]}
    study = {"seeds": [0, 1], "arms": arms, "selection_metric": "val_raw_loss"}
    assert verdict(study).startswith("staged training helps")
    arms["staged"][1] = row(1, -6.805, 0.3)                       # inside the B1 seed spread
    assert verdict(study).startswith("no gain")
    arms["staged"][1] = row(1, -6.9, 0.2)                         # better loss, lower test rank IC
    assert verdict(study).startswith("no gain")
    assert verdict({**study, "seeds": [0]}).startswith("cannot judge")
    assert verdict({**study, "selection_metric": "val_rank_ic_reg"}).startswith("cannot judge")


def test_tables_warn_when_the_steps_differ():
    arms = {"staged": [{"seed": 0, "steps": 10, "best_value": -1.0}],
            "b1": [{"seed": 0, "steps": 12, "best_value": -1.0}]}
    assert "WARNING" in comparison(arms)
    row = {"stage": 0, "rows": 5, "lr": 1e-3, "steps": 4, "steps_to_sign": None, "ended_by": "cap", "best_value": -1.0,
           "gap_at_end": 0.1, "rank_ic": {"high": 0.1}}
    assert "| 0 | 5 | - |" in stage_table([row])


def test_a_study_needs_the_staged_section(art, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="staged"):
        run_staged_study(updated(settings(art, tmp_path), {"train.staged": None}), (0,), tmp_path / "o")
