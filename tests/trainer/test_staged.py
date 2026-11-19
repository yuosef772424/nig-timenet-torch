"""ROADMAP ST1 / card F section 6: staged training on the fixture model, CPU, a few seconds.

One stage over all rows equals ``Trainer.fit``; row sets are nested and end at the full split; the sign fires on a
planted memorisation case and not when the train probe is flat; resume in the middle of a stage equals an
uninterrupted run; the validation rows are the same in every stage; the baselines get the staged run's step count."""

import dataclasses

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.trainer.staged import (
    GROWTH,
    GrowthContext,
    SignDetector,
    StageSettings,
    baseline_stages,
    default_eval_every,
    plan_stages,
    staged_fit,
)
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings
from tests.trainer.helpers import data, identity_for, make_trainer, state_equal  # noqa: F401  (fixture)

BS = 32
RAW = EarlyStopSettings(patience=1000, smoothing="none")      # raw selection, no early stop (N-13)


def ctx_of(n, coins=8, seed=0):
    return GrowthContext(coin=np.arange(n) // (n // coins), time=np.tile(np.arange(n // coins), coins), seed=seed)


def run_staged(trainer, d, stages, st, fit, **kw):
    return staged_fit(trainer, d[0], d[1], d[2], d[3], stages, st, fit, **kw)


def test_one_stage_over_all_rows_equals_plain_fit(data):  # noqa: F811
    fit = FitSettings(epochs=4, batch_size=BS, early_stop=RAW)
    plain = make_trainer()
    res = plain.fit(*data, fit)
    steps = -(-len(data[0]) // BS)
    st = StageSettings(start_share=1.0, gamma=1.0, eval_every=steps, step_cap_epochs=4)
    staged = make_trainer()
    out = run_staged(staged, data, plan_stages(st, ctx_of(len(data[0])), BS, 1e-3), st, fit)
    assert len(out.stages) == 1 and out.total_steps == 4 * steps and out.finished
    assert state_equal(plain.model, staged.model)
    assert [h["val_raw_loss"] for h in res.history] == [h["val_raw_loss"] for h in out.history]
    assert out.best_eval == res.best_epoch and out.best_value == res.best_value


@pytest.mark.parametrize("arm", ["stride", "coins", "time"])
def test_row_sets_are_nested_and_end_at_the_full_split(arm):
    st = StageSettings(growth=arm, start_share=1 / 8)
    stages = plan_stages(st, ctx_of(128), BS, 3e-4)
    assert [len(s.rows) for s in stages][-1] == 128 and len(stages) == 4
    for a, b in zip(stages, stages[1:], strict=False):
        assert np.isin(a.rows, b.rows).all() and len(b.rows) > len(a.rows)
        assert np.array_equal(np.sort(np.concatenate([a.rows, b.new_rows])), b.rows)
    assert np.array_equal(stages[-1].rows, np.arange(128)) and len(stages[0].new_rows) == 0
    lrs = [s.lr for s in stages]
    assert lrs == sorted(lrs, reverse=True) and lrs[1] == pytest.approx(1.5e-4) and all(s.restart for s in stages[1:])


def test_stride_halves_the_stride_and_the_lr_never_leaves_its_floor():
    st = StageSettings(start_share=1 / 32, lr_floor=1e-5)
    stages = plan_stages(st, ctx_of(256), BS, 3e-4)
    assert [len(s.rows) for s in stages] == [8, 16, 32, 64, 128, 256] and len(set(map(len, GROWTH.names()))) > 0
    assert stages[-1].lr == pytest.approx(1e-5) and min(s.lr for s in stages) >= 1e-5
    with pytest.raises(ValueError, match="1 / 2\\^j"):
        plan_stages(StageSettings(start_share=0.3), ctx_of(256), BS, 3e-4)


def test_time_arm_starts_with_the_oldest_rows():
    ctx = ctx_of(64)
    first = plan_stages(StageSettings(growth="time", start_share=1 / 4), ctx, BS, 1e-3)[0].rows
    assert ctx.time[first].max() <= np.sort(ctx.time)[len(first) - 1]


def test_settings_are_strict():
    with pytest.raises(ValueError, match="unknown key"):
        from_dict(StageSettings, {"gama": 0.5})
    for bad in ({"growth": "random"}, {"gamma": 0.0}, {"grow": 1.0}, {"k": 0}, {"new_weight": 0}):
        with pytest.raises(ValueError):
            StageSettings(**bad)


def test_sign_needs_a_val_plateau_and_a_falling_probe():
    d = SignDetector(k=3, smooth=3, min_delta=1e-3)
    seq = [(1.0, 1.0), (0.9, 0.9), (0.8, 0.8), (0.8, 0.7), (0.82, 0.6), (0.85, 0.5), (0.9, 0.4), (0.95, 0.3)]
    fired = [d.update(v, p) for v, p in seq]
    assert fired[-1] and not any(fired[:5])
    flat = SignDetector(k=3, smooth=3, min_delta=1e-3)    # val plateau, probe flat: finished learning, no sign
    assert not any(flat.update(0.8, 0.7) for _ in range(20))
    rising = SignDetector(k=3, smooth=3, min_delta=1e-3)  # both still improving: no sign
    assert not any(rising.update(1.0 - 0.01 * i, 1.0 - 0.02 * i) for i in range(20))


def memorise_case(data):  # noqa: F811
    """64 train rows with SHUFFLED labels (only memorisation lowers their loss) against the true-label val rows."""
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(64, generator=g)
    x, y = data[0][:64], {k: v[:64][perm] for k, v in data[1].items()}
    return x, y, data[2], data[3]


def test_the_sign_fires_when_a_model_memorises_shuffled_labels(data):  # noqa: F811
    d = memorise_case(data)
    st = StageSettings(start_share=1.0, gamma=1.0, eval_every=4, k=3, min_delta=1e-3, step_cap_epochs=200)
    fit = FitSettings(epochs=1, batch_size=16, early_stop=RAW)
    t = make_trainer(lr=1e-2)
    out = run_staged(t, d, plan_stages(st, ctx_of(64), 16, 1e-2), st, fit)
    rec = out.stages[0]
    assert rec["ended_by"] == "sign" and rec["steps_to_sign"] == rec["steps"] < 200 * 4
    assert rec["gap_at_end"] > 0 and out.history[-1]["probe_raw_loss"] < out.history[0]["probe_raw_loss"]


def test_the_sign_does_not_fire_when_the_train_probe_is_flat(data):  # noqa: F811
    d = memorise_case(data)
    st = StageSettings(start_share=1.0, gamma=1.0, eval_every=4, k=3, min_delta=1e-1, step_cap_epochs=12)
    fit = FitSettings(epochs=1, batch_size=16, early_stop=RAW)
    out = run_staged(make_trainer(lr=1e-12), d, plan_stages(st, ctx_of(64), 16, 1e-12), st, fit)
    assert out.stages[0]["ended_by"] == "cap" and out.stages[0]["steps_to_sign"] is None and out.finished


def two_stage(rows=96):
    st = StageSettings(growth="stride", start_share=0.5, gamma=0.5, eval_every=2, step_cap_epochs=2, k=2,
                       warmup_steps=2, bn_refresh_steps=2, probe_rows=16)
    return st, plan_stages(st, ctx_of(rows), BS, 1e-3)


def test_resume_in_the_middle_of_a_stage_equals_an_uninterrupted_run(data, tmp_path):  # noqa: F811
    st, stages = two_stage()
    fit = FitSettings(epochs=1, batch_size=BS, early_stop=RAW)
    ident = identity_for(fit)
    full = make_trainer()
    full_res = run_staged(full, data, stages, st, fit, run_dir=tmp_path / "full", identity=ident)
    assert len(full_res.stages) == 2 and full_res.stages[0]["rows"] == 48 and full_res.stages[1]["rows"] == 96

    class Cut(Exception):
        pass

    def stop_after_three(row):
        if row["eval"] == 3:    # stage 0 has 3 steps per pass and an evaluation every 2: this is mid-pass
            raise Cut

    first = make_trainer()
    with pytest.raises(Cut):
        run_staged(first, data, stages, st, fit, run_dir=tmp_path / "cut", identity=ident, on_eval=stop_after_three)
    second = make_trainer(init_seed=123)             # a new process: everything comes from the checkpoint
    res = run_staged(second, data, stages, st, fit, run_dir=tmp_path / "cut", identity=ident)
    assert state_equal(second.model, full.model) and res.history == full_res.history
    assert res.stages == full_res.stages and res.total_steps == full_res.total_steps
    with pytest.raises(FileExistsError):
        run_staged(make_trainer(), data, stages, st, fit, run_dir=tmp_path / "cut", identity=ident, resume=False)
    with pytest.raises(ValueError, match="changed"):
        run_staged(make_trainer(), data, stages, dataclasses.replace(st, k=3), fit, run_dir=tmp_path / "cut",
                   identity=ident)


def test_the_validation_rows_are_the_same_in_every_stage(data):  # noqa: F811
    st, stages = two_stage()
    st = dataclasses.replace(st, val_max_rows=20)
    seen = []
    t = make_trainer()
    original = t.evaluate

    def spy(x, y, batch_size):
        if len(x) == 20:
            seen.append((x.clone(), {k: v.clone() for k, v in y.items()}))
        return original(x, y, batch_size)

    t.evaluate = spy
    res = run_staged(t, data, stages, st, FitSettings(epochs=1, batch_size=BS, early_stop=RAW))
    pick = np.unique(np.linspace(0, len(data[2]) - 1, 20).astype(np.int64))
    assert res.val_rows == 20 and len(seen) >= 2 * len(res.stages) and {r["stage"] for r in res.history} == {0, 1}
    for x, y in seen:
        assert torch.equal(x, data[2][torch.as_tensor(pick)])
        assert all(torch.equal(y[k], data[3][k][torch.as_tensor(pick)]) for k in y)


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_baselines_get_the_staged_runs_number_of_steps(data):  # noqa: F811
    st, stages = two_stage()
    fit = FitSettings(epochs=1, batch_size=BS, early_stop=RAW)
    res = run_staged(make_trainer(), data, stages, st, fit)
    for kind in ("b1", "b2", "b3"):
        plan = baseline_stages(kind, stages, res, st, BS)
        out = run_staged(make_trainer(), data, plan, st, fit)
        assert out.total_steps == res.total_steps and len(out.stages) == 1 and out.stages[0]["ended_by"] == "cap"
        assert plan[0].stop_at_sign is False and plan[0].restart is False
    b2 = baseline_stages("b2", stages, res, st, BS)[0]
    assert b2.lr_at == ((res.stages[0]["steps"], stages[1].lr),) and len(b2.rows) == 96
    assert len(baseline_stages("b3", stages, res, st, BS)[0].rows) == len(stages[0].rows)
    with pytest.raises(ValueError, match="b1, b2 or b3"):
        baseline_stages("b4", stages, res, st, BS)
    assert default_eval_every(60_000, 256) == 100 and default_eval_every(1_000_000, 256) == 977


def test_unsupported_combinations_raise(data):  # noqa: F811
    st, stages = two_stage()
    t = make_trainer()
    with pytest.raises(ValueError, match="ema_decay"):
        run_staged(t, data, stages, st, FitSettings(epochs=1, batch_size=BS, ema_decay=0.9))
    t.group_channel = True
    with pytest.raises(ValueError, match="cross-section"):
        run_staged(t, data, stages, st, FitSettings(epochs=1, batch_size=BS))
