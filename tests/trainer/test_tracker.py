"""T1: BestTracker (best epoch + early stopping), the numbers only."""

import math

import pytest

from nig_timenet_torch.trainer.tracker import BestTracker, EarlyStopSettings


def feed(tracker, values):
    return [tracker.update(i + 1, v) for i, v in enumerate(values)]


def test_min_mode_tracks_the_best_and_waits():
    t = BestTracker(EarlyStopSettings(patience=3, smoothing="none"), higher_is_better=False)
    assert feed(t, [5.0, 4.0, 4.5, 4.2]) == [True, True, False, False]
    assert (t.best, t.best_epoch, t.wait) == (4.0, 2, 2)
    assert not t.should_stop()
    t.update(5, 4.1)
    assert t.should_stop()


def test_max_mode_follows_the_metric_direction():
    t = BestTracker(EarlyStopSettings(patience=2, smoothing="none"), higher_is_better=True)
    assert feed(t, [0.5, 0.6, 0.55]) == [True, True, False]
    assert (t.best, t.best_epoch) == (0.6, 2)


def test_min_delta_demands_a_real_improvement():
    t = BestTracker(EarlyStopSettings(min_delta=0.1, smoothing="none"), higher_is_better=False)
    assert feed(t, [1.0, 0.95, 0.85]) == [True, False, True]


def test_window_smoothing_compares_the_mean_of_the_last_epochs():
    t = BestTracker(EarlyStopSettings(smoothing="window", window=2, min_delta=0.0), higher_is_better=False)
    feed(t, [4.0, 2.0])               # smoothed: 4.0, 3.0
    assert t.best == 3.0
    assert t.update(3, 6.0) is False  # mean(2, 6) = 4.0 is worse although 2.0 was the raw best


def test_ema_smoothing_matches_the_formula():
    t = BestTracker(EarlyStopSettings(smoothing="ema", ema_beta=0.5, min_delta=0.0), higher_is_better=False)
    feed(t, [4.0, 2.0, 1.0])          # s1 = 4, s2 = 0.5*4 + 0.5*2 = 3, s3 = 0.5*3 + 0.5*1 = 2
    assert t.ema == 2.0 and t.best == 2.0


def test_non_finite_values_are_ignored():
    t = BestTracker(EarlyStopSettings(smoothing="none", patience=1), higher_is_better=False)
    t.update(1, 1.0)
    assert t.update(2, float("nan")) is False and t.update(3, math.inf) is False and t.update(4, None) is False
    assert t.wait == 0 and not t.should_stop()


def test_state_round_trip_continues_identically():
    a = BestTracker(EarlyStopSettings(smoothing="ema"), higher_is_better=False)
    feed(a, [3.0, 2.5, 2.6])
    b = BestTracker(EarlyStopSettings(smoothing="ema"), higher_is_better=False)
    b.load_state(a.get_state())
    assert feed(a, [2.4, 2.7]) == feed(b, [2.4, 2.7]) and a.get_state() == b.get_state()
    with pytest.raises(ValueError, match="opposite"):
        BestTracker(EarlyStopSettings(), higher_is_better=True).load_state(a.get_state())


def test_bad_settings_raise():
    with pytest.raises(ValueError, match="unknown smoothing"):
        EarlyStopSettings(smoothing="median")
    with pytest.raises(ValueError):
        EarlyStopSettings(patience=0)
