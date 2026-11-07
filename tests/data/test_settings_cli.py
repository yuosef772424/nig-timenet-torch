"""DataSettings is strict, round-trips and updates like the notebooks use it; the command line drives the same."""

import json

import pytest

from nig_timenet_torch.core import config as cfg
from nig_timenet_torch.data.cli import main, settings_from_args
from nig_timenet_torch.data.presets import PRESETS, preset
from nig_timenet_torch.data.settings import DataSettings, WindowSettings


def test_defaults_are_the_production_hourly_preset():
    s = DataSettings()
    assert s.windows.tf_order == ("1h",) and s.windows.window_sizes == {"1h": 32} and s.windows.stride == 8
    assert s.targets.reg_target_scale == 100.0 and s.targets.horizon == 1 and s.scaling.scaler_type == "robust"


def test_round_trip_through_a_dict_and_json():
    s = preset("hourly_4h_closed", "dated_splits")
    again = DataSettings.from_dict(json.loads(json.dumps(s.to_dict(), default=list)))
    assert again == s and again.windows.higher_tf_mode == "closed"


def test_unknown_keys_and_bad_values_raise_with_help():
    with pytest.raises(ValueError, match="did you mean 'stride'"):
        DataSettings().updated({"windows.strid": 4})
    with pytest.raises(ValueError, match="unknown key"):
        DataSettings.from_dict({"windows": {"nope": 1}})
    with pytest.raises(ValueError):
        WindowSettings(stride=0)
    with pytest.raises(ValueError, match="one of"):
        DataSettings().updated({"source.kind": "ftp"})


def test_updated_merges_nested_sections_and_leaves_the_original():
    s = DataSettings()
    t = s.updated({"features": {"breadth": {"enabled": False}}, "windows.stride": 4})
    assert t.features.breadth.enabled is False and t.windows.stride == 4 and s.windows.stride == 8
    assert t.features.market == s.features.market
    assert cfg.to_dict(t)["windows"]["stride"] == 4


def test_presets_apply_in_order_and_unknown_names_fail():
    assert preset("hourly_w32_s8") == DataSettings()
    assert preset("tiny").source.kind == "synthetic" and set(PRESETS) >= {"tiny", "hourly_4h_closed"}
    with pytest.raises(KeyError, match="known"):
        preset("nope")


def test_cli_builds_writes_and_shows(tmp_path, capsys):
    out = tmp_path / "art"
    sha = "cd" * 20
    assert main(["build", "--preset", "tiny", "--set", f"output.path={out}", "--set", f"output.source_commit={sha}",
                 "--set", 'windows.window_sizes={"1h": 16}']) == 0
    assert "written to" in capsys.readouterr().out and (out / "manifest.json").exists()
    assert main(["show", "--preset", "tiny"]) == 0
    assert json.loads(capsys.readouterr().out)["source"]["kind"] == "synthetic"


def test_cli_config_file_then_overrides(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"windows": {"stride": 16}}))
    s = settings_from_args(["tiny"], str(f), ["windows.stride=2"])
    assert s.windows.stride == 2 and s.source.kind == "synthetic"
    with pytest.raises(SystemExit):
        settings_from_args([], "", ["no_equals_sign"])


def test_updated_helper_on_a_plain_dataclass():
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Inner:
        a: int = 1

    @dataclass(frozen=True)
    class Outer:
        inner: Inner = field(default_factory=Inner)
        b: int = 2

    o = cfg.updated(Outer(), {"inner.a": 5}, b=3)
    assert o == Outer(Inner(5), 3)
    with pytest.raises(ValueError, match="b is not a settings section"):
        cfg.updated(Outer(), {"b.x": 1})
