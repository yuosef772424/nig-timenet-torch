"""Pins of the model assembly (hardening H3, rule B2): sorted ``state_dict`` keys + shapes AND a forward-output
fingerprint (seeded build, fixed input, eval mode) for encoder, input-skip, timeframe, head and architecture variants.

PURPOSE:  Proves that refactors of model/builder.py, encoders, heads and architectures keep every parameter name
  (the oracle weights.json attr_paths load by them) and every output value.
TAGS:     model state_dict fingerprint refactor pin builder encoder head
PITFALLS: The pin file is tests/pins/model_assembly.json; regenerate it only for a deliberate model change
  (``UPDATE_MODEL_PINS=1 pytest tests/model/test_assembly_pins.py``) and list that change as a behaviour change.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from nig_timenet_torch.model.architectures import ARCHITECTURES, ModelSettings, build_model

PIN = Path(__file__).resolve().parents[1] / "pins" / "model_assembly.json"
_BASE = {"d_model": 8, "num_layers": 2, "num_heads": 2, "num_kv_heads": 1, "patch_len": 2, "stride": 1,
         "kernel_sizes": [3, 5], "max_rel_pos": 4, "head_hidden": 8, "class_head_hidden": 6}
_NIG = {t: ["nig_regression"] for t in ("high", "low", "close")}
_BIN = {t: ["nig_regression", "binary_classification"] for t in ("high", "low", "close")}
_MULTI = {t: ["nig_regression", "multiclass_classification"] for t in ("high", "low", "close")}
_ANTI = {"encoder": "gru", "stats_mode": "symlog", "input_clip": 4.0, "input_noise_std": 0.1,
         "feature_dropout": 0.1, "level_passthrough": True, "level_norm": "batch"}

CASES = {
    "transformer": ("nig_timenet_v2", {}), "gru": ("nig_timenet_v2", {"encoder": "gru"}),
    "tcn": ("nig_timenet_v2", {"encoder": "tcn"}),
    "skip_concat_each_block": ("nig_timenet_v2", {"input_skip": "concat_each_block"}),
    "skip_dense_skip": ("nig_timenet_v2", {"input_skip": "dense_skip"}),
    "skip_late_raw": ("nig_timenet_v2", {"input_skip": "late_raw"}),
    "skip_no_proj": ("nig_timenet_v2", {"input_skip": "no_proj"}),
    "binary_heads": ("nig_timenet_v2", {"head_types": _BIN}),
    "multiclass_heads": ("nig_timenet_v2", {"head_types": _MULTI, "n_classes": 3}),
    "anti_memorization": ("nig_timenet_v2", _ANTI),
    "anti_memorization_transformer": ("nig_timenet_v2", {**_ANTI, "encoder": "transformer"}),
    "two_timeframes_transformer": ("nig_timenet_v2", {"timeframes": True}),
    "two_timeframes_tcn": ("nig_timenet_v2", {"timeframes": True, "encoder": "tcn"}),
    "tiny_mlp": ("tiny_mlp", {}),
}


def _build(case):
    arch, over = CASES[case]
    over = dict(over)
    multi = over.pop("timeframes", False)
    cfg = {} if arch == "tiny_mlp" else {**_BASE, "head_types": _NIG}
    cfg.update(over)
    if arch == "tiny_mlp":
        cfg.setdefault("hidden", 8)
    torch.manual_seed(1234)
    if multi:
        seq, nf = {"1h": 8, "4h": 6}, {"1h": 4, "4h": 3}
        model = build_model(ModelSettings(arch, cfg), seq, nf)
        g = torch.Generator().manual_seed(7)
        x = {"1h": torch.randn(3, 8, 4, generator=g), "4h": torch.randn(3, 6, 3, generator=g)}
    else:
        model = build_model(ModelSettings(arch, cfg), 8, 4)
        x = torch.randn(3, 8, 4, generator=torch.Generator().manual_seed(7))
    return model.eval(), x


def _snapshot(case):
    model, x = _build(case)
    with torch.no_grad():
        out = model(x)
    return {"state_dict": {k: list(v.shape) for k, v in sorted(model.state_dict().items())},
            "outputs": {k: [round(float(v.double().sum()), 5), round(float(v.double().abs().max()), 5),
                            list(v.shape)] for k, v in sorted(out.items())}}


def _baseline_config():
    from nig_timenet_torch.jobs.baseline import model_settings

    meta = ARCHITECTURES.entry("nig_timenet_v2").meta
    return json.loads(json.dumps({**meta["default_config"], **model_settings(("high", "low", "close")).overrides}))


def _pins():
    return json.loads(PIN.read_text()) if PIN.exists() else {}


def test_regenerate_pins_when_asked():
    if os.environ.get("UPDATE_MODEL_PINS") != "1":
        pytest.skip("set UPDATE_MODEL_PINS=1 to rewrite tests/pins/model_assembly.json")
    pins = {c: _snapshot(c) for c in CASES}
    pins["baseline_job_resolved_config"] = _baseline_config()
    PIN.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("case", sorted(CASES))
def test_state_dict_keys_and_shapes_pinned(case):
    assert _snapshot(case)["state_dict"] == _pins()[case]["state_dict"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_forward_fingerprint_pinned(case):
    got, want = _snapshot(case)["outputs"], _pins()[case]["outputs"]
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k][2] == want[k][2], k
        assert got[k][:2] == pytest.approx(want[k][:2], abs=2e-4), k


def test_baseline_job_resolved_model_config_pinned():
    """The effective model config of the baseline job (registry default merged with the job's preset) is unchanged."""
    assert _baseline_config() == _pins()["baseline_job_resolved_config"]
