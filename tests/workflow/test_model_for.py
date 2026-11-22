"""model_for: the one rule for "model overrides merge over the default preset" (C-20)."""

from nig_timenet_torch.workflow.settings import default_model_settings, model_for


def test_default_architecture_merges_over_the_preset():
    base = default_model_settings()
    ms = model_for(None, {"bn_warmup": False})
    assert ms.architecture == base.architecture
    assert ms.overrides == {**base.overrides, "bn_warmup": False}
    assert model_for(base.architecture, None).overrides == base.overrides


def test_other_architecture_or_no_preset_starts_clean():
    assert model_for("tiny_mlp", {"a": 1}).overrides == {"a": 1}
    assert model_for(None, {"a": 1}, preset=False).overrides == {"a": 1}
