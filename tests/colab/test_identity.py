"""colab/identity.py: one place builds a run's RunIdentity, from the built Trainer (T2 review note)."""

import ast

import pytest
import torch

from nig_timenet_torch.colab.identity import identity_for_run
from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer.identity import build_run_identity
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from tests.data_io.test_artifact import art, write_manifest  # noqa: F401  (fixture)
from tests.scan import sources
from tests.trainer.fixture_setup import build_model, trainer_config
from tests.trainer.test_fit import make_trainer


def test_identity_is_built_from_the_trainer_and_includes_the_kendall_group(art):  # noqa: F811
    manifest = load_artifact(art).manifest
    seed_everything(0)
    model = build_model(16, 18, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fit, ms = FitSettings(epochs=2, batch_size=32), ModelSettings("tiny_mlp")
    early = build_run_identity(fit, ms, trainer_config(), opt, manifest.data_fingerprint, code_sha="c" * 40)
    t = Trainer(model, trainer_config(), optimizer=opt)  # adds the Kendall log-variances to `opt` as a second group
    assert t.uncertainty_layer is not None and len(opt.param_groups) == 2, "fixture config must use Kendall"
    ident = identity_for_run(t, fit, ms, manifest, "c" * 40)
    assert ident == build_run_identity(fit, ms, t.config, opt, manifest.data_fingerprint, code_sha="c" * 40)
    assert ident.data_fingerprint == manifest.data_fingerprint
    assert early.settings_hash != ident.settings_hash, "an identity built before the Trainer is another optimizer"


def test_a_config_instead_of_a_trainer_and_a_foreign_optimizer_are_refused(art):  # noqa: F811
    manifest = load_artifact(art).manifest
    t = make_trainer()
    with pytest.raises(TypeError, match="takes the built Trainer"):
        identity_for_run(t.config, FitSettings(epochs=1), ModelSettings("tiny_mlp"), manifest, "c" * 40)
    t.optimizer = torch.optim.AdamW(t.model.parameters(), lr=1e-3)  # lost the Kendall group
    with pytest.raises(ValueError, match="does not hold"):
        identity_for_run(t, FitSettings(epochs=1), ModelSettings("tiny_mlp"), manifest, "c" * 40)
    t.optimizer = None
    with pytest.raises(ValueError, match="no optimizer"):
        identity_for_run(t, FitSettings(epochs=1), ModelSettings("tiny_mlp"), manifest, "c" * 40)


def test_nothing_else_in_src_builds_an_identity():
    """B3 guard: build_run_identity / RunIdentity(...) are called only where defined and in colab/identity.py."""
    allowed = {"trainer/identity.py", "colab/identity.py"}
    offenders = []
    for rel, (_, tree) in sources().items():
        if rel in allowed:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                if name in ("build_run_identity", "RunIdentity"):
                    offenders.append(f"{rel}:{n.lineno}: {name}(...)")
    assert not offenders, "use colab.identity.identity_for_run:\n  " + "\n  ".join(offenders)
