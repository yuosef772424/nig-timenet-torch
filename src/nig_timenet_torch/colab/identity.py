"""The ONE place a real run's ``RunIdentity`` is built (T2 review, deferred to T3).

``Trainer.__init__`` adds the Kendall log-variances to the optimizer as an extra param group, so an identity built from
the optimizer BEFORE the Trainer exists describes a different optimizer than the one that trains (and resume would
compare the wrong thing). ``identity_for_run`` therefore takes the built ``Trainer`` itself: it cannot be called
earlier, and it refuses an optimizer that does not hold every parameter the trainer trains. Job entries call this and
nothing else; ``tests/colab/test_identity.py`` scans ``src/`` for any other caller of ``build_run_identity``.

A torch upgrade can change ``optimizer.defaults`` (new keys, new default values), which changes ``settings_hash``:
a checkpoint written under the old torch is then refused by ``Trainer.fit`` (``ResumeMismatchError`` naming
``settings_hash``). That is intended, the identity cannot tell a harmless default from a real change; start a new run.
"""

from nig_timenet_torch.data_io.artifact import ArtifactManifest
from nig_timenet_torch.trainer.identity import RunIdentity, build_run_identity
from nig_timenet_torch.trainer.trainer import Trainer


def identity_for_run(trainer: Trainer, fit_settings, model_settings, manifest: ArtifactManifest,
                     code_sha: str) -> RunIdentity:
    """Identity of the run ``trainer`` is about to fit on the artifact described by ``manifest``."""
    if not isinstance(trainer, Trainer):
        raise TypeError("identity_for_run takes the built Trainer (its optimizer holds the Kendall group only after "
                        f"construction), got {type(trainer).__name__}")
    if trainer.optimizer is None:
        raise ValueError("the trainer has no optimizer: a run identity needs one")
    held = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    missing = [tuple(p.shape) for p in trainer.parameters() if id(p) not in held]
    if missing:
        raise ValueError(f"the optimizer does not hold {len(missing)} trainer parameter(s) (shapes {missing[:3]}): the "
                         f"identity would describe an optimizer that does not train them")
    return build_run_identity(fit_settings, model_settings, trainer.config, trainer.optimizer,
                              manifest.data_fingerprint, code_sha=code_sha)
