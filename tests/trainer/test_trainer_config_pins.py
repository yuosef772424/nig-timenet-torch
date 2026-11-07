"""Pins of the baseline job's trainer config and of the RunIdentity built from it (hardening H4, rule B2).

PURPOSE:  Proves that moving the config construction (``targets_config``) and the trainer's validation / defaults keeps
  the resolved trainer config dict and the run identity fingerprint byte-identical: a checkpoint of an existing run
  resumes only while its identity is unchanged.
TAGS:     trainer config golden pin run identity fingerprint refactor baseline job
PITFALLS: The pin file is tests/pins/trainer_config.json; regenerate it only for a deliberate change of the baseline
  objective (``UPDATE_TRAINER_PINS=1 pytest tests/trainer/test_trainer_config_pins.py``) and list that as a behaviour
  change. ``identity_fingerprint`` also depends on the torch AdamW defaults (a torch upgrade can move it, see
  colab/identity.py); ``trainer_hash`` does not.
"""

import hashlib
import json
import os
from pathlib import Path

import torch

from nig_timenet_torch.colab.identity import identity_for_run
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.jobs import baseline
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from tests.data_io.test_artifact import art, write_manifest  # noqa: F401  (fixture)
from tests.trainer.fixture_setup import build_model

PIN = Path(__file__).resolve().parents[1] / "pins" / "trainer_config.json"
TARGET_SETS = {"high_low_close": ("high", "low", "close"), "close_only": ("close",)}


def _hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity_fingerprint(manifest, cfg: dict) -> str:
    torch.manual_seed(0)
    model = build_model(16, 18, dropout=0.1)
    trainer = Trainer(model, cfg, optimizer=lambda ps: torch.optim.AdamW(ps, lr=baseline.LR,
                                                                         weight_decay=baseline.WEIGHT_DECAY))
    ms = ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG))
    return identity_for_run(trainer, FitSettings(epochs=5, batch_size=64, seed=0, trace=True), ms, manifest,
                            "0" * 40).fingerprint()


def _current(manifest) -> dict:
    out = {}
    for name, targets in TARGET_SETS.items():
        cfg = baseline.trainer_config(targets)
        out[name] = {"config": cfg, "trainer_hash": _hash(cfg),
                     "identity_fingerprint": _identity_fingerprint(manifest, cfg) if len(targets) == 3 else None}
    return out


def test_baseline_trainer_config_and_identity_are_pinned(art):  # noqa: F811
    cur = _current(load_artifact(art).manifest)
    if os.environ.get("UPDATE_TRAINER_PINS"):
        PIN.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    pinned = json.loads(PIN.read_text(encoding="utf-8"))
    for name in TARGET_SETS:
        assert cur[name]["config"] == pinned[name]["config"], f"{name}: the resolved trainer config changed"
        assert cur[name]["trainer_hash"] == pinned[name]["trainer_hash"]
        assert cur[name]["identity_fingerprint"] == pinned[name]["identity_fingerprint"], (
            f"{name}: the RunIdentity of the baseline run changed: existing runs would refuse to resume")
