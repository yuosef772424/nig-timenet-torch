"""P32/P33/P33b: the ARCHITECTURES registry — pluggable models behind one boundary contract (core.model_contract).

Each entry: ``build_fn(seq_len, n_features, **config) -> nn.Module`` whose forward returns the HEAD_REGISTRY-named
output dict, plus a ``default_config`` (constructor kwargs without seq_len / n_features). ``nig_timenet_v2`` is the
first and default entry; its default_config is the first oracle's constructor_kwargs (reference/data/config.json).
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.heads import classification_head_forward
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans

__all__ = ["ARCHITECTURES", "DEFAULT_ARCHITECTURE", "ModelSettings", "build_model", "NIG_TIMENET_V2_DEFAULT_CONFIG",
           "TinyMLP", "build_tiny_mlp"]

ARCHITECTURES = Registry("architecture", Contract("architecture", required_meta=("build_fn", "default_config")))
DEFAULT_ARCHITECTURE = "nig_timenet_v2"

_HEADS = {t: ["nig_regression", "binary_classification"] for t in ("high", "low", "close")}

# reference/data/config.json constructor_kwargs minus seq_len/n_features (equality asserted in tests)
NIG_TIMENET_V2_DEFAULT_CONFIG: dict[str, Any] = {
    "d_model": 16, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2, "patch_len": 4, "stride": 2,
    "kernel_sizes": [3, 5, 9, 17], "max_rel_pos": 16, "dropout": 0.1, "attn_dropout": 0.0, "causal": False,
    "window": None, "use_instance_norm": True, "use_decomposition": True, "use_linear_path": True,
    "price_targets": ["high", "low", "close"], "head_types": _HEADS, "enforce_order": True, "head_hidden": 128,
    "nu_min": 0.1, "alpha_min": 2.0, "beta_min": 0.01, "unc_max": 20.0, "class_head_hidden": 64, "n_classes": 3,
    "norm_eps": 0.0001, "input_clip": None, "stats_mode": "full", "input_noise_std": 0.0, "feature_dropout": 0.0,
    "linear_path_l2": 0.0001, "level_passthrough": False, "level_norm": None,
}


@dataclass(frozen=True)
class ModelSettings:
    """Mirrors the TF side's ModelSettings.architecture (S23): which ARCHITECTURES entry to build, plus overrides
    of that entry's default_config."""
    architecture: str = DEFAULT_ARCHITECTURE
    overrides: dict[str, Any] = field(default_factory=dict)


def build_model(settings: ModelSettings, seq_len: int, n_features: int) -> nn.Module:
    meta = ARCHITECTURES.entry(settings.architecture).meta
    return meta["build_fn"](seq_len, n_features, **{**meta["default_config"], **settings.overrides})


ARCHITECTURES.register("nig_timenet_v2", build_nig_timenet_v2, version=1, build_fn=build_nig_timenet_v2,
                       default_config=NIG_TIMENET_V2_DEFAULT_CONFIG)


class TinyMLP(nn.Module):
    """P33 plug-in proof, deliberately trivial: flatten -> Linear -> GELU -> Linear trunk, then the SAME head
    modules nig_timenet_v2 uses (NIG regression + binary classification per target), so it returns the same
    HEAD_REGISTRY-named 24-key output dict. Not a candidate model."""

    def __init__(self, seq_len: int, n_features: int, price_targets=("high", "low", "close"), hidden: int = 32,
                 head_types: dict[str, list[str]] | None = None, dropout: float = 0.0, enforce_order: bool = True):
        super().__init__()
        self.price_targets = tuple(price_targets)
        head_types = head_types or {t: ["nig_regression", "binary_classification"] for t in self.price_targets}
        self.fc1, self.fc2 = nn.Linear(seq_len * n_features, hidden), nn.Linear(hidden, hidden)
        self.nig_targets = [t for t in self.price_targets if "nig_regression" in head_types.get(t, [])]
        self.class_targets = [t for t in self.price_targets if "binary_classification" in head_types.get(t, [])]
        self.nig = nn.ModuleDict({t: NIGHead(hidden, hidden, dropout) for t in self.nig_targets})
        self.unc = nn.ModuleDict({t: NIGUncertainty() for t in self.nig_targets})
        self.conf = nn.ModuleDict({t: ConfidenceHead(hidden) for t in self.nig_targets})
        self.apply_order = enforce_order and {"high", "low", "close"} <= set(self.nig_targets)
        self.ordered_means = OrderedMeans() if self.apply_order else None
        self.class_fc = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in self.class_targets})
        self.class_out = nn.ModuleDict({t: nn.Linear(hidden, 1) for t in self.class_targets})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fc2(F.gelu(self.fc1(x.flatten(1))))
        raw = {t: self.nig[t](h) for t in self.nig_targets}
        mus = {t: raw[t][0] for t in self.nig_targets}
        if self.apply_order:
            mus["high"], mus["close"], mus["low"] = self.ordered_means(mus["close"], mus["high"], mus["low"])
        out = {}
        for t in self.nig_targets:
            _, nu, alpha, beta = raw[t]
            epi, ale = self.unc[t](nu, alpha, beta)
            out.update({f"y_{t}": mus[t], f"y_{t}_nu": nu, f"y_{t}_alpha": alpha, f"y_{t}_beta": beta,
                        f"y_{t}_epistemic": epi, f"y_{t}_aleatoric": ale,
                        f"y_{t}_confidence": self.conf[t](h, nu, alpha, beta)})
        for t in self.class_targets:
            out[f"y_{t}_class_logits"] = classification_head_forward(h, self.class_fc[t], self.class_out[t])
        return out

    def regularization_loss(self) -> torch.Tensor:
        return sum((m.regularization_loss() for m in self.nig.values()), torch.zeros(()))


def build_tiny_mlp(seq_len: int, n_features: int, **config) -> TinyMLP:
    return TinyMLP(int(seq_len), int(n_features), **config)


ARCHITECTURES.register("tiny_mlp", build_tiny_mlp, version=1, build_fn=build_tiny_mlp,
                       default_config={"price_targets": ["high", "low", "close"], "hidden": 32})
