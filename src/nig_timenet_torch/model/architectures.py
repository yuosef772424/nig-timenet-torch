"""P32/P33/P33b: the ARCHITECTURES registry — pluggable models behind one boundary contract (core.model_contract).

Each entry: ``build_fn(seq_len, n_features, **config) -> nn.Module`` whose forward returns the HEAD_REGISTRY-named
output dict, plus a ``default_config`` (constructor kwargs without seq_len / n_features). ``nig_timenet_v2`` is the
first and default entry; its default_config is the first oracle's constructor_kwargs (reference/data/config.json).
"""

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.model.builder import build_nig_timenet_v2

__all__ = ["ARCHITECTURES", "DEFAULT_ARCHITECTURE", "ModelSettings", "build_model", "NIG_TIMENET_V2_DEFAULT_CONFIG"]

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
