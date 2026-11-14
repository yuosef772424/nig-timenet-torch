"""P32/P33/P33b: the ARCHITECTURES registry — pluggable models behind one boundary contract (model.contract).

PURPOSE:  The ARCHITECTURES registry (build_fn + default_config per architecture), ModelSettings / build_model, the
          anti-memorization preset and the TinyMLP plug-in proof; every architecture exposes the health-report hooks.
TAGS:     architectures, registry, model settings, default config, tiny mlp, plugin, health report, probe points
PITFALLS: THE default of nig_timenet_v2 is NIG_TIMENET_V2_DEFAULT_CONFIG (d_model 16, 2 layers, NIG + binary heads);
          build_nig_timenet_v2's own signature mirrors TF and differs (HF-08). An architecture's model must provide
          ``head_modules() -> {target: NIGHead}`` and ``default_probe_points`` (module names) for health_report;
          optional ``activation_points`` (pre-activation Linear names) for gate A7.

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
from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model.builder import build_nig_timenet_v2
from nig_timenet_torch.model.contract import ModelContract, contract_of
from nig_timenet_torch.model.heads import (
    HEAD_REGISTRY,
    class_head_kind,
    class_outputs,
    head_spec,
    head_width,
    kinds_with_task,
    nig_outputs,
)
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans

__all__ = ["ARCHITECTURES", "DEFAULT_ARCHITECTURE", "ModelSettings", "build_model", "NIG_TIMENET_V2_DEFAULT_CONFIG",
           "TinyMLP", "build_tiny_mlp", "ANTI_MEMORIZATION_CONFIG", "architecture_contract"]

ARCHITECTURES = Registry("architecture", Contract("architecture", required_meta=("build_fn", "default_config")))
DEFAULT_ARCHITECTURE = "nig_timenet_v2"

_HEADS = {t: ["nig_regression", "binary_classification"] for t in TARGET_COLUMNS}

# reference/data/config.json constructor_kwargs minus seq_len/n_features (equality asserted in tests)
NIG_TIMENET_V2_DEFAULT_CONFIG: dict[str, Any] = {
    "d_model": 16, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2, "patch_len": 4, "stride": 2,
    "kernel_sizes": [3, 5, 9, 17], "max_rel_pos": 16, "dropout": 0.1, "attn_dropout": 0.0, "causal": False,
    "window": None, "use_instance_norm": True, "use_decomposition": True, "use_linear_path": True,
    "price_targets": list(TARGET_COLUMNS), "head_types": _HEADS, "enforce_order": True, "head_hidden": 128,
    "nu_min": 0.1, "alpha_min": 2.0, "beta_min": 0.01, "unc_max": 20.0, "class_head_hidden": 64, "n_classes": 3,
    "norm_eps": 0.0001, "input_clip": None, "stats_mode": "full", "input_noise_std": 0.0, "feature_dropout": 0.0,
    "linear_path_l2": 0.0001, "level_passthrough": False, "level_norm": None,
}


# P33b: the TF side's ANTI_MEMORIZATION_CONFIG (model/config.py), a preset over the same builder (merged over
# default_config), equal to reference/data_default/config.json "overrides" (asserted in tests)
ANTI_MEMORIZATION_CONFIG: dict[str, Any] = {
    "d_model": 64, "num_layers": 2, "head_hidden": 64, "class_head_hidden": 32, "dropout": 0.25, "input_clip": 4.0,
    "stats_mode": "symlog", "input_noise_std": 0.1, "feature_dropout": 0.1, "linear_path_l2": 1e-3,
    "level_passthrough": True, "level_norm": "batch", "encoder": "gru",
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


def architecture_contract(settings: ModelSettings, seq_len: int, n_features: int) -> ModelContract:
    """The boundary contract of the model ``settings`` builds (one eval-mode forward on zeros). A new architecture
    is checked with ``check_contract(its_model, architecture_contract(reference_settings, T, F), x)``."""
    return contract_of(build_model(settings, seq_len, n_features), torch.zeros(1, seq_len, n_features))


ARCHITECTURES.register("nig_timenet_v2", build_nig_timenet_v2, version=1, build_fn=build_nig_timenet_v2,
                       default_config=NIG_TIMENET_V2_DEFAULT_CONFIG,
                       anti_memorization_config=ANTI_MEMORIZATION_CONFIG)


class TinyMLP(nn.Module):
    """P33 plug-in proof, deliberately trivial: flatten -> Linear -> GELU -> Linear trunk, then the SAME head
    modules nig_timenet_v2 uses (NIG regression + binary classification per target), so it returns the same
    HEAD_REGISTRY-named 24-key output dict. Not a candidate model."""

    default_probe_points = ("fc2",)  # the last trunk layer

    def __init__(self, seq_len: int, n_features: int, price_targets=TARGET_COLUMNS, hidden: int = 32,
                 head_types: dict[str, list[str]] | None = None, dropout: float = 0.0, enforce_order: bool = True,
                 order_detach_close: bool = False, nig_scale: float | None = None):
        super().__init__()
        self.price_targets = tuple(price_targets)
        head_types = head_types or {t: ["nig_regression", "binary_classification"] for t in self.price_targets}
        self.fc1, self.fc2 = nn.Linear(seq_len * n_features, hidden), nn.Linear(hidden, hidden)
        self.nig_targets = [t for t in self.price_targets if kinds_with_task(head_types.get(t, []), "evidential")]
        self.class_kind = {t: k for t in self.price_targets if (k := class_head_kind(head_types.get(t, []), t))}
        if any(head_width(k, 1) != 1 for k in self.class_kind.values()):
            raise NotImplementedError("TinyMLP builds single-output (binary) classification heads only")
        self.class_targets = list(self.class_kind)
        self.nig = nn.ModuleDict({t: NIGHead(hidden, hidden, dropout, scale=nig_scale) for t in self.nig_targets})
        self.unc = nn.ModuleDict({t: NIGUncertainty() for t in self.nig_targets})
        self.conf = nn.ModuleDict({t: ConfidenceHead(hidden) for t in self.nig_targets})
        self.apply_order = enforce_order and set(TARGET_COLUMNS) <= set(self.nig_targets)
        self.ordered_means = (OrderedMeans(nig_scale, order_detach_close) if self.apply_order
                              else None)
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
            out.update(nig_outputs(head_spec("nig_regression", t), mus[t], nu, alpha, beta, epi, ale,
                                   self.conf[t](h, nu, alpha, beta)))
        for t, kind in self.class_kind.items():
            probs = HEAD_REGISTRY.get(kind)(h, self.class_fc[t], self.class_out[t])
            out.update(class_outputs(head_spec(kind, t), probs))
        return out

    def head_modules(self) -> dict[str, NIGHead]:
        return dict(self.nig.items())

    @property
    def activation_points(self) -> tuple[str, ...]:
        """Linear layers whose output feeds a ReLU-family activation (gate A7)."""
        return ("fc1", *(f"nig.{t}.fc" for t in self.nig_targets), *(f"conf.{t}.d1" for t in self.nig_targets),
                *(f"class_fc.{t}" for t in self.class_targets))

    def regularization_loss(self) -> torch.Tensor:
        return sum((m.regularization_loss() for m in self.head_modules().values()), torch.zeros(()))


def build_tiny_mlp(seq_len: int, n_features: int, **config) -> TinyMLP:
    return TinyMLP(int(seq_len), int(n_features), **config)


ARCHITECTURES.register("tiny_mlp", build_tiny_mlp, version=1, build_fn=build_tiny_mlp,
                       default_config={"price_targets": list(TARGET_COLUMNS), "hidden": 32})


# chart_cnn (owner 2026-11-10: a structurally different model that uses none of the project's layers)
from nig_timenet_torch.model.chart_cnn import CHART_CNN_DEFAULT_CONFIG, build_chart_cnn  # noqa: E402

ARCHITECTURES.register("chart_cnn", build_chart_cnn, version=1, build_fn=build_chart_cnn,
                       default_config=CHART_CNN_DEFAULT_CONFIG)

# conv1d_net / no_dense_net (owner 2026-11-10: change the structure completely to find where memorization starts)
from nig_timenet_torch.model.scratch_nets import (  # noqa: E402
    CONV1D_NET_DEFAULT_CONFIG,
    NO_DENSE_NET_DEFAULT_CONFIG,
    build_conv1d_net,
    build_no_dense_net,
)

ARCHITECTURES.register("conv1d_net", build_conv1d_net, version=1, build_fn=build_conv1d_net,
                       default_config=CONV1D_NET_DEFAULT_CONFIG)
ARCHITECTURES.register("no_dense_net", build_no_dense_net, version=1, build_fn=build_no_dense_net,
                       default_config=NO_DENSE_NET_DEFAULT_CONFIG)
