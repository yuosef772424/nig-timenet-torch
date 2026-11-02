"""build_nig_timenet_v2 / NigTimeNetV2: the full model, wired as crypto-signal-prediction model/builder.py
@ aeafc564 builds the single-timeframe "fused" transformer (BUILD_PLAN.md P15).

Module attribute names equal the Keras layer names, so weights.json ``attr_path`` strings map 1:1 onto torch
parameter names (see ``torch_param_name``). Options of the upstream builder that this port does not implement yet
are accepted and validated: any non-default value raises NotImplementedError naming the step that adds it.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.model.heads import classification_head_forward
from nig_timenet_torch.model.layers import (
    AttentionPool,
    CausalMultiScaleDecomp,
    InstanceNorm,
    LastToken,
    PatchEmbedding,
    RMSNorm,
    TransformerBlock,
    keras_conv1d_weight_to_torch,
    keras_dense_weight_to_torch,
)
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans

__all__ = ["DEFAULT_HEAD_TYPES", "NigTimeNetV2", "build_nig_timenet_v2", "torch_param_name"]

DEFAULT_HEAD_TYPES = {"high": ["nig_regression"], "low": ["nig_regression"], "close": ["nig_regression"]}
HEAD_KINDS = ("nig_regression", "binary_classification")

# option -> (value this port implements, step that adds the rest)
_NOT_YET = {
    "encoder": ("transformer", "P34"), "input_skip": ("none", "P34"),
    "use_instance_norm": (True, "P34"), "use_decomposition": (True, "P34"), "use_linear_path": (True, "P34"),
    "input_clip": (None, "P33b"), "stats_mode": ("full", "P33b"), "input_noise_std": (0.0, "P33b"),
    "feature_dropout": (0.0, "P33b"), "level_passthrough": (False, "P33b"), "level_norm": (None, "P33b"),
    "level_film": (False, "P34"), "architecture": ("fused", "P34"), "fusion": ("concat", "P34"), "n_coins": (0, "P34"),
}


def torch_param_name(attr_path: str) -> str:
    """weights.json attr_path -> torch parameter name (Keras 'kernel' -> torch 'weight'; 'pos_emb' -> 'pos')."""
    module, _, leaf = attr_path.rpartition(".")
    leaf = {"kernel": "weight", "pos_emb": "pos"}.get(leaf, leaf)
    return f"{module}.{leaf}"


def keras_to_torch_value(attr_path: str, value: torch.Tensor) -> torch.Tensor:
    if attr_path.endswith(".kernel"):
        return keras_conv1d_weight_to_torch(value) if value.ndim == 3 else keras_dense_weight_to_torch(value)
    return value


class NigTimeNetV2(nn.Module):
    def __init__(self, seq_len: int, n_features: int, *, d_model, num_layers, num_heads, num_kv_heads, patch_len,
                 stride, kernel_sizes, max_rel_pos, dropout, attn_dropout, causal, window, price_targets,
                 head_types, enforce_order, head_hidden, nu_min, alpha_min, beta_min, unc_max, class_head_hidden,
                 norm_eps, linear_path_l2):
        super().__init__()
        self.price_targets = tuple(price_targets)
        self.head_types = {t: list(head_types.get(t, [])) for t in self.price_targets}
        for t, kinds in self.head_types.items():
            bad = [k for k in kinds if k not in HEAD_KINDS]
            if bad:
                raise NotImplementedError(f"head types {bad} for {t!r} are not ported yet (P17 HEAD_REGISTRY)")
        self.linear_path_l2 = linear_path_l2  # loss-side L2, consumed by the trainer (M3)
        self.num_layers = num_layers

        self.instance_norm = InstanceNorm(n_features, eps=norm_eps)
        self.decomp = CausalMultiScaleDecomp(n_features, kernel_sizes)
        self.patch_embed = PatchEmbedding(2 * n_features, d_model, patch_len, stride, seq_len=seq_len)
        for i in range(num_layers):
            setattr(self, f"block_{i + 1}", TransformerBlock(d_model, num_heads, num_kv_heads, max_rel_pos, causal,
                                                             window, dropout, attn_dropout))
        self.final_norm = RMSNorm(d_model)
        self.last_token = LastToken()
        self.attn_pool = AttentionPool(d_model)
        self.stats_proj = nn.Linear(2 * n_features, d_model)
        self.readout_fc = nn.Linear(3 * d_model, d_model)
        self.linear_path = nn.Linear(seq_len * n_features, d_model)
        self.trunk_norm = RMSNorm(d_model)
        self.trunk_drop = nn.Dropout(dropout)

        self.nig_targets = [t for t in self.price_targets if "nig_regression" in self.head_types[t]]
        for t in self.nig_targets:
            setattr(self, f"nig_{t}", NIGHead(d_model, head_hidden, dropout, nu_min, alpha_min, beta_min))
            setattr(self, f"unc_{t}", NIGUncertainty(unc_max))
            setattr(self, f"conf_{t}", ConfidenceHead(d_model))
        self.apply_order = enforce_order and {"high", "low", "close"} <= set(self.nig_targets)
        if self.apply_order:
            self.ordered_means = OrderedMeans()
        self.class_targets = [t for t in self.price_targets if "binary_classification" in self.head_types[t]]
        for t in self.class_targets:
            setattr(self, f"class_fc_{t}", nn.Linear(d_model, class_head_hidden))
            setattr(self, f"class_drop_{t}", nn.Dropout(dropout))
            setattr(self, f"y_{t}_class_logits", nn.Linear(class_head_hidden, 1))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        z, stats = self.instance_norm(x)
        trend, seasonal = self.decomp(z)
        h = self.patch_embed(torch.cat([trend, seasonal], dim=-1))
        for i in range(self.num_layers):
            h = getattr(self, f"block_{i + 1}")(h)
        h = self.final_norm(h)
        parts = [self.last_token(h), self.attn_pool(h), F.gelu(self.stats_proj(stats))]
        h = F.gelu(self.readout_fc(torch.cat(parts, dim=-1))) + self.linear_path(z.flatten(1))
        return self.trunk_drop(self.trunk_norm(h))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        raw = {t: getattr(self, f"nig_{t}")(h) for t in self.nig_targets}
        if self.apply_order:
            mu_h, mu_c, mu_l = self.ordered_means(raw["close"][0], raw["high"][0], raw["low"][0])
            mus = {"high": mu_h, "close": mu_c, "low": mu_l}
        else:
            mus = {t: raw[t][0] for t in self.nig_targets}
        out: dict[str, torch.Tensor] = {}
        for t in self.nig_targets:
            _, nu, alpha, beta = raw[t]
            epi, ale = getattr(self, f"unc_{t}")(nu, alpha, beta)
            out.update({f"y_{t}": mus[t], f"y_{t}_nu": nu, f"y_{t}_alpha": alpha, f"y_{t}_beta": beta,
                        f"y_{t}_epistemic": epi, f"y_{t}_aleatoric": ale,
                        f"y_{t}_confidence": getattr(self, f"conf_{t}")(h, nu, alpha, beta)})
        for t in self.class_targets:
            out[f"y_{t}_class_logits"] = classification_head_forward(
                h, getattr(self, f"class_fc_{t}"), getattr(self, f"y_{t}_class_logits"),
                getattr(self, f"class_drop_{t}"))
        return out

    @torch.no_grad()
    def load_oracle_weights(self, weights: dict[str, torch.Tensor], attr_paths: dict[str, str] | None = None) -> None:
        """Assign every parameter from ``weights`` keyed by attr_path (or by weights.json name, translated through
        ``attr_paths`` name -> attr_path). 'trainer/...' entries are skipped. Raises on any unmatched parameter,
        unknown key, or shape mismatch — no silent partial load."""
        attr_paths = attr_paths or {}
        params = dict(self.named_parameters())
        assigned = set()
        for key, value in weights.items():
            path = attr_paths.get(key, key)
            if path.startswith("trainer/"):
                continue
            name = torch_param_name(path)
            if name not in params:
                raise KeyError(f"weight {key!r} (attr_path {path!r}) has no torch parameter {name!r}")
            v = keras_to_torch_value(path, value)
            if v.shape != params[name].shape:
                raise ValueError(f"{name}: shape {tuple(v.shape)} != {tuple(params[name].shape)}")
            params[name].copy_(v)
            assigned.add(name)
        missing = sorted(set(params) - assigned)
        if missing:
            raise KeyError(f"parameters not assigned by the oracle weights: {missing}")


def build_nig_timenet_v2(seq_len: int, n_features: int, *, d_model=128, num_layers=4, num_heads=4, num_kv_heads=2,
                         patch_len=4, stride=2, kernel_sizes=(3, 5, 9, 17), max_rel_pos=16, dropout=0.1,
                         attn_dropout=0.0, causal=False, window=None, price_targets=("high", "low", "close"),
                         head_types: dict[str, list[str]] | None = None, enforce_order=True, head_hidden=128,
                         nu_min=0.1, alpha_min=2.0, beta_min=0.01, unc_max=20.0, class_head_hidden=64,
                         encoder: str = "transformer", input_skip: str = "none", n_classes=3, norm_eps=1e-4,
                         linear_path_l2=1e-4, **options) -> NigTimeNetV2:
    """Same keyword names and defaults as upstream build_nig_timenet_v2 (``n_classes`` only matters for the
    multiclass head, P17)."""
    options = {"encoder": encoder, "input_skip": input_skip, **options}
    for opt, value in options.items():
        if opt not in _NOT_YET:
            raise TypeError(f"build_nig_timenet_v2() got an unexpected keyword argument {opt!r}")
        supported, step = _NOT_YET[opt]
        if value != supported:
            raise NotImplementedError(f"{opt}={value!r} is not ported yet (only {supported!r}); added in {step}")
    if isinstance(seq_len, dict) or isinstance(n_features, dict):
        raise NotImplementedError("multi-timeframe seq_len/n_features dicts are not ported yet (P34)")
    return NigTimeNetV2(
        int(seq_len), int(n_features), d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        num_kv_heads=num_kv_heads, patch_len=patch_len, stride=stride, kernel_sizes=tuple(kernel_sizes),
        max_rel_pos=max_rel_pos, dropout=dropout, attn_dropout=attn_dropout, causal=causal, window=window,
        price_targets=price_targets, head_types=head_types or DEFAULT_HEAD_TYPES, enforce_order=enforce_order,
        head_hidden=head_hidden, nu_min=nu_min, alpha_min=alpha_min, beta_min=beta_min, unc_max=unc_max,
        class_head_hidden=class_head_hidden, norm_eps=norm_eps, linear_path_l2=linear_path_l2)
