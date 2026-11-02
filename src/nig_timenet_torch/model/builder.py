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
    BatchNormChannelsLast,
    CausalMultiScaleDecomp,
    GaussianNoise,
    InstanceNorm,
    KerasGRU,
    LastToken,
    PatchEmbedding,
    RMSNorm,
    SpatialDropout1D,
    SymLog,
    TransformerBlock,
    keras_conv1d_weight_to_torch,
    keras_dense_weight_to_torch,
)
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans
from nig_timenet_torch.model.parts import INPUT_SKIP_VARIANTS, build_input_skip

__all__ = ["DEFAULT_HEAD_TYPES", "NigTimeNetV2", "build_nig_timenet_v2", "torch_param_name"]

DEFAULT_HEAD_TYPES = {"high": ["nig_regression"], "low": ["nig_regression"], "close": ["nig_regression"]}
HEAD_KINDS = ("nig_regression", "binary_classification")

# option -> (value this port implements, step that adds the rest)
_NOT_YET = {
    "use_instance_norm": (True, "P34"), "use_decomposition": (True, "P34"), "use_linear_path": (True, "P34"),
    "level_film": (False, "P34"), "architecture": ("fused", "P34"), "fusion": ("concat", "P34"), "n_coins": (0, "P34"),
}
ENCODERS = ("transformer", "gru")  # upstream also has "tcn": not in BUILD_PLAN.md, raises NotImplementedError
LEVEL_NORMS = (None, "batch")


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
                 norm_eps, linear_path_l2, encoder="transformer", input_clip=None, stats_mode="full",
                 input_noise_std=0.0, feature_dropout=0.0, level_passthrough=False, level_norm=None,
                 input_skip="none"):
        super().__init__()
        self.price_targets = tuple(price_targets)
        self.head_types = {t: list(head_types.get(t, [])) for t in self.price_targets}
        for t, kinds in self.head_types.items():
            bad = [k for k in kinds if k not in HEAD_KINDS]
            if bad:
                raise NotImplementedError(f"head types {bad} for {t!r} are not ported yet (P17 HEAD_REGISTRY)")
        self.linear_path_l2 = linear_path_l2  # kernel L2 of linear_path, see regularization_loss()
        self.num_layers = num_layers

        self.encoder = encoder
        self.use_stats = stats_mode != "none"
        self.instance_norm = InstanceNorm(n_features, eps=norm_eps, clip=input_clip, stats_mode=stats_mode)
        if feature_dropout > 0:
            self.feature_dropout = SpatialDropout1D(feature_dropout)
        if input_noise_std > 0:
            self.input_noise = GaussianNoise(input_noise_std)
        self.decomp = CausalMultiScaleDecomp(n_features, kernel_sizes)
        tok_width = 2 * n_features
        self.level_passthrough = level_passthrough
        if level_passthrough:  # anti-memorization: SymLog(raw level) re-enters the token path (P33b)
            self.level_symlog = SymLog()
            if level_norm == "batch":
                self.level_bn = BatchNormChannelsLast(n_features)
            if feature_dropout > 0:
                self.level_feature_dropout = SpatialDropout1D(feature_dropout)
            if input_noise_std > 0:
                self.level_noise = GaussianNoise(input_noise_std)
            tok_width += n_features
        if self.use_stats and level_norm == "batch":
            self.stats_bn = BatchNormChannelsLast(2 * n_features)
        self.input_skip = input_skip
        seq_width = d_model  # width of the encoder output sequence
        if encoder == "transformer":
            if input_skip == "no_proj":
                seq_width = tok_width
            else:
                self.patch_embed = PatchEmbedding(tok_width, d_model, patch_len, stride, seq_len=seq_len)
            if input_skip != "none":  # "none" adds no module: P15 parameter names stay identical
                self.skip = build_input_skip(input_skip, d_model=d_model, tok_width=tok_width, num_layers=num_layers,
                                             patch_len=patch_len, stride=stride)
        else:
            self.step_proj = nn.Linear(tok_width, d_model)
        for i in range(num_layers):
            if encoder == "transformer":
                setattr(self, f"block_{i + 1}", TransformerBlock(seq_width, num_heads, num_kv_heads, max_rel_pos,
                                                                 causal, window, dropout, attn_dropout))
            else:  # RMSNorm -> GRU -> Dropout, residual add (upstream encoder == "gru")
                setattr(self, f"gru_norm_{i + 1}", RMSNorm(d_model))
                setattr(self, f"gru_{i + 1}", KerasGRU(d_model, d_model))
                setattr(self, f"gru_drop_{i + 1}", nn.Dropout(dropout))
        self.final_norm = RMSNorm(seq_width)
        self.last_token = LastToken()
        self.attn_pool = AttentionPool(seq_width)
        if self.use_stats:
            self.stats_proj = nn.Linear(2 * n_features, d_model)
        extra = self.skip.readout_extra_dim if hasattr(self, "skip") else 0
        self.readout_fc = nn.Linear(2 * seq_width + (d_model if self.use_stats else 0) + extra, d_model)
        lin_width = n_features * (2 if level_passthrough else 1)
        self.linear_path = nn.Linear(seq_len * lin_width, d_model)
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
        for name in ("feature_dropout", "input_noise"):
            if hasattr(self, name):
                z = getattr(self, name)(z)
        trend, seasonal = self.decomp(z)
        tok, lvl = torch.cat([trend, seasonal], dim=-1), None
        if self.level_passthrough:
            lvl = self.level_symlog(x)
            for name in ("level_bn", "level_feature_dropout", "level_noise"):
                if hasattr(self, name):
                    lvl = getattr(self, name)(lvl)
            tok = torch.cat([tok, lvl], dim=-1)
        if stats is not None and hasattr(self, "stats_bn"):
            stats = self.stats_bn(stats)
        if self.encoder == "transformer":
            h = self.encode_tokens(tok)
        else:
            h = self.step_proj(tok)
            for i in range(self.num_layers):
                y = getattr(self, f"gru_{i + 1}")(getattr(self, f"gru_norm_{i + 1}")(h))
                h = h + getattr(self, f"gru_drop_{i + 1}")(y)
        h = self.final_norm(h)
        parts = [self.last_token(h), self.attn_pool(h)]
        if stats is not None:
            parts.append(F.gelu(self.stats_proj(stats)))
        if hasattr(self, "skip") and self.skip.readout_extra_dim:
            parts.append(self.skip.readout_extra(tok))
        lin_in = z if lvl is None else torch.cat([z, lvl], dim=-1)
        h = F.gelu(self.readout_fc(torch.cat(parts, dim=-1))) + self.linear_path(lin_in.flatten(1))
        return self.trunk_drop(self.trunk_norm(h))

    def encode_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        """Transformer encoder on the normalised token input (B, T, W) -> output sequence (the input_skip seam,
        P34). Per-position: with causal=True, position j depends on no bar after the end of its patch (or bar)."""
        blocks = [getattr(self, f"block_{i + 1}") for i in range(self.num_layers)]
        if hasattr(self, "skip"):
            return self.skip.encode(tok, getattr(self, "patch_embed", None), blocks)
        h = self.patch_embed(tok)
        for block in blocks:
            h = block(h)
        return h

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

    def regularization_loss(self) -> torch.Tensor:
        """Sum of the Keras kernel_regularizer terms (l2 * sum(w^2), no 0.5): every NIGHead.fc (l2=1e-5) and
        linear_path (linear_path_l2). The trainer adds it to total and raw loss (TF: model.losses)."""
        reg = self.linear_path_l2 * self.linear_path.weight.square().sum()
        for t in self.nig_targets:
            reg = reg + getattr(self, f"nig_{t}").regularization_loss()
        return reg

    @staticmethod
    def _oracle_targets(path: str, value: torch.Tensor, modules: dict) -> dict[str, torch.Tensor]:
        """attr_path -> {torch parameter/buffer name: value}; a module with ``keras_weight_map`` (KerasGRU, whose
        attr paths look like 'gru_1.cell.kernel') converts its own Keras weights."""
        parts = path.split(".")
        for i in range(len(parts) - 1, 0, -1):
            mod = modules.get(".".join(parts[:i]))
            if mod is not None and hasattr(mod, "keras_weight_map"):
                prefix = ".".join(parts[:i])
                return {f"{prefix}.{k}": v for k, v in mod.keras_weight_map(parts[-1], value).items()}
        return {torch_param_name(path): keras_to_torch_value(path, value)}

    @torch.no_grad()
    def load_oracle_weights(self, weights: dict[str, torch.Tensor], attr_paths: dict[str, str] | None = None) -> None:
        """Assign every parameter from ``weights`` keyed by attr_path (or by weights.json name, translated through
        ``attr_paths`` name -> attr_path). 'trainer/...' entries are skipped. Raises on any unmatched parameter,
        unknown key, or shape mismatch — no silent partial load."""
        attr_paths = attr_paths or {}
        params = {**dict(self.named_parameters()), **dict(self.named_buffers())}  # BN moving stats are buffers
        modules = dict(self.named_modules())
        assigned = set()
        for key, value in weights.items():
            path = attr_paths.get(key, key)
            if path.startswith("trainer/"):
                continue
            targets = self._oracle_targets(path, value, modules)
            for name, v in targets.items():
                if name not in params:
                    raise KeyError(f"weight {key!r} (attr_path {path!r}) has no torch parameter {name!r}")
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
                         linear_path_l2=1e-4, input_clip=None, stats_mode="full", input_noise_std=0.0,
                         feature_dropout=0.0, level_passthrough=False, level_norm=None, **options) -> NigTimeNetV2:
    """Same keyword names and defaults as upstream build_nig_timenet_v2 (``n_classes`` only matters for the
    multiclass head, P17). The anti-memorization options (encoder="gru", stats_mode, input_clip, input_noise_std,
    feature_dropout, level_passthrough, level_norm; P33b) are opt-in: their defaults reproduce the P15 graph."""
    if encoder not in ENCODERS:
        if encoder == "tcn":
            raise NotImplementedError("encoder='tcn' is not ported (not in BUILD_PLAN.md)")
        raise ValueError(f"encoder must be one of {ENCODERS + ('tcn',)}, not {encoder!r}")
    if input_skip not in INPUT_SKIP_VARIANTS:
        INPUT_SKIP_VARIANTS.entry(input_skip)  # raises UnknownNameError with the known names + did-you-mean
    if input_skip != "none" and encoder != "transformer":
        raise NotImplementedError(f"input_skip={input_skip!r} is defined on the transformer encoder only (P34)")
    if level_norm not in LEVEL_NORMS:
        raise ValueError(f"level_norm must be one of {LEVEL_NORMS}, not {level_norm!r}")
    options = dict(options)
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
        class_head_hidden=class_head_hidden, norm_eps=norm_eps, linear_path_l2=linear_path_l2, encoder=encoder,
        input_clip=input_clip, stats_mode=stats_mode, input_noise_std=input_noise_std, feature_dropout=feature_dropout,
        level_passthrough=level_passthrough, level_norm=level_norm, input_skip=input_skip)
