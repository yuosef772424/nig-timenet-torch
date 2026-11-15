"""build_nig_timenet_v2 / NigTimeNetV2: the full model, wired as crypto-signal-prediction model/builder.py
@ aeafc564 builds: the "fused" model with a transformer (P15), GRU (P33b) or TCN (P40) encoder, one timeframe or one
encoder branch per timeframe (P41), nig_regression / binary / multiclass heads (P17, P42).

Module attribute names equal the Keras layer names, so weights.json ``attr_path`` strings map 1:1 onto torch
parameter names (see ``torch_param_name``). Options of the upstream builder that this port does not implement yet
are accepted and validated: any non-default value raises NotImplementedError naming the step that adds it.

PURPOSE:  NigTimeNetV2 (the fused model: input path, one encoder branch per timeframe, trunk, heads) and
          build_nig_timenet_v2; encoders come from model.encoders.ENCODERS, head kinds from HEAD_REGISTRY.
TAGS:     builder, nig timenet v2, model assembly, encoder dispatch, head dispatch, oracle weights, state dict
PITFALLS: THE default config is the registry's (architectures.NIG_TIMENET_V2_DEFAULT_CONFIG: d_model 16, 2 layers);
          this signature mirrors the TF builder (d_model 128, 4 layers, NIG heads only) and must not be edited to
          "fix" that (HF-08). Attribute names are Keras layer names = checkpoint keys: never rename.
          ``activation_points`` only lists module names for gate A7 (no behaviour).
"""

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model import label_scale
from nig_timenet_torch.model.encoders import ENCODERS, CausalConv1D, EncoderDims, tcn_n_blocks
from nig_timenet_torch.model.heads import (
    HEAD_REGISTRY,
    class_head_kind,
    class_outputs,
    head_spec,
    head_width,
    kinds_with_task,
    nig_outputs,
)
from nig_timenet_torch.model.layers import (
    STATS_TRANSFORMS,
    AttentionPool,
    BatchNormChannelsLast,
    CausalMultiScaleDecomp,
    GaussianNoise,
    InstanceNorm,
    LastToken,
    RMSNorm,
    SpatialDropout1D,
    SymLog,
    keras_conv1d_weight_to_torch,
    keras_dense_weight_to_torch,
)
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans, grad_scale
from nig_timenet_torch.model.parts import INPUT_SKIP_VARIANTS, skip_variant

__all__ = ["DEFAULT_HEAD_TYPES", "NigTimeNetV2", "build_nig_timenet_v2", "torch_param_name", "CausalConv1D",
           "tcn_n_blocks"]

DEFAULT_HEAD_TYPES = {t: ["nig_regression"] for t in TARGET_COLUMNS}
#: InstanceNorm stats_mode values that return no per-window statistics
NO_STATS_MODES = tuple(m for m, f in STATS_TRANSFORMS.items() if f is None)

# option -> (value this port implements, step that adds the rest)
_NOT_YET = {
    "use_instance_norm": (True, "P34"), "use_decomposition": (True, "P34"), "use_linear_path": (True, "P34"),
    "level_film": (False, "P34"), "architecture": ("fused", "P34"), "fusion": ("concat", "P34"), "n_coins": (0, "P34"),
}
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
    """seq_len/n_features: ints (one timeframe, the original graph and names) or dicts {timeframe: value}
    (``n_features`` may stay an int for every timeframe) = one encoder branch per timeframe with separate weights,
    every module name suffixed ``_{timeframe}`` (Keras names), branch embeddings concatenated before trunk_norm.
    A dict with one timeframe is the single-timeframe model (upstream rule)."""

    def __init__(self, seq_len, n_features, *, d_model, num_layers, num_heads, num_kv_heads, patch_len,
                 stride, kernel_sizes, max_rel_pos, dropout, attn_dropout, causal, window, price_targets,
                 head_types, enforce_order, head_hidden, nu_min, alpha_min, beta_min, unc_max, class_head_hidden,
                 norm_eps, linear_path_l2, encoder="transformer", input_clip=None, stats_mode="full",
                 input_noise_std=0.0, feature_dropout=0.0, level_passthrough=False, level_norm=None,
                 input_skip="none", n_classes=3, order_detach_close=False, nig_scale=None,
                 bn_warmup=False, readout_dropout=None, nig_trunk_grad=1.0, nig_scale_fit="fixed"):
        super().__init__()
        self.nig_trunk_grad = float(nig_trunk_grad)
        self.label_scale_name, fit_s = nig_scale_fit, label_scale.fitted(nig_scale_fit)   # FX2 (B-052)
        self.price_targets = tuple(price_targets)
        self.head_types = {t: list(head_types.get(t, [])) for t in self.price_targets}
        for t, kinds in self.head_types.items():
            bad = [k for k in kinds if k not in HEAD_REGISTRY]
            if bad:
                raise NotImplementedError(f"head types {bad} for {t!r} are not ported (known: {HEAD_REGISTRY.names()})")
        self.class_kind = {t: k for t in self.price_targets if (k := class_head_kind(self.head_types[t], t))}
        self.linear_path_l2 = linear_path_l2  # kernel L2 of every linear_path, see regularization_loss()
        self.num_layers = num_layers
        self.encoder = encoder
        self.use_stats = stats_mode not in NO_STATS_MODES
        self.level_passthrough = level_passthrough
        self.input_skip = input_skip

        if isinstance(seq_len, dict):
            tfs = [str(k) for k in seq_len]
            if isinstance(n_features, dict) and set(map(str, n_features)) != set(tfs):
                raise ValueError(f"seq_len and n_features have different timeframes: {tfs} != {list(n_features)}")
            feats = {str(k): int(v) for k, v in n_features.items()} if isinstance(n_features, dict) else None
            branches = [(f"_{tf}", int(seq_len[tf]), feats[tf] if feats else int(n_features)) for tf in tfs]
            if len(branches) == 1:
                branches = [("", branches[0][1], branches[0][2])]
        else:
            tfs = None
            branches = [("", int(seq_len), int(n_features))]
        self.timeframes = tuple(tfs) if tfs and len(tfs) > 1 else None
        if self.timeframes and skip_variant(input_skip).adds_modules:
            raise NotImplementedError("input_skip is defined on the single-timeframe transformer only (P34)")
        self.branch_suffixes = tuple(b[0] for b in branches)
        self.branch_blocks = {}
        dims = EncoderDims(d_model, num_layers, num_heads, num_kv_heads, patch_len, stride, max_rel_pos, dropout,
                           attn_dropout, causal, window)
        for sfx, T, nf in branches:
            self._build_branch(sfx, T, nf, dims, kernel_sizes=kernel_sizes, norm_eps=norm_eps, input_clip=input_clip,
                               stats_mode=stats_mode, input_noise_std=input_noise_std,
                               feature_dropout=feature_dropout, level_norm=level_norm, bn_warmup=bn_warmup)
        h_dim = d_model * len(branches)
        self.trunk_norm = RMSNorm(h_dim)
        readout_p = dropout if readout_dropout is None else readout_dropout  # FX1: trunk_drop + NIG head dropout
        self.trunk_drop = nn.Dropout(readout_p)
        self.trunk_dim = h_dim

        self.nig_targets = [t for t in self.price_targets
                            if kinds_with_task(self.head_types[t], "evidential")]
        self.head_specs = {t: head_spec(k, t) for t in self.nig_targets
                           for k in kinds_with_task(self.head_types[t], "evidential")}
        for t in self.nig_targets:
            setattr(self, f"nig_{t}", NIGHead(h_dim, head_hidden, readout_p, nu_min, alpha_min, beta_min,
                                              scale=nig_scale, fitted_scale=fit_s))
            setattr(self, f"unc_{t}", NIGUncertainty(unc_max))
            setattr(self, f"conf_{t}", ConfidenceHead(h_dim))
        self.apply_order = enforce_order and set(TARGET_COLUMNS) <= set(self.nig_targets)
        if self.apply_order:
            self.ordered_means = OrderedMeans(nig_scale, order_detach_close, fitted_scale=fit_s)
        self.class_targets = list(self.class_kind)
        self.class_specs = {t: head_spec(kind, t) for t, kind in self.class_kind.items()}
        for t, kind in self.class_kind.items():
            k = head_width(kind, n_classes[t] if isinstance(n_classes, dict) else n_classes)
            setattr(self, f"class_fc_{t}", nn.Linear(h_dim, class_head_hidden))
            setattr(self, f"class_drop_{t}", nn.Dropout(dropout))
            setattr(self, self.class_specs[t].output_keys["logits"], nn.Linear(class_head_hidden, k))

    def _build_branch(self, sfx, T, nf, dims, *, kernel_sizes, norm_eps, input_clip, stats_mode, input_noise_std,
                      feature_dropout, level_norm, bn_warmup):
        def add(name, module):
            setattr(self, name + sfx, module)

        add("instance_norm", InstanceNorm(nf, eps=norm_eps, clip=input_clip, stats_mode=stats_mode))
        if feature_dropout > 0:
            add("feature_dropout", SpatialDropout1D(feature_dropout))
        if input_noise_std > 0:
            add("input_noise", GaussianNoise(input_noise_std))
        add("decomp", CausalMultiScaleDecomp(nf, kernel_sizes))
        tok_width = 2 * nf
        if self.level_passthrough:  # anti-memorization: SymLog(raw level) re-enters the token path (P33b)
            add("level_symlog", SymLog())
            if level_norm == "batch":
                add("level_bn", BatchNormChannelsLast(nf, warmup=bn_warmup))
            if feature_dropout > 0:
                add("level_feature_dropout", SpatialDropout1D(feature_dropout))
            if input_noise_std > 0:
                add("level_noise", GaussianNoise(input_noise_std))
            tok_width += nf
        if self.use_stats and level_norm == "batch":
            add("stats_bn", BatchNormChannelsLast(2 * nf, warmup=bn_warmup))
        d_model = dims.d_model
        seq_width = ENCODERS.get(self.encoder).build(self, add, sfx, seq_len=T, tok_width=tok_width, dims=dims)
        add("final_norm", RMSNorm(seq_width))
        add("last_token", LastToken())
        add("attn_pool", AttentionPool(seq_width))
        if self.use_stats:
            add("stats_proj", nn.Linear(2 * nf, d_model))
        extra = self.skip.readout_extra_dim if hasattr(self, "skip") else 0
        add("readout_fc", nn.Linear(2 * seq_width + (d_model if self.use_stats else 0) + extra, d_model))
        lin_width = nf * (2 if self.level_passthrough else 1)
        add("linear_path", nn.Linear(T * lin_width, d_model))

    def _m(self, name, sfx=""):
        return getattr(self, name + sfx, None)

    def branch(self, x: torch.Tensor, sfx: str = "") -> torch.Tensor:
        """One timeframe's encoder up to its pooled embedding h (after readout_fc + linear_path)."""
        z, stats = self._m("instance_norm", sfx)(x)
        for name in ("feature_dropout", "input_noise"):
            if self._m(name, sfx) is not None:
                z = self._m(name, sfx)(z)
        trend, seasonal = self._m("decomp", sfx)(z)
        tok, lvl = torch.cat([trend, seasonal], dim=-1), None
        if self.level_passthrough:
            lvl = self._m("level_symlog", sfx)(x)
            for name in ("level_bn", "level_feature_dropout", "level_noise"):
                if self._m(name, sfx) is not None:
                    lvl = self._m(name, sfx)(lvl)
            tok = torch.cat([tok, lvl], dim=-1)
        if stats is not None and self._m("stats_bn", sfx) is not None:
            stats = self._m("stats_bn", sfx)(stats)
        h = self.encode_tokens(tok, sfx)
        h = self._m("final_norm", sfx)(h)
        parts = [self._m("last_token", sfx)(h), self._m("attn_pool", sfx)(h)]
        if stats is not None:
            parts.append(F.gelu(self._m("stats_proj", sfx)(stats)))
        if hasattr(self, "skip") and self.skip.readout_extra_dim:
            parts.append(self.skip.readout_extra(tok))
        lin_in = z if lvl is None else torch.cat([z, lvl], dim=-1)
        return F.gelu(self._m("readout_fc", sfx)(torch.cat(parts, dim=-1))) + self._m("linear_path", sfx)(
            lin_in.flatten(1))

    def _branch_inputs(self, x) -> list[torch.Tensor]:
        if self.timeframes is None:
            if isinstance(x, dict):
                if len(x) != 1:
                    raise ValueError(f"single-timeframe model got inputs {list(x)}")
                x = next(iter(x.values()))
            elif isinstance(x, (list, tuple)):
                (x,) = x
            return [x]
        if isinstance(x, dict):
            missing = [tf for tf in self.timeframes if tf not in x]
            if missing:
                raise KeyError(f"missing timeframe inputs {missing}; expected {list(self.timeframes)}")
            return [x[tf] for tf in self.timeframes]
        if not isinstance(x, (list, tuple)) or len(x) != len(self.timeframes):
            raise ValueError(f"multi-timeframe model expects a dict or a sequence in order {list(self.timeframes)}")
        return list(x)

    def trunk(self, x) -> torch.Tensor:
        """x: (B, T, F) tensor, or for multi-timeframe a dict {timeframe: (B, T_tf, F_tf)} / sequence in the
        timeframe order. Returns trunk_drop's output (B, d_model * n_timeframes) — the panel encoder output."""
        xs = [xi.to(torch.float32) for xi in self._branch_inputs(x)]
        hs = [self.branch(xi, sfx) for xi, sfx in zip(xs, self.branch_suffixes, strict=True)]
        h = hs[0] if len(hs) == 1 else torch.cat(hs, dim=-1)  # branch_concat
        return self.trunk_drop(self.trunk_norm(h))

    def encode_tokens(self, tok: torch.Tensor, sfx: str = "") -> torch.Tensor:
        """The registered encoder on the normalised token input (B, T, W) -> output sequence (for the transformer
        this is the input_skip seam, P34)."""
        return ENCODERS.get(self.encoder).encode(self, tok, sfx)

    def forward(self, x) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        h_nig = grad_scale(h, self.nig_trunk_grad)   # FX2: 1.0 = h itself
        raw = {t: getattr(self, f"nig_{t}")(h_nig) for t in self.nig_targets}
        if self.apply_order:
            mu_h, mu_c, mu_l = self.ordered_means(raw["close"][0], raw["high"][0], raw["low"][0])
            mus = {"high": mu_h, "close": mu_c, "low": mu_l}
        else:
            mus = {t: raw[t][0] for t in self.nig_targets}
        out: dict[str, torch.Tensor] = {}
        for t in self.nig_targets:
            _, nu, alpha, beta = raw[t]
            epi, ale = getattr(self, f"unc_{t}")(nu, alpha, beta)
            out.update(nig_outputs(self.head_specs[t], mus[t], nu, alpha, beta, epi, ale,
                                   getattr(self, f"conf_{t}")(h, nu, alpha, beta)))
        for t, kind in self.class_kind.items():
            spec = self.class_specs[t]
            out_layer = getattr(self, spec.output_keys["logits"])
            probs = HEAD_REGISTRY.get(kind)(h, getattr(self, f"class_fc_{t}"), out_layer,
                                            getattr(self, f"class_drop_{t}"))
            out.update(class_outputs(spec, probs))
        return out

    default_probe_points = ("trunk_norm",)  # layer-probe points of the health report (module names)
    balance_point = "trunk_drop"            # the shared representation h a loss balancer measures (FX2)

    @property
    def activation_points(self) -> tuple[str, ...]:
        """Linear layers whose output feeds a ReLU-family activation (gate A7 hooks them; activations are functional):
        stats projection and readout per branch, each NIG head's hidden layer, confidence hidden layer, class hidden
        layer."""
        names = dict(self.named_modules())
        branch = [n + sfx for sfx in self.branch_suffixes for n in ("stats_proj", "readout_fc")]
        heads = [f"nig_{t}.fc" for t in self.nig_targets] + [f"conf_{t}.d1" for t in self.nig_targets]
        return tuple(n for n in branch + heads + [f"class_fc_{t}" for t in self.class_targets] if n in names)

    def head_modules(self) -> dict[str, NIGHead]:
        """{target: NIGHead} of every nig_regression head (floors and diagnostics read them, never by name)."""
        return {t: getattr(self, f"nig_{t}") for t in self.nig_targets}

    def regularization_loss(self) -> torch.Tensor:
        """Sum of the Keras kernel_regularizer terms (l2 * sum(w^2), no 0.5): every NIGHead.fc (l2=1e-5) and
        linear_path (linear_path_l2). The trainer adds it to total and raw loss (TF: model.losses)."""
        reg = sum(self.linear_path_l2 * self._m("linear_path", sfx).weight.square().sum()
                  for sfx in self.branch_suffixes)
        for head in self.head_modules().values():
            reg = reg + head.regularization_loss()
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
                         attn_dropout=0.0, causal=False, window=None, price_targets=TARGET_COLUMNS,
                         head_types: dict[str, list[str]] | None = None, enforce_order=True, head_hidden=128,
                         nu_min=0.1, alpha_min=2.0, beta_min=0.01, unc_max=20.0, class_head_hidden=64,
                         encoder: str = "transformer", input_skip: str = "none", n_classes=3, norm_eps=1e-4,
                         linear_path_l2=1e-4, input_clip=None, stats_mode="full", input_noise_std=0.0,
                         feature_dropout=0.0, level_passthrough=False, level_norm=None, order_detach_close=False,
                         nig_scale=None, bn_warmup=False, readout_dropout=None, nig_trunk_grad=1.0,
                         nig_scale_fit="fixed", **options) -> NigTimeNetV2:
    """Same keyword names and defaults as upstream build_nig_timenet_v2 (``n_classes`` only matters for the
    multiclass head, P17). The anti-memorization options (encoder="gru", stats_mode, input_clip, input_noise_std,
    feature_dropout, level_passthrough, level_norm; P33b) and the FX1 switches (nig_scale, order_detach_close: see
    nig_layers; bn_warmup: layers.BatchNormChannelsLast; readout_dropout: the rate of trunk_drop and of the NIG
    heads' dropout, None = ``dropout``; FX2: nig_trunk_grad (nig_layers.grad_scale), nig_scale_fit (model.label_scale))
    are opt-in: their defaults
    reproduce the P15 graph."""
    ENCODERS.entry(encoder)  # raises UnknownNameError (a ValueError) with the known names + did-you-mean
    INPUT_SKIP_VARIANTS.entry(input_skip)  # likewise
    if skip_variant(input_skip).adds_modules and not ENCODERS.entry(encoder).meta["supports_input_skip"]:
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
    if isinstance(n_features, dict) and not isinstance(seq_len, dict):
        raise ValueError("n_features is a dict but seq_len is not: pass both as {timeframe: value}")
    return NigTimeNetV2(
        seq_len if isinstance(seq_len, dict) else int(seq_len),
        n_features if isinstance(n_features, dict) else int(n_features), d_model=d_model, num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads, patch_len=patch_len, stride=stride, kernel_sizes=tuple(kernel_sizes),
        max_rel_pos=max_rel_pos, dropout=dropout, attn_dropout=attn_dropout, causal=causal, window=window,
        price_targets=price_targets, head_types=head_types or DEFAULT_HEAD_TYPES, enforce_order=enforce_order,
        head_hidden=head_hidden, nu_min=nu_min, alpha_min=alpha_min, beta_min=beta_min, unc_max=unc_max,
        class_head_hidden=class_head_hidden, norm_eps=norm_eps, linear_path_l2=linear_path_l2, encoder=encoder,
        input_clip=input_clip, stats_mode=stats_mode, input_noise_std=input_noise_std, feature_dropout=feature_dropout,
        level_passthrough=level_passthrough, level_norm=level_norm, input_skip=input_skip, n_classes=n_classes,
        order_detach_close=order_detach_close, nig_scale=nig_scale,
        bn_warmup=bn_warmup, readout_dropout=readout_dropout, nig_trunk_grad=nig_trunk_grad,
        nig_scale_fit=nig_scale_fit)
