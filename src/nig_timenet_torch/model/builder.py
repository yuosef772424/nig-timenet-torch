"""build_nig_timenet_v2 / NigTimeNetV2: the full model, wired as crypto-signal-prediction model/builder.py
@ aeafc564 builds: the "fused" model with a transformer (P15), GRU (P33b) or TCN (P40) encoder, one timeframe or one
encoder branch per timeframe (P41), nig_regression / binary / multiclass heads (P17, P42).

Module attribute names equal the Keras layer names, so weights.json ``attr_path`` strings map 1:1 onto torch
parameter names (see ``torch_param_name``). Options of the upstream builder that this port does not implement yet
are accepted and validated: any non-default value raises NotImplementedError (not implemented, no step planned).

PURPOSE:  NigTimeNetV2 (the fused model: input path, one encoder branch per timeframe, trunk, heads) and
          build_nig_timenet_v2; encoders come from model.encoders.ENCODERS, head kinds from HEAD_REGISTRY.
TAGS:     builder, nig timenet v2, model assembly, encoder dispatch, head dispatch, oracle weights, state dict
PITFALLS: THE default config is the registry's (architectures.NIG_TIMENET_V2_DEFAULT_CONFIG: d_model 16, 2 layers);
          the signature keeps the TF builder's core arguments and defaults (d_model 128, 4 layers; the options
          documented in its docstring beyond the TF ones are port-only) and must not be edited to "fix" that (HF-08).
          Attribute names are Keras layer names = checkpoint keys: never rename.
          ``activation_points`` only lists module names for gate A7 (no behaviour).
"""

from dataclasses import dataclass, fields, replace

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model import label_scale
from nig_timenet_torch.model.attn_heads import check_nig_head, nig_head_builder, reads_sequence
from nig_timenet_torch.model.cross_coin import CROSS_INPUTS, CrossCoin, center_groups, split_group
from nig_timenet_torch.model.encoders import ENCODERS, CausalConv1D, EncoderDims, tcn_n_blocks
from nig_timenet_torch.model.har_range import HarRange
from nig_timenet_torch.model.heads import (
    AUX_TASK_TYPES,
    HEAD_REGISTRY,
    aux_outputs,
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
    TransformerBlock,
)
from nig_timenet_torch.model.nig_layers import ConfidenceHead, NIGHead, NIGUncertainty, OrderedMeans, grad_scale
from nig_timenet_torch.model.oracle_weights import OracleWeights, keras_to_torch_value, torch_param_name  # noqa: F401
from nig_timenet_torch.model.parts import INPUT_SKIP_VARIANTS, skip_variant
from nig_timenet_torch.model.scalers import INPUT_SCALERS
from nig_timenet_torch.model.target_norm import TARGET_NORMS

__all__ = ["DEFAULT_HEAD_TYPES", "BuilderOptions", "NigTimeNetV2", "build_nig_timenet_v2", "torch_param_name",
           "CausalConv1D", "tcn_n_blocks"]

DEFAULT_HEAD_TYPES = {t: ["nig_regression"] for t in TARGET_COLUMNS}
AUX_TARGET = "close"   # the target the auxiliary heads (aux_heads) are attached to: the two-sided one
#: InstanceNorm stats_mode values that return no per-window statistics
NO_STATS_MODES = tuple(m for m, f in STATS_TRANSFORMS.items() if f is None)

# upstream option -> the one value this port implements; any other value is not implemented (no step planned)
_UNIMPLEMENTED_OPTIONS = {
    "use_decomposition": True, "use_linear_path": True, "level_film": False, "architecture": "fused",
    "fusion": "concat", "n_coins": 0,
}
LEVEL_NORMS = (None, "batch")


@dataclass(frozen=True)
class BuilderOptions:
    """The model's keyword options, written ONCE: ``build_nig_timenet_v2(seq_len, n_features, **options)`` fills it
    (unknown names raise), ``NigTimeNetV2`` reads it. The core fields keep the TF builder's names and defaults
    (d_model 128, 4 layers; HF-08), the rest are the port's opt-ins (see ``build_nig_timenet_v2``)."""
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    patch_len: int = 4
    stride: int = 2
    kernel_sizes: tuple = (3, 5, 9, 17)
    max_rel_pos: int = 16
    dropout: float = 0.1
    attn_dropout: float = 0.0
    causal: bool = False
    window: int | None = None
    price_targets: tuple = tuple(TARGET_COLUMNS)
    head_types: dict | None = None
    enforce_order: bool = True
    head_hidden: int = 128
    nu_min: float = 0.1
    alpha_min: float = 2.0
    beta_min: float = 0.01
    unc_max: float = 20.0
    class_head_hidden: int = 64
    encoder: str = "transformer"
    input_skip: str = "none"
    n_classes: int | dict = 3
    norm_eps: float = 1e-4
    linear_path_l2: float = 1e-4
    input_clip: float | None = None
    stats_mode: str = "full"
    input_noise_std: float = 0.0
    feature_dropout: float = 0.0
    level_passthrough: bool = False
    level_norm: str | None = None
    order_detach_close: bool = False
    nig_scale: float | str | None = None
    bn_warmup: bool = False
    readout_dropout: float | None = None
    nig_trunk_grad: float = 1.0
    nig_scale_fit: str = "fixed"
    input_scaler: str = "none"
    nig_head: str = "dense"
    nig_head_dropout: float = 0.0
    ffn: str = "swiglu"
    ffn_dim: int | None = None
    cross_coin_layers: int = 0
    cross_coin_heads: int = 4
    cross_coin_peer_dropout: float = 0.0
    cross_coin_kind: str = "attention"
    cross_input: str = "none"
    cross_center: bool = False
    target_norm: str = "none"
    har_range: tuple | None = None
    use_instance_norm: bool = True
    order_in_label_units: bool = False
    aux_heads: tuple = ()
    aux_trunk_grad: float = 1.0


_OPTION_NAMES = frozenset(f.name for f in fields(BuilderOptions))


class NigTimeNetV2(OracleWeights, nn.Module):
    """seq_len/n_features: ints (one timeframe, the original graph and names) or dicts {timeframe: value}
    (``n_features`` may stay an int for every timeframe) = one encoder branch per timeframe with separate weights,
    every module name suffixed ``_{timeframe}`` (Keras names), branch embeddings concatenated before trunk_norm.
    A dict with one timeframe is the single-timeframe model (upstream rule)."""

    def __init__(self, seq_len, n_features, options: BuilderOptions):
        super().__init__()
        o = options
        self._set_options(o, seq_len, n_features)
        branches = self._set_branches(o, seq_len, n_features)
        h_dim = self._build_input_branches(o, branches)
        readout_p = self._build_trunk(o, h_dim)
        self._build_nig_heads(o, branches, h_dim, readout_p)
        self._build_cross_coin(o, h_dim)
        self._build_target_norm_and_order(o)
        self._build_class_heads(o, h_dim)
        self._build_aux_heads(o, h_dim)

    def _set_options(self, o: BuilderOptions, seq_len, n_features) -> None:
        """The plain attributes (and the one module, ``cross_input``, that must exist before the branches) the forward
        pass and the trainer read; validates the head kinds and cross_input."""
        self.use_instance_norm = bool(o.use_instance_norm)   # owner 2026-11-18: False = no per-window normalisation
        self.har_range = None if o.har_range is None else tuple(int(c) for c in o.har_range)   # H3
        self.cross_center = bool(o.cross_center)   # D-032: the close mean is taken relative to its timestamp
        self.nig_trunk_grad = float(o.nig_trunk_grad)
        self.label_scale_name, self._fit_s = o.nig_scale_fit, label_scale.fitted(o.nig_scale_fit)   # FX2 (B-052)
        if o.cross_input not in CROSS_INPUTS:
            raise ValueError(f"cross_input must be one of {tuple(CROSS_INPUTS)}, not {o.cross_input!r}")
        if CROSS_INPUTS[o.cross_input] is not None:
            if isinstance(seq_len, dict):
                raise NotImplementedError("cross_input needs one timeframe")
            self.cross_input = CROSS_INPUTS[o.cross_input]()
        self.n_group_features = None if isinstance(n_features, dict) else int(n_features)  # split_group reads it
        self.nig_head_kind = check_nig_head(o.nig_head)
        self.input_scaler_kind = o.input_scaler
        self._stream = None  # set by model.taps.stream_taps while recording (a plain attribute, not saved)
        self._stream_edit = {}  # {stream name: fn(tensor) -> tensor}, read only while recording
        self.price_targets = tuple(o.price_targets)
        self.head_types = {t: list(o.head_types.get(t, [])) for t in self.price_targets}
        for t, kinds in self.head_types.items():
            bad = [k for k in kinds if k not in HEAD_REGISTRY]
            if bad:
                raise NotImplementedError(f"head types {bad} for {t!r} are not ported (known: {HEAD_REGISTRY.names()})")
        self.class_kind = {t: k for t in self.price_targets if (k := class_head_kind(self.head_types[t], t))}
        self.linear_path_l2 = o.linear_path_l2  # kernel L2 of every linear_path, see regularization_loss()
        self.num_layers, self.encoder = o.num_layers, o.encoder
        self.use_stats = o.stats_mode not in NO_STATS_MODES
        self.level_passthrough = o.level_passthrough
        self.input_skip = o.input_skip

    def _set_branches(self, o: BuilderOptions, seq_len, n_features) -> list[tuple[str, int, int]]:
        """[(module-name suffix, seq_len, n_features)] per timeframe; sets ``timeframes`` / ``branch_suffixes``."""
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
        if self.timeframes and skip_variant(o.input_skip).adds_modules:
            raise NotImplementedError("input_skip on several timeframes is not implemented (single-timeframe only)")
        self.branch_suffixes = tuple(b[0] for b in branches)
        return branches

    def _build_input_branches(self, o: BuilderOptions, branches) -> int:
        """One encoder branch per timeframe; returns the width of the concatenated embeddings."""
        self.branch_blocks = {}
        dims = EncoderDims(o.d_model, o.num_layers, o.num_heads, o.num_kv_heads, o.patch_len, o.stride, o.max_rel_pos,
                           o.dropout, o.attn_dropout, o.causal, o.window, o.ffn, o.ffn_dim)
        for sfx, T, nf in branches:
            self._build_branch(sfx, T, nf, dims, kernel_sizes=o.kernel_sizes, norm_eps=o.norm_eps,
                               input_clip=o.input_clip, stats_mode=o.stats_mode, input_noise_std=o.input_noise_std,
                               feature_dropout=o.feature_dropout, level_norm=o.level_norm, bn_warmup=o.bn_warmup)
        return o.d_model * len(branches)

    def _build_trunk(self, o: BuilderOptions, h_dim: int) -> float:
        """trunk_norm + trunk_drop; returns the readout dropout rate (FX1: trunk_drop + NIG head dropout)."""
        self.trunk_norm = RMSNorm(h_dim)
        readout_p = o.dropout if o.readout_dropout is None else o.readout_dropout
        self.trunk_drop = nn.Dropout(readout_p)
        self.trunk_dim = h_dim
        return readout_p

    def _build_nig_heads(self, o: BuilderOptions, branches, h_dim: int, readout_p: float) -> None:
        """Per NIG target: the head, its uncertainty map and its confidence head."""
        self.nig_targets = [t for t in self.price_targets if kinds_with_task(self.head_types[t], "evidential")]
        self.head_specs = {t: head_spec(k, t) for t in self.nig_targets
                           for k in kinds_with_task(self.head_types[t], "evidential")}
        head = nig_head_builder(self, o.nig_head, h_dim, o.d_model,
                                lambda: ENCODERS.get(o.encoder).n_tokens(branches[0][1], o.patch_len, o.stride),
                                dict(nu_min=o.nu_min, alpha_min=o.alpha_min, beta_min=o.beta_min, scale=o.nig_scale,
                                     fitted_scale=self._fit_s),
                                dense=dict(hidden=o.head_hidden, dropout=readout_p), attn_dropout=o.nig_head_dropout)
        for t in self.nig_targets:
            setattr(self, f"nig_{t}", head())
            setattr(self, f"unc_{t}", NIGUncertainty(o.unc_max))
            setattr(self, f"conf_{t}", ConfidenceHead(h_dim))

    def _build_cross_coin(self, o: BuilderOptions, h_dim: int) -> None:
        if o.cross_coin_layers:  # rows of one timestamp attend to each other between the trunk and the heads
            if self.timeframes or reads_sequence(o.nig_head):
                raise NotImplementedError("cross_coin_layers needs one timeframe and a head that reads the trunk")
            self.cross_coin = CrossCoin(h_dim, int(o.cross_coin_layers), o.cross_coin_heads, o.dropout,
                                        o.cross_coin_peer_dropout, o.cross_coin_kind)

    def _build_target_norm_and_order(self, o: BuilderOptions) -> None:
        self.target_norm_kind = o.target_norm  # owner 2026-11-17: per-head label standardisation (model.target_norm)
        if TARGET_NORMS.entry(o.target_norm).meta["fitted"]:
            self.target_norm = TARGET_NORMS.get(o.target_norm)(self.nig_targets)
        self.apply_order = o.enforce_order and set(TARGET_COLUMNS) <= set(self.nig_targets)
        # owner 2026-11-19 (B-084): with a fitted target normalisation, order the means on prices (after the inverse
        # map) instead of in the per-head normalised space, where the order has no meaning
        self.order_in_label_units = bool(o.order_in_label_units)
        if self.order_in_label_units and not (self.apply_order and hasattr(self, "target_norm")):
            raise ValueError("order_in_label_units needs the ordered means (enforce_order, the three price targets) "
                             "and a fitted target normalisation (target_norm, e.g. 'logret_std'): without them it "
                             "would change nothing")
        if self.apply_order:
            self.ordered_means = OrderedMeans(o.nig_scale, o.order_detach_close, fitted_scale=self._fit_s)

    def _build_class_heads(self, o: BuilderOptions, h_dim: int) -> None:
        self.class_targets = list(self.class_kind)
        self.class_specs = {t: head_spec(kind, t) for t, kind in self.class_kind.items()}
        for t, kind in self.class_kind.items():
            k = head_width(kind, o.n_classes[t] if isinstance(o.n_classes, dict) else o.n_classes)
            setattr(self, f"class_fc_{t}", nn.Linear(h_dim, o.class_head_hidden))
            setattr(self, f"class_drop_{t}", nn.Dropout(o.dropout))
            setattr(self, self.class_specs[t].output_keys["logits"], nn.Linear(o.class_head_hidden, k))

    def _build_aux_heads(self, o: BuilderOptions, h_dim: int) -> None:
        """owner 2026-11-19 (LS1): auxiliary heads on the close target (side payoffs, an upper quantile); they read h
        through grad_scale(h, aux_trunk_grad): 0 = a read-out of the trunk the other heads train (no feedback)."""
        self.aux_trunk_grad = float(o.aux_trunk_grad)
        self.aux_specs = {}
        for kind in o.aux_heads:
            if HEAD_REGISTRY.entry(kind).meta["task_type"] not in AUX_TASK_TYPES:
                raise ValueError(f"aux_heads takes the auxiliary head kinds (task types {AUX_TASK_TYPES}), "
                                 f"not {kind!r}")
            if AUX_TARGET not in self.nig_targets:
                raise ValueError(f"aux_heads need the {AUX_TARGET!r} NIG head (their label)")
            spec = head_spec(kind, AUX_TARGET)
            if spec.name in self.aux_specs:
                raise ValueError(f"aux head {kind!r} is named twice")
            self.aux_specs[spec.name] = spec
            setattr(self, f"aux_fc_{spec.name}", nn.Linear(h_dim, o.class_head_hidden))
            setattr(self, f"aux_drop_{spec.name}", nn.Dropout(o.dropout))
            out_layer = nn.Linear(o.class_head_hidden, head_width(kind, 1))
            nn.init.normal_(out_layer.weight, std=1e-3)   # start near 0: a payoff / quantile is ~1e-2, not ~1
            nn.init.zeros_(out_layer.bias)
            setattr(self, f"aux_out_{spec.name}", out_layer)

    def _build_branch(self, sfx, T, nf, dims, *, kernel_sizes, norm_eps, input_clip, stats_mode, input_noise_std,
                      feature_dropout, level_norm, bn_warmup):
        def add(name, module):
            setattr(self, name + sfx, module)

        if INPUT_SCALERS.entry(self.input_scaler_kind).meta["fitted"]:  # train-fitted standardisation (scalers.py)
            add("input_scaler", INPUT_SCALERS.get(self.input_scaler_kind)(nf))
        if hasattr(self, "cross_input"):  # the scaled input grows to [own, relative, relative z, market]
            nf = nf * self.cross_input.width_mult
        add("instance_norm", InstanceNorm(nf, eps=norm_eps, clip=input_clip, stats_mode=stats_mode,
                                          normalize=self.use_instance_norm))
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
        if self.har_range is not None:   # H3: the HAR range terms of the RAW input join the readout
            add("har_range", HarRange(self.har_range, d_model, bn_warmup=bn_warmup))
            extra += d_model
        add("readout_fc", nn.Linear(2 * seq_width + (d_model if self.use_stats else 0) + extra, d_model))
        lin_width = nf * (2 if self.level_passthrough else 1)
        add("linear_path", nn.Linear(T * lin_width, d_model))

    def _m(self, name, sfx=""):
        return getattr(self, name + sfx, None)

    def _tap(self, name: str, value):
        """Record a stream tensor for ``model.taps.stream_taps`` (no-op, no copy, unless a recorder is active)."""
        if self._stream is not None and value is not None:
            if name in self._stream_edit:        # taps.edit_stream: an ablation replaces the stream from here on
                value = self._stream_edit[name](value)
            self._stream[name] = value
        return value

    def branch(self, x: torch.Tensor, sfx: str = "", gid=None) -> torch.Tensor:
        """One timeframe's encoder up to its pooled embedding h (after readout_fc + linear_path). Stream taps (in
        order): input, scaled, inst_norm, stats, level, tokens, enc_in, enc_block_i, enc_out, pooled, readout,
        linear_path, branch_out (+ the branch suffix). ``gid``: the rows' timestamp ids (cross_input reads them)."""
        return self.branch_tokens(x, sfx, gid)[0]

    def branch_tokens(self, x: torch.Tensor, sfx: str = "", gid=None) -> tuple[torch.Tensor, torch.Tensor]:
        """``branch`` plus the encoder's output tokens (B, T', d_model) after final_norm (attention heads' input)."""
        tap = lambda name, v: self._tap(name + sfx, v)  # noqa: E731
        x = tap("input", x)
        raw = x
        if self._m("input_scaler", sfx) is not None:
            x = tap("scaled", self._m("input_scaler", sfx)(x))
        if hasattr(self, "cross_input"):
            x = tap("cross_input", self.cross_input(x, gid))
        z, stats = self._m("instance_norm", sfx)(x)
        z = tap("inst_norm", z)
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
            tok = torch.cat([tok, tap("level", lvl)], dim=-1)
        if stats is not None and self._m("stats_bn", sfx) is not None:
            stats = self._m("stats_bn", sfx)(stats)
        stats = tap("stats", stats)
        h = self.encode_tokens(tap("tokens", tok), sfx)
        h = tap("enc_out", self._m("final_norm", sfx)(h))
        parts = [self._m("last_token", sfx)(h), self._m("attn_pool", sfx)(h)]
        if stats is not None:
            parts.append(F.gelu(self._m("stats_proj", sfx)(stats)))
        if hasattr(self, "skip") and self.skip.readout_extra_dim:
            parts.append(self.skip.readout_extra(tok))
        if self._m("har_range", sfx) is not None:
            parts.append(tap("har_range", self._m("har_range", sfx)(raw)))
        lin_in = z if lvl is None else torch.cat([z, lvl], dim=-1)
        readout = tap("readout", F.gelu(self._m("readout_fc", sfx)(tap("pooled", torch.cat(parts, dim=-1)))))
        return tap("branch_out", readout + tap("linear_path", self._m("linear_path", sfx)(lin_in.flatten(1)))), h

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

    def trunk(self, x, gid=None) -> torch.Tensor:
        """x: (B, T, F) tensor, or for multi-timeframe a dict {timeframe: (B, T_tf, F_tf)} / sequence in the
        timeframe order. Returns trunk_drop's output (B, d_model * n_timeframes) — the panel encoder output."""
        xs = [xi.to(torch.float32) for xi in self._branch_inputs(x)]
        hs = [self.branch(xi, sfx, gid) for xi, sfx in zip(xs, self.branch_suffixes, strict=True)]
        h = hs[0] if len(hs) == 1 else torch.cat(hs, dim=-1)  # branch_concat
        return self.trunk_drop(self._tap("trunk", self.trunk_norm(h)))

    def encode_tokens(self, tok: torch.Tensor, sfx: str = "") -> torch.Tensor:
        """The registered encoder on the normalised token input (B, T, W) -> output sequence (for the transformer
        this is the input_skip seam, P34)."""
        return ENCODERS.get(self.encoder).encode(self, tok, sfx)

    def forward(self, x) -> dict[str, torch.Tensor]:
        gid = None
        if torch.is_tensor(x) and self.n_group_features is not None:  # an optional last channel = the row's timestamp
            x, gid = split_group(x, self.n_group_features)
        h, raw = self._shared_and_raw_heads(x, gid)
        mus, betas = self._nig_means(raw, gid)
        out = self._nig_outputs(raw, mus, betas, h)
        out.update(self._class_outputs(h))
        out.update(self._aux_outputs(h))
        return out

    def _shared_and_raw_heads(self, x, gid):
        """(h, raw): the shared representation (the confidence / class / aux heads read it) and every NIG head's raw
        output ``{target: (mu, nu, alpha, beta)}``."""
        if not reads_sequence(self.nig_head_kind):
            h = self.trunk(x, gid)
            if hasattr(self, "cross_coin"):
                h = self._tap("cross_coin", self.cross_coin(h, gid))
            h_nig = grad_scale(h, self.nig_trunk_grad)   # FX2: 1.0 = h itself
            return h, {t: getattr(self, f"nig_{t}")(h_nig) for t in self.nig_targets}
        # experiment: the NIG heads read the head sequence, the confidence / class heads the pooled trunk
        (xi,) = self._branch_inputs(x)
        br, tokens = self.branch_tokens(xi.to(torch.float32), gid=gid)
        h = self.trunk_drop(self._tap("trunk", self.trunk_norm(br)))
        seq = self.head_seq_norm(torch.cat([br[:, None, :], tokens], dim=1))
        return h, {t: getattr(self, f"nig_{t}")(grad_scale(seq, self.nig_trunk_grad)) for t in self.nig_targets}

    def _nig_means(self, raw: dict, gid) -> tuple[dict, dict]:
        """Means and betas in label units: the order of the means (in the heads' space, or on prices after the inverse
        target-norm map: ``order_in_label_units``), the inverse map, the cross-section centring of the close mean."""
        if self.apply_order and not self.order_in_label_units:
            mu_h, mu_c, mu_l = self.ordered_means(raw["close"][0], raw["high"][0], raw["low"][0])
            mus = {"high": mu_h, "close": mu_c, "low": mu_l}
        else:
            mus = {t: raw[t][0] for t in self.nig_targets}
        betas = {t: raw[t][3] for t in self.nig_targets}
        if hasattr(self, "target_norm"):  # the heads work in z = (y - mean) / std: back to label units
            for t in self.nig_targets:
                mus[t], betas[t] = self.target_norm.inverse(mus[t], betas[t], t)
        if self.order_in_label_units:     # the order on prices: high / low = close +- (their label sd) x offset
            tn = self.target_norm
            mu_h, mu_c, mu_l = self.ordered_means(mus["close"], raw["high"][0], raw["low"][0],
                                                  unit_h=tn.std_of("high"), unit_l=tn.std_of("low"))
            mus.update(high=mu_h, close=mu_c, low=mu_l)
        if self.cross_center and gid is not None and "close" in mus:
            mus["close"] = center_groups(mus["close"], gid)
        return mus, betas

    def _nig_outputs(self, raw: dict, mus: dict, betas: dict, h) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for t in self.nig_targets:
            _, nu, alpha, _ = raw[t]
            beta = betas[t]
            epi, ale = getattr(self, f"unc_{t}")(nu, alpha, beta)
            out.update(nig_outputs(self.head_specs[t], mus[t], nu, alpha, beta, epi, ale,
                                   getattr(self, f"conf_{t}")(h, nu, alpha, beta)))
        return out

    def _class_outputs(self, h) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for t, kind in self.class_kind.items():
            spec = self.class_specs[t]
            out_layer = getattr(self, spec.output_keys["logits"])
            probs = HEAD_REGISTRY.get(kind)(h, getattr(self, f"class_fc_{t}"), out_layer,
                                            getattr(self, f"class_drop_{t}"))
            out.update(class_outputs(spec, probs))
        return out

    def _aux_outputs(self, h) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.aux_specs:
            h_aux = grad_scale(h, self.aux_trunk_grad)
            for name, spec in self.aux_specs.items():
                fc, out_layer, drop = (getattr(self, f"aux_{part}_{name}") for part in ("fc", "out", "drop"))
                values = HEAD_REGISTRY.get(spec.kind)(h_aux, fc, out_layer, drop)
                out.update(aux_outputs(spec, values))
        return out

    @property
    def aux_kinds(self) -> dict[str, list[str]]:
        """{target: [auxiliary head kind, ...]} of the aux heads (the trainer adds their losses; {} = none)."""
        return {AUX_TARGET: [spec.kind for spec in self.aux_specs.values()]} if self.aux_specs else {}

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
        aux = [f"aux_fc_{name}" for name in self.aux_specs]
        return tuple(n for n in branch + heads + [f"class_fc_{t}" for t in self.class_targets] + aux if n in names)

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


def build_nig_timenet_v2(seq_len: int, n_features: int, **options) -> NigTimeNetV2:
    """The core keyword names and defaults are the upstream build_nig_timenet_v2's (``n_classes`` only matters for
    the multiclass head, P17); the options described below are port-only additions (TF has none of them). The
    anti-memorization options (encoder="gru", stats_mode, input_clip, input_noise_std,
    feature_dropout, level_passthrough, level_norm; P33b) and the FX1 switches (nig_scale, order_detach_close: see
    nig_layers; bn_warmup: layers.BatchNormChannelsLast; readout_dropout: the rate of trunk_drop and of the NIG
    heads' dropout, None = ``dropout``; FX2: nig_trunk_grad (nig_layers.grad_scale), nig_scale_fit (model.label_scale))
    are opt-in: their defaults reproduce the P15 graph (the run default switches
    some on, D-023). ``input_scaler`` (a name of
    scalers.INPUT_SCALERS) and ``nig_scale="train_std"`` add train-fitted buffers: call ``scalers.fit_scalers``.
    ``ffn`` / ``ffn_dim``: the feed-forward sublayer of the transformer blocks (TransformerBlock.FFN_KINDS; "none" =
    attention only), read by the transformer and itransformer encoders. ``cross_coin_layers`` > 0 adds
    model/cross_coin.py between the trunk and the heads: the input then carries one extra channel, the row's timestamp
    id (``workflow`` ``SplitSettings.cross_section``), and batches must hold whole timestamps. ``cross_center``
    (D-032): the close mean leaves the model minus its mean over the rows of the timestamp (a relative move; rows
    without the timestamp channel are left as they are). ``target_norm`` (a name of ``target_norm.TARGET_NORMS``,
    owner 2026-11-17): ``logret_std`` makes the NIG heads work on per-head train-standardised labels and maps their
    outputs back to label units (call ``target_norm.fit_target_norm`` before the label scale is fitted).
    ``har_range`` (H3, owner 2026-11-18; None = off): the (high, low) channel indices of the input; the HAR range terms
    of the RAW window (model/har_range.py) join the readout. ``use_instance_norm=False`` (owner 2026-11-18): the
    per-window normalisation is skipped (``InstanceNorm(normalize=False)``: z = the scaled input; stats unchanged).
    ``order_in_label_units`` (owner 2026-11-19, B-084; needs the ordered means and a fitted ``target_norm``): the
    order mu_low <= mu_close <= mu_high is applied on prices after the inverse map (high / low = close +- their label
    sd x the offset) instead of in the per-head normalised space, where it has no meaning. False = as before.
    ``aux_heads`` (owner 2026-11-19, LS1; () = none): auxiliary head kinds of model.heads (task types
    ``AUX_TASK_TYPES``: side_payoff, upper_quantile) on the close target, each reading the trunk through
    ``grad_scale(h, aux_trunk_grad)`` (0 = a read-out: the trunk and the other heads train as without them)."""
    known = {k: v for k, v in options.items() if k in _OPTION_NAMES}
    o = BuilderOptions(**known)
    ENCODERS.entry(o.encoder)
    TARGET_NORMS.entry(o.target_norm)  # raises UnknownNameError (a ValueError) with the known names + did-you-mean
    INPUT_SKIP_VARIANTS.entry(o.input_skip)  # likewise
    if skip_variant(o.input_skip).adds_modules and not ENCODERS.entry(o.encoder).meta["supports_input_skip"]:
        raise NotImplementedError(
            f"input_skip={o.input_skip!r} is defined on the transformer encoder only (not planned)")
    if o.ffn not in TransformerBlock.FFN_KINDS:
        raise ValueError(f"ffn must be one of {tuple(TransformerBlock.FFN_KINDS)}, not {o.ffn!r}")
    if o.level_norm not in LEVEL_NORMS:
        raise ValueError(f"level_norm must be one of {LEVEL_NORMS}, not {o.level_norm!r}")
    for opt, value in options.items():
        if opt in _OPTION_NAMES:
            continue
        if opt not in _UNIMPLEMENTED_OPTIONS:
            raise TypeError(f"build_nig_timenet_v2() got an unexpected keyword argument {opt!r}")
        supported = _UNIMPLEMENTED_OPTIONS[opt]
        if value != supported:
            raise NotImplementedError(f"{opt}={value!r} is not implemented (only {supported!r}; no step planned)")
    if isinstance(n_features, dict) and not isinstance(seq_len, dict):
        raise ValueError("n_features is a dict but seq_len is not: pass both as {timeframe: value}")
    o = replace(o, kernel_sizes=tuple(o.kernel_sizes), head_types=o.head_types or DEFAULT_HEAD_TYPES,
                aux_heads=tuple(o.aux_heads))
    return NigTimeNetV2(seq_len if isinstance(seq_len, dict) else int(seq_len),
                        n_features if isinstance(n_features, dict) else int(n_features), o)
