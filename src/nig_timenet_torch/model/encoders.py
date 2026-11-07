"""ENCODERS: the registered sequence encoders of NigTimeNetV2 (transformer P15, GRU P33b, TCN P40).

PURPOSE:  The ENCODERS registry: each encoder builds its modules through the model's ``add(name, module)`` (Keras
          layer names, so state_dict keys are unchanged) and owns its forward over the token sequence.
TAGS:     encoders, encoder registry, transformer, gru, tcn, causal conv, sequence encoder, plugin
PITFALLS: Module names (``patch_embed``, ``step_proj``, ``block_i``, ``gru_i``, ``tcn_conv_i``, ...) are weights.json
          attr_paths and checkpoint keys: never rename. A new encoder = one class + one ``ENCODERS.register`` line;
          ``supports_input_skip`` says whether the input_skip variants apply to it (transformer only today).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.model.layers import KerasGRU, PatchEmbedding, RMSNorm, TransformerBlock
from nig_timenet_torch.model.parts import build_input_skip, skip_variant

__all__ = ["ENCODERS", "Encoder", "EncoderDims", "CausalConv1D", "tcn_n_blocks"]

ENCODERS = Registry("encoder", Contract("encoder", required_meta=("supports_input_skip",)))


@dataclass(frozen=True)
class EncoderDims:
    """The size and attention options every encoder reads (a subset applies to each)."""
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    patch_len: int
    stride: int
    max_rel_pos: int
    dropout: float
    attn_dropout: float
    causal: bool
    window: int | None


class CausalConv1D(nn.Conv1d):
    """Keras Conv1D(padding="causal"): left-pad (kernel_size - 1) * dilation zeros, then a valid convolution.
    Channels-last in and out, like the Keras layer; weight is torch (out, in, k)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x.transpose(1, 2), (self.left_pad, 0))).transpose(1, 2)


def tcn_n_blocks(seq_len: int) -> int:
    """Upstream: max(1, ceil(log2(max(T, 2))) - 1) blocks, receptive field 1 + 2 * (2^n - 1) >= T."""
    return max(1, int(math.ceil(math.log2(max(seq_len, 2)))) - 1)


def _get(model: nn.Module, name: str, sfx: str):
    return getattr(model, name + sfx)


class Encoder:
    """Base: ``build`` adds the encoder's modules to the model branch ``sfx`` and returns the output sequence width;
    ``encode`` maps the normalised token input (B, T, W) to the output sequence (B, n, width)."""

    def n_blocks(self, dims: EncoderDims, seq_len: int) -> int:
        return dims.num_layers

    def build(self, model, add, sfx: str, *, seq_len: int, tok_width: int, dims: EncoderDims) -> int:
        raise NotImplementedError

    def encode(self, model, tok: torch.Tensor, sfx: str) -> torch.Tensor:
        raise NotImplementedError


class TransformerEncoder(Encoder):
    def build(self, model, add, sfx, *, seq_len, tok_width, dims):
        variant = skip_variant(model.input_skip)
        seq_width = dims.d_model
        if variant.replaces_patch_embed:
            seq_width = tok_width
        else:
            add("patch_embed", PatchEmbedding(tok_width, dims.d_model, dims.patch_len, dims.stride, seq_len=seq_len))
        if variant.adds_modules:  # "none" adds no module: P15 parameter names stay identical
            model.skip = build_input_skip(model.input_skip, d_model=dims.d_model, tok_width=tok_width,
                                          num_layers=dims.num_layers, patch_len=dims.patch_len, stride=dims.stride)
        model.branch_blocks[sfx] = self.n_blocks(dims, seq_len)
        for i in range(model.branch_blocks[sfx]):
            add(f"block_{i + 1}", TransformerBlock(seq_width, dims.num_heads, dims.num_kv_heads, dims.max_rel_pos,
                                                   dims.causal, dims.window, dims.dropout, dims.attn_dropout))
        return seq_width

    def encode(self, model, tok, sfx):
        """Per-position: with causal=True, position j depends on no bar after the end of its patch (or bar)."""
        blocks = [_get(model, f"block_{i + 1}", sfx) for i in range(model.branch_blocks[sfx])]
        if hasattr(model, "skip"):  # no_proj has no patch_embed: the variant gets None
            return model.skip.encode(tok, getattr(model, "patch_embed" + sfx, None), blocks)
        h = _get(model, "patch_embed", sfx)(tok)
        for block in blocks:
            h = block(h)
        return h


class GRUEncoder(Encoder):
    """step_proj, then per block RMSNorm -> GRU -> Dropout, residual add."""

    def build(self, model, add, sfx, *, seq_len, tok_width, dims):
        add("step_proj", nn.Linear(tok_width, dims.d_model))
        model.branch_blocks[sfx] = self.n_blocks(dims, seq_len)
        for i in range(model.branch_blocks[sfx]):
            add(f"gru_norm_{i + 1}", RMSNorm(dims.d_model))
            add(f"gru_{i + 1}", KerasGRU(dims.d_model, dims.d_model))
            add(f"gru_drop_{i + 1}", nn.Dropout(dims.dropout))
        return dims.d_model

    def encode(self, model, tok, sfx):
        h = _get(model, "step_proj", sfx)(tok)
        for i in range(model.branch_blocks[sfx]):
            y = _get(model, f"gru_{i + 1}", sfx)(_get(model, f"gru_norm_{i + 1}", sfx)(h))
            h = h + _get(model, f"gru_drop_{i + 1}", sfx)(y)
        return h


class TCNEncoder(Encoder):
    """step_proj, then per block RMSNorm -> causal Conv1D(k=3, dilation 2^i, gelu) -> Dropout, residual add."""

    def n_blocks(self, dims, seq_len):
        return tcn_n_blocks(seq_len)

    def build(self, model, add, sfx, *, seq_len, tok_width, dims):
        add("step_proj", nn.Linear(tok_width, dims.d_model))
        model.branch_blocks[sfx] = self.n_blocks(dims, seq_len)
        for i in range(model.branch_blocks[sfx]):
            add(f"tcn_norm_{i + 1}", RMSNorm(dims.d_model))
            add(f"tcn_conv_{i + 1}", CausalConv1D(dims.d_model, dims.d_model, 3, 2 ** i))
            add(f"tcn_drop_{i + 1}", nn.Dropout(dims.dropout))
        return dims.d_model

    def encode(self, model, tok, sfx):
        h = _get(model, "step_proj", sfx)(tok)
        for i in range(model.branch_blocks[sfx]):
            y = F.gelu(_get(model, f"tcn_conv_{i + 1}", sfx)(_get(model, f"tcn_norm_{i + 1}", sfx)(h)))
            h = h + _get(model, f"tcn_drop_{i + 1}", sfx)(y)
        return h


ENCODERS.register("transformer", TransformerEncoder(), version=1, supports_input_skip=True)
ENCODERS.register("gru", GRUEncoder(), version=1, supports_input_skip=False)
ENCODERS.register("tcn", TCNEncoder(), version=1, supports_input_skip=False)
