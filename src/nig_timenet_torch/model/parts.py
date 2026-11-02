"""P34: the ``input_skip`` seam on the transformer encoder (TF-side S15; crypto-signal-prediction
docs/research/hardening/PLAN_2026-11.md:171-177). Registered variants; the default ``"none"`` is the P15 graph with
identical parameter names and shapes (checkpoint-compatible) — it adds no parameters at all.

Every variant maps the normalised token-input sequence ``tok`` (B, T, W) (trend|seasonal[|level], before any
projection) to the encoder's output sequence. "Raw tokens" ``r`` are the normalised input patchified exactly like
PatchEmbedding (same left pad, patch_len, stride): token j sees only the bars of patch j, so a variant built on r
is causal whenever the blocks are (no future bar reaches step t).

  none              h = blocks(patch_embed(tok))
  concat_each_block before block i: h = proj_i([h ; r])   (its own Linear(d + patch_len*W -> d) per block)
  dense_skip        block i input = proj_i([r ; h_0 ; ... ; h_{i-1}]), h_0 = patch_embed(tok) (DenseNet-style)
  late_raw          sequence as "none"; the last ``late_k`` normalised bars, flattened, join the readout concat
  no_proj           no patch/step projection: blocks run on tok at width W (one token per bar, W % num_heads == 0)
"""

import torch
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.model.layers import num_patches

__all__ = ["INPUT_SKIP_VARIANTS", "InputSkip", "patch_tokens", "build_input_skip", "LATE_RAW_K"]

INPUT_SKIP_VARIANTS = Registry("input_skip", Contract("input_skip", required_meta=("build_fn",)))
LATE_RAW_K = 4  # bars; the PLAN says "last-k", k unspecified (docs/friction_log.csv, P34)


def patch_tokens(tok: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """(B, T, W) -> (B, n_tokens, patch_len * W): the same patches PatchEmbedding convolves (first bar repeated as
    left pad), flattened time-major."""
    _, pad = num_patches(tok.shape[1], patch_len, stride)
    if pad > 0:
        tok = torch.cat([tok[:, :1, :].expand(-1, pad, -1), tok], dim=1)
    patches = tok.unfold(1, patch_len, stride)  # (B, n_tokens, W, patch_len)
    return patches.transpose(2, 3).flatten(2)


class InputSkip(nn.Module):
    """Base = "none". ``patchwise``: output position j is a patch (False: one position per bar)."""
    name = "none"
    patchwise = True
    readout_extra_dim = 0

    def __init__(self, *, d_model: int, tok_width: int, num_layers: int, patch_len: int, stride: int):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride

    def encode(self, tok: torch.Tensor, embed: nn.Module | None, blocks: list[nn.Module]) -> torch.Tensor:
        h = embed(tok)
        for block in blocks:
            h = block(h)
        return h

    def readout_extra(self, tok: torch.Tensor) -> torch.Tensor | None:
        return None


class ConcatEachBlock(InputSkip):
    name = "concat_each_block"

    def __init__(self, *, d_model, tok_width, num_layers, patch_len, stride):
        super().__init__(d_model=d_model, tok_width=tok_width, num_layers=num_layers, patch_len=patch_len,
                         stride=stride)
        self.projs = nn.ModuleList(nn.Linear(d_model + patch_len * tok_width, d_model) for _ in range(num_layers))

    def encode(self, tok, embed, blocks):
        r, h = patch_tokens(tok, self.patch_len, self.stride), embed(tok)
        for block, proj in zip(blocks, self.projs, strict=True):
            h = block(proj(torch.cat([h, r], dim=-1)))
        return h


class DenseSkip(InputSkip):
    name = "dense_skip"

    def __init__(self, *, d_model, tok_width, num_layers, patch_len, stride):
        super().__init__(d_model=d_model, tok_width=tok_width, num_layers=num_layers, patch_len=patch_len,
                         stride=stride)
        self.projs = nn.ModuleList(nn.Linear(patch_len * tok_width + (i + 1) * d_model, d_model)
                                   for i in range(num_layers))

    def encode(self, tok, embed, blocks):
        feats = [patch_tokens(tok, self.patch_len, self.stride), embed(tok)]
        h = feats[-1]
        for block, proj in zip(blocks, self.projs, strict=True):
            h = block(proj(torch.cat(feats, dim=-1)))
            feats.append(h)
        return h


class LateRaw(InputSkip):
    name = "late_raw"

    def __init__(self, *, d_model, tok_width, num_layers, patch_len, stride, late_k: int = LATE_RAW_K):
        super().__init__(d_model=d_model, tok_width=tok_width, num_layers=num_layers, patch_len=patch_len,
                         stride=stride)
        self.late_k = late_k
        self.readout_extra_dim = late_k * tok_width

    def readout_extra(self, tok):
        return tok[:, -self.late_k:, :].flatten(1)


class NoProj(InputSkip):
    name = "no_proj"
    patchwise = False

    def encode(self, tok, embed, blocks):
        h = tok
        for block in blocks:
            h = block(h)
        return h


for _cls in (InputSkip, ConcatEachBlock, DenseSkip, LateRaw, NoProj):
    INPUT_SKIP_VARIANTS.register(_cls.name, _cls, version=1, build_fn=_cls)


def build_input_skip(name: str, **dims) -> InputSkip:
    return INPUT_SKIP_VARIANTS.entry(name).meta["build_fn"](**dims)
