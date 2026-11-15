"""PURPOSE: Experimental NIG output head without a Dense readout: ScalarOutputAttention (one learned query per output
    attends over the head sequence) and ScalarNIGHead (mu, nu, alpha, beta from it); selected by ``nig_head``.
TAGS: attention head, scalar output attention, nig head, joint attention, factorized attention, head sequence, dense
    readout, experiment
PITFALLS: an EXPERIMENT (owner 2026-11-15, the idea comes from an AI-written note: not a decision): default
    ``nig_head='dense'`` builds the old model byte for byte. Single-timeframe only. The head sequence is
    [branch embedding ; encoder tokens after final_norm] normalised by ``head_seq_norm``; its length is fixed at build
    (``n_tokens``). Computed in float32. The doctor's head probes read ``head.fc`` and do not apply to this head.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nig_timenet_torch.model.encoders import ENCODERS
from nig_timenet_torch.model.nig_layers import _Scaled

__all__ = ["NIG_HEADS", "check_nig_head", "ScalarOutputAttention", "ScalarNIGHead", "n_encoder_tokens"]

NIG_HEADS = ("dense", "scalar_joint", "scalar_factorized")


def check_nig_head(name: str) -> str:
    if name not in NIG_HEADS:
        raise ValueError(f"nig_head must be one of {NIG_HEADS}, not {name!r}")
    return name


def n_encoder_tokens(encoder: str, seq_len: int, patch_len: int, stride: int) -> int:
    """How many tokens the registered encoder hands to the readout (``Encoder.n_tokens``)."""
    return ENCODERS.get(encoder).n_tokens(seq_len, patch_len, stride)


class ScalarOutputAttention(nn.Module):
    """(B, T, d) -> (B, J): output j has its own query q_j over the tokens.

    joint:       p_j = softmax over ALL (t, d) cells of  k_{t,d} q_{j,d} + tb_{j,t} + fl_{j,d}
    factorized:  p_j = softmax_t(q_j . k_t / sqrt(d) + tb_{j,t})  x  softmax_d(fl_j)
    y_j = gain_j * sum_{t,d} p_{j,t,d} * value_scale_{j,d} * x_{t,d} + offset_j      (k = Wk x, no bias)."""

    def __init__(self, d_model: int, n_tokens: int, num_outputs: int = 4, mode: str = "joint", dropout: float = 0.0,
                 l2: float = 1e-5):
        super().__init__()
        if mode not in ("joint", "factorized"):
            raise ValueError(f"mode must be 'joint' or 'factorized', not {mode!r}")
        self.d, self.t, self.j, self.mode, self.l2 = int(d_model), int(n_tokens), int(num_outputs), mode, l2
        self.wk = nn.Linear(self.d, self.d, bias=False)
        self.query = nn.Parameter(torch.randn(self.j, self.d) * 0.02)
        self.time_bias = nn.Parameter(torch.zeros(self.j, self.t))
        self.feat_logits = nn.Parameter(torch.zeros(self.j, self.d))
        self.value_scale = nn.Parameter(1.0 + 0.1 * torch.randn(self.j, self.d))
        self.gain = nn.Parameter(torch.ones(self.j))
        self.offset = nn.Parameter(torch.zeros(self.j))
        self.drop = nn.Dropout(dropout)

    def regularization_loss(self) -> torch.Tensor:
        return self.l2 * self.wk.weight.square().sum()

    def probs(self, k: torch.Tensor) -> torch.Tensor:
        """(B, J, T, d) attention probabilities."""
        if self.mode == "joint":
            s = torch.einsum("btd,jd->bjtd", k, self.query) + self.time_bias[None, :, :, None]
            s = s + self.feat_logits[None, :, None, :]
            p = torch.softmax(s.flatten(2), dim=-1).view(k.shape[0], self.j, self.t, self.d)
            return self.drop(p)
        a = torch.einsum("jd,btd->bjt", self.query, k) / math.sqrt(self.d) + self.time_bias[None]
        a = self.drop(torch.softmax(a, dim=-1))
        return a[..., None] * torch.softmax(self.feat_logits, dim=-1)[None, :, None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.t or x.shape[2] != self.d:
            raise ValueError(f"expected (B, {self.t}, {self.d}), got {tuple(x.shape)}")
        x32 = x.float()
        p = self.probs(self.wk(x32))
        v = x32[:, None] * self.value_scale[None, :, None, :]
        return (p * v).sum(dim=(2, 3)) * self.gain + self.offset


class ScalarNIGHead(_Scaled):
    """The head sequence -> (mu, nu, alpha, beta), each (B, 1): mu linear, nu / alpha / beta = softplus + floor, and
    the ``scale`` switch of NIGHead (label units / s). Same floors and output contract as ``NIGHead``."""

    def __init__(self, d_model: int, n_tokens: int, mode: str = "joint", dropout: float = 0.0, nu_min: float = 0.1,
                 alpha_min: float = 2.0, beta_min: float = 0.01, l2: float = 1e-5, scale: float | str | None = None):
        super().__init__()
        self._init_scale(scale)
        self.nu_min, self.alpha_min, self.beta_min = nu_min, alpha_min, beta_min
        self.attn = ScalarOutputAttention(d_model, n_tokens, 4, mode, dropout, l2)

    def regularization_loss(self) -> torch.Tensor:
        return self.attn.regularization_loss()

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.attn(seq)
        mu = p[:, 0:1]
        nu = F.softplus(p[:, 1:2]) + self.nu_min
        alpha = F.softplus(p[:, 2:3]) + self.alpha_min
        beta = F.softplus(p[:, 3:4]) + self.beta_min
        s = self._s()
        if s is not None:
            mu, beta = s * mu, (s * s) * beta
        return mu, nu, alpha, beta
