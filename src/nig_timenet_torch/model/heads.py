"""Non-NIG output heads (source: crypto-signal-prediction model/heads.py @ aeafc564). HEAD_REGISTRY / HeadSpec
arrive in P17; for now only the binary-classification head's forward."""

import torch
from torch import nn

__all__ = ["classification_head_forward"]


def classification_head_forward(x: torch.Tensor, fc: nn.Linear, out: nn.Linear,
                                drop: nn.Module | None = None) -> torch.Tensor:
    """Dense(relu) -> Dropout -> Dense(sigmoid). The output is a PROBABILITY although upstream names it
    ``y_{target}_class_logits``."""
    h = torch.relu(fc(x))
    if drop is not None:
        h = drop(h)
    return torch.sigmoid(out(h))
