"""Output heads (source: crypto-signal-prediction model/heads.py @ aeafc564): HeadSpec, HEAD_REGISTRY and the
binary-classification head's forward (BUILD_PLAN.md P14/P17).

One HEAD_REGISTRY entry per head KIND; its metadata are templates with ``{target}`` that ``head_spec(kind, target)``
fills into a HeadSpec — the single vocabulary linking a head's output keys, its label key and its loss task type
(section 1, C1). nig_regression is registered too (built in builder.py, which needs all three targets at once for
OrderedMeans), so the registry lists every kind the builder accepts.
"""

from dataclasses import dataclass

import torch
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["HeadSpec", "HEAD_REGISTRY", "head_spec", "classification_head_forward", "multiclass_head_forward"]


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: str
    output_keys: dict[str, str]
    label_key: str
    task_type: str
    label_encoding: str
    default_weight: float
    eval_role: str


HEAD_REGISTRY = Registry("head", Contract(
    "head", required_meta=("output_keys", "label_key", "task_type"),
    optional_meta=("head_name", "label_encoding", "default_weight", "eval_role", "extra_output_keys")))


def head_spec(kind: str, target: str) -> HeadSpec:
    meta = HEAD_REGISTRY.entry(kind).meta
    return HeadSpec(name=meta.get("head_name", "{target}").format(target=target), kind=kind,
                    output_keys={role: k.format(target=target) for role, k in meta["output_keys"].items()},
                    label_key=meta["label_key"].format(target=target), task_type=meta["task_type"],
                    label_encoding=meta.get("label_encoding", ""),
                    default_weight=float(meta.get("default_weight", 1.0)), eval_role=meta.get("eval_role", ""))


def classification_head_forward(x: torch.Tensor, fc: nn.Linear, out: nn.Linear,
                                drop: nn.Module | None = None) -> torch.Tensor:
    """Dense(relu) -> Dropout -> Dense(sigmoid). The output is a PROBABILITY although upstream names it
    ``y_{target}_class_logits``."""
    h = torch.relu(fc(x))
    if drop is not None:
        h = drop(h)
    return torch.sigmoid(out(h))


def multiclass_head_forward(x: torch.Tensor, fc: nn.Linear, out: nn.Linear,
                            drop: nn.Module | None = None) -> torch.Tensor:
    """Dense(relu) -> Dropout -> Dense(n_classes, softmax): class PROBABILITIES under the name
    ``y_{target}_class_logits`` (upstream convention; the loss uses from_logits=False)."""
    h = torch.relu(fc(x))
    if drop is not None:
        h = drop(h)
    return torch.softmax(out(h), dim=-1)


HEAD_REGISTRY.register(
    "nig_regression", None, version=1, head_name="{target}_reg",
    output_keys={"mu": "y_{target}", "nu": "y_{target}_nu", "alpha": "y_{target}_alpha", "beta": "y_{target}_beta",
                 "confidence": "y_{target}_confidence"},
    label_key="y_{target}_reg", task_type="evidential", label_encoding="continuous", default_weight=1.0,
    eval_role="price",
    # model outputs of this head kind that no loss reads (so not in output_keys); used by core.model_contract (P31)
    extra_output_keys={"epistemic": "y_{target}_epistemic", "aleatoric": "y_{target}_aleatoric"})
HEAD_REGISTRY.register(
    "binary_classification", classification_head_forward, version=1, head_name="{target}_class",
    output_keys={"logits": "y_{target}_class_logits"}, label_key="y_{target}_class", task_type="classification",
    label_encoding="binary", default_weight=1.0, eval_role="direction")
HEAD_REGISTRY.register(
    "multiclass_classification", multiclass_head_forward, version=1, head_name="{target}_class",
    output_keys={"logits": "y_{target}_class_logits"}, label_key="y_{target}_class", task_type="classification",
    label_encoding="multiclass", default_weight=1.0, eval_role="direction")
