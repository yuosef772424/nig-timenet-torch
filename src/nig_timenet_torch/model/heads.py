"""Output heads (source: crypto-signal-prediction model/heads.py @ aeafc564): HeadSpec, HEAD_REGISTRY and the
classification heads' forwards (BUILD_PLAN.md P14/P17).

PURPOSE:  The ONE head vocabulary: HEAD_REGISTRY (a head KIND -> output keys, label key, task type, output width,
          forward), head_spec, and the nig_outputs / class_outputs helpers that assemble a head's output dict.
TAGS:     heads, head registry, head spec, output keys, nig outputs, class outputs, task type, output width
PITFALLS: nig_regression is registered with no forward (the builder needs all three targets at once for OrderedMeans);
          a classification kind's forward returns PROBABILITIES under a key named ``..._class_logits``. Builders and
          architectures assemble output dicts ONLY through nig_outputs / class_outputs (rule G1).

One HEAD_REGISTRY entry per head KIND; its metadata are templates with ``{target}`` that ``head_spec(kind, target)``
fills into a HeadSpec — the single vocabulary linking a head's output keys, its label key and its loss task type
(section 1, C1). nig_regression is registered too, so the registry lists every kind the builder accepts. Metadata
``output_width``: width of a classification head's last Linear (1 = binary; None = the model's ``n_classes``).
"""

from dataclasses import dataclass

import torch
from torch import nn

from nig_timenet_torch.core.registry import Contract, Registry

__all__ = ["HeadSpec", "HEAD_REGISTRY", "head_spec", "classification_head_forward", "multiclass_head_forward",
           "kinds_with_task", "class_head_kind", "head_width", "nig_outputs", "class_outputs"]


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
    extra_output_keys: dict[str, str]


HEAD_REGISTRY = Registry("head", Contract(
    "head", required_meta=("output_keys", "label_key", "task_type"),
    optional_meta=("head_name", "label_encoding", "default_weight", "eval_role", "extra_output_keys",
                                     "output_width")))


def head_spec(kind: str, target: str) -> HeadSpec:
    meta = HEAD_REGISTRY.entry(kind).meta
    return HeadSpec(name=meta.get("head_name", "{target}").format(target=target), kind=kind,
                    output_keys={role: k.format(target=target) for role, k in meta["output_keys"].items()},
                    label_key=meta["label_key"].format(target=target), task_type=meta["task_type"],
                    label_encoding=meta.get("label_encoding", ""),
                    default_weight=float(meta.get("default_weight", 1.0)), eval_role=meta.get("eval_role", ""),
                    extra_output_keys={role: k.format(target=target)
                                       for role, k in meta.get("extra_output_keys", {}).items()})


def kinds_with_task(kinds, task_type: str) -> list[str]:
    """The head kinds of ``kinds`` whose registered task type is ``task_type`` ("evidential" / "classification");
    an unregistered kind raises UnknownNameError."""
    return [k for k in kinds if HEAD_REGISTRY.entry(k).meta["task_type"] == task_type]


def class_head_kind(kinds, target: str) -> str | None:
    """The one classification kind of a target's head list (None if it has none). Two of them would share the
    Keras layer names ``class_fc_{target}``, so that raises."""
    found = kinds_with_task(kinds, "classification")
    if len(found) > 1:
        raise ValueError(f"{target!r}: {found} share the Keras layer names class_fc_{target}")
    return found[0] if found else None


def head_width(kind: str, n_classes: int) -> int:
    """Width of a classification head's last Linear: the registered ``output_width``, else ``n_classes``."""
    meta = HEAD_REGISTRY.entry(kind).meta
    if "output_width" not in meta:
        raise ValueError(f"head kind {kind!r} has no classification output width")
    width = meta["output_width"]
    return int(n_classes if width is None else width)


def nig_outputs(spec: HeadSpec, mu, nu, alpha, beta, epistemic, aleatoric, confidence) -> dict:
    """The output dict of one nig_regression head, keys from ``spec`` (head_spec("nig_regression", target))."""
    k, e = spec.output_keys, spec.extra_output_keys
    return {k["mu"]: mu, k["nu"]: nu, k["alpha"]: alpha, k["beta"]: beta, e["epistemic"]: epistemic,
            e["aleatoric"]: aleatoric, k["confidence"]: confidence}


def class_outputs(spec: HeadSpec, probabilities) -> dict:
    """The output dict of one classification head: its probabilities under ``spec.output_keys["logits"]``."""
    return {spec.output_keys["logits"]: probabilities}


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
    # model outputs of this head kind that no loss reads (so not in output_keys); used by model.contract (P31)
    extra_output_keys={"epistemic": "y_{target}_epistemic", "aleatoric": "y_{target}_aleatoric"})
HEAD_REGISTRY.register(
    "binary_classification", classification_head_forward, version=1, head_name="{target}_class",
    output_keys={"logits": "y_{target}_class_logits"}, label_key="y_{target}_class", task_type="classification",
    label_encoding="binary", default_weight=1.0, eval_role="direction", output_width=1)
HEAD_REGISTRY.register(
    "multiclass_classification", multiclass_head_forward, version=1, head_name="{target}_class",
    output_keys={"logits": "y_{target}_class_logits"}, label_key="y_{target}_class", task_type="classification",
    label_encoding="multiclass", default_weight=1.0, eval_role="direction", output_width=None)
