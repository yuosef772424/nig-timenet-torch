"""P31: the boundary contract every architecture must satisfy (D-012 point 3 / TF-side S23).

A ModelContract describes ONLY a model's boundary: input names/shapes/dtypes and output names/shapes/dtypes plus
``head_kind`` (how to decode an output, from HEAD_REGISTRY). It never reaches into a model's layers.
``contract_of`` reads it from any built model by running one forward pass in eval mode; ``check_contract`` raises
ContractMismatchError with a field-level diff (e.g. ``outputs['y_close'].shape: expected (None, 2), got (None, 1)``).
Dimension 0 is the batch and is recorded as None.
"""

import re
from dataclasses import dataclass

import torch

__all__ = ["InputSpec", "OutputSpec", "ModelContract", "ContractMismatchError", "contract_of", "check_contract",
           "head_kind_of"]


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int | None, ...]
    dtype: str


@dataclass(frozen=True)
class OutputSpec:
    name: str
    shape: tuple[int | None, ...]
    dtype: str
    head_kind: str


@dataclass(frozen=True)
class ModelContract:
    inputs: tuple[InputSpec, ...]
    outputs: tuple[OutputSpec, ...]


class ContractMismatchError(ValueError):
    pass


def _key_patterns() -> list[tuple[re.Pattern, str]]:
    from nig_timenet_torch.model.heads import HEAD_REGISTRY  # tier: core reads the head vocabulary lazily

    pats = []
    for kind in HEAD_REGISTRY.names():
        meta = HEAD_REGISTRY.entry(kind).meta
        for tmpl in {**meta.get("extra_output_keys", {}), **meta["output_keys"]}.values():
            rx = re.escape(tmpl).replace(re.escape("{target}"), r"(?P<target>[A-Za-z0-9]+)")
            pats.append((re.compile(f"^{rx}$"), kind))
    return sorted(pats, key=lambda p: -len(p[0].pattern))  # most specific template first


def head_kind_of(output_name: str) -> str:
    """HEAD_REGISTRY kind whose output-key template matches ``output_name``; raises if none does."""
    for rx, kind in _key_patterns():
        if rx.match(output_name):
            return kind
    raise ContractMismatchError(f"output {output_name!r} matches no registered head kind's output keys")


def _shape(t: torch.Tensor) -> tuple[int | None, ...]:
    return (None, *t.shape[1:])


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


@torch.no_grad()
def contract_of(model: torch.nn.Module, sample_input: torch.Tensor, *, input_name: str = "x") -> ModelContract:
    was_training = model.training
    model.eval()
    try:
        out = model(sample_input)
    finally:
        model.train(was_training)
    if not isinstance(out, dict):
        raise ContractMismatchError(f"model forward must return a dict of named outputs, got {type(out).__name__}")
    outputs = tuple(OutputSpec(k, _shape(v), _dtype(v), head_kind_of(k)) for k, v in sorted(out.items()))
    return ModelContract((InputSpec(input_name, _shape(sample_input), _dtype(sample_input)),), outputs)


def _diff(section: str, expected: tuple, actual: tuple) -> list[str]:
    e, a = {s.name: s for s in expected}, {s.name: s for s in actual}
    lines = [f"{section}[{n!r}]: missing" for n in sorted(set(e) - set(a))]
    lines += [f"{section}[{n!r}]: unexpected" for n in sorted(set(a) - set(e))]
    for n in sorted(set(e) & set(a)):
        for field in e[n].__dataclass_fields__:
            ev, av = getattr(e[n], field), getattr(a[n], field)
            if tuple(ev) != tuple(av) if field == "shape" else ev != av:
                lines.append(f"{section}[{n!r}].{field}: expected {ev!r}, got {av!r}")
    return lines


def check_contract(model: torch.nn.Module, contract: ModelContract, sample_input: torch.Tensor) -> None:
    input_name = contract.inputs[0].name if contract.inputs else "x"
    actual = contract_of(model, sample_input, input_name=input_name)
    lines = _diff("inputs", contract.inputs, actual.inputs) + _diff("outputs", contract.outputs, actual.outputs)
    if lines:
        raise ContractMismatchError("model does not satisfy its contract:\n  " + "\n  ".join(lines))
