"""PURPOSE: Loading the TF oracle's weights (weights.json attr_paths) into a torch model: the name / layout
    translation (Keras kernel -> torch weight) and ``OracleWeights.load_oracle_weights`` (a mixin of NigTimeNetV2).
TAGS: oracle weights, keras to torch, attr path, weights json, load oracle weights, parity
PITFALLS: raises on any unmatched parameter, unknown key or shape mismatch (no silent partial load); a module with
    ``keras_weight_map`` (KerasGRU) converts its own weights; moved out of model/builder.py unchanged (refactor).
"""

import torch

from nig_timenet_torch.model.layers import keras_conv1d_weight_to_torch, keras_dense_weight_to_torch

__all__ = ["torch_param_name", "keras_to_torch_value", "OracleWeights"]


def torch_param_name(attr_path: str) -> str:
    """weights.json attr_path -> torch parameter name (Keras 'kernel' -> torch 'weight'; 'pos_emb' -> 'pos')."""
    module, _, leaf = attr_path.rpartition(".")
    leaf = {"kernel": "weight", "pos_emb": "pos"}.get(leaf, leaf)
    return f"{module}.{leaf}"


def keras_to_torch_value(attr_path: str, value: torch.Tensor) -> torch.Tensor:
    if attr_path.endswith(".kernel"):
        return keras_conv1d_weight_to_torch(value) if value.ndim == 3 else keras_dense_weight_to_torch(value)
    return value


class OracleWeights:
    """Mixin of the models that take the oracle's weights."""

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
