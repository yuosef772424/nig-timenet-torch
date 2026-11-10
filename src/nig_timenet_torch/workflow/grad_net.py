"""PURPOSE: GradNet (grad_stack variant grad_full): model B sees ONLY the full per-row gradients of a frozen model A
    (every layer and unit), one encoder per layer, a project backbone over the layer tokens.
TAGS: grad net, full gradients, per-layer encoders, frozen model a, gradient tokens, grad stack, experiment
PITFALLS: the gradients are computed inside every forward (never stored: ~6k floats per row x 320k rows does not fit),
    under ``torch.enable_grad`` (prediction runs under no_grad) and cudnn off (RNN backward in eval mode); A stays in
    eval mode and frozen even when the trainer calls ``train()``; "per_head" costs one gradient pass per A output.

Per row and per layer l of A: ``g_l`` = d(scalar_c)/d(output of l) for each A output scalar c (``per_head``: the mean,
class logit and confidence of every target; ``sum``: one scalar, the sum of all outputs), shape (C, D_l). The encoder
of l sees ``g_l / rms(g_l)`` and ``log rms(g_l)`` per channel (scale-free shape + magnitude) -> a d-dim token. The L
tokens (input gradient first, then A's layers in forward order) form a (L, d) sequence for the backbone (any project
architecture), so the backbone's heads, losses, tables and the battery work unchanged.
"""

import copy

import torch
from torch import nn

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.trainer.switches import param_groups
from nig_timenet_torch.trainer.trainer import Trainer
from nig_timenet_torch.workflow import flat_run, run, run_common

__all__ = ["GradNet", "GRAD_OUTPUTS", "build_grad_net_run"]

GRAD_OUTPUTS = ("per_head", "sum")
_PER_HEAD = ("{t}", "{t}_class_logits", "{t}_confidence")      # A outputs given their own gradient channel


def _learning_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules()
            if not list(m.children()) and any(p.requires_grad for p in m.parameters(recurse=False))]


class GradNet(nn.Module):
    """B = per-layer gradient encoders + backbone; ``model_a`` is copied, frozen and kept in eval mode."""

    def __init__(self, model_a: nn.Module, x_probe: torch.Tensor, backbone: ModelSettings, *, dim: int = 32,
                 outputs: str = "per_head"):
        super().__init__()
        if outputs not in GRAD_OUTPUTS:
            raise ValueError(f"outputs must be one of {GRAD_OUTPUTS}, got {outputs!r}")
        self.a = copy.deepcopy(model_a).eval()
        layers = _learning_layers(self.a)          # before freezing: "learning" = owns trainable parameters
        for p in self.a.parameters():
            p.requires_grad_(False)
        self.outputs, self._acts = outputs, {}
        for n, m in layers:
            m.register_forward_hook(self._hook(n))
        dev = next(self.a.parameters()).device
        with torch.no_grad():
            out = self.a(x_probe[:2].to(dev))
        self.keys = (sorted(out) if outputs == "sum" else
                     [f.format(t=k) for k in sorted(out) if f"{k}_class_logits" in out for f in _PER_HEAD])
        self.names = ["input"] + [n for n, _ in layers if isinstance(self._acts.get(n), torch.Tensor)]
        grads = self.gradients(x_probe[:2].to(dev))
        self.enc = nn.ModuleList(nn.Sequential(nn.Linear(g.shape[1] * (g.shape[2] + 1), dim), nn.GELU(),
                                               nn.LayerNorm(dim)) for g in grads)
        self.backbone = build_model(backbone, len(self.names), dim)

    def _hook(self, name):
        def f(_m, _i, out):
            self._acts[name] = out
        return f

    def train(self, mode: bool = True):
        super().train(mode)
        self.a.eval()                        # A is fixed: dropout off, batch norm on its moving statistics
        return self

    def gradients(self, x: torch.Tensor) -> list[torch.Tensor]:
        """[(n, C, D_l)] per layer (input first): the full gradients of A's output scalars, detached."""
        self._acts.clear()
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            xa = x.detach().requires_grad_(True)
            out = self.a(xa)
            scalars = [sum(v.sum() for v in out.values())] if self.outputs == "sum" else [out[k].sum()
                                                                                        for k in self.keys]
            wrt = [xa] + [self._acts[n] for n in self.names[1:]]
            live = [i for i, w in enumerate(wrt) if w.requires_grad]   # a detached layer output has no gradient
            per = []
            for s in scalars:   # an output computed from detached features has no gradient: its channel stays zero
                got = (torch.autograd.grad(s, [wrt[i] for i in live], retain_graph=True, allow_unused=True)
                       if s.requires_grad else (None,) * len(live))
                full = [None] * len(wrt)
                for i, gi in zip(live, got, strict=True):
                    full[i] = gi
                per.append(full)
        n = x.shape[0]
        return [torch.stack([(p[j] if p[j] is not None else torch.zeros_like(w)).reshape(n, -1) for p in per],
                            dim=1).detach() for j, w in enumerate(wrt)]

    def forward(self, x: torch.Tensor) -> dict:
        tokens = []
        for g, enc in zip(self.gradients(x), self.enc, strict=True):
            rms = g.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-12
            tokens.append(enc(torch.cat([(g / rms).flatten(1), rms.squeeze(2).log()], dim=1)))
        return self.backbone(torch.stack(tokens, dim=1))


def build_grad_net_run(r: run.Run, model_a: nn.Module, *, dim: int, outputs: str) -> run.Run:
    """``run.build`` for a GradNet: same seed, trainer config, optimizer and identity path as ``flat_run.make_trainer``
    (the backbone is the run's model settings)."""
    s, t = r.settings, r.settings.train
    seed_everything(t.seed)
    r.ms = flat_run.model_settings(s.model, r.targets)
    model = GradNet(model_a, r.x["train"], r.ms, dim=dim, outputs=outputs).to(r.device)
    r.trainer = Trainer(model, flat_run.trainer_config(r.targets, t.label_smoothing, t),
                        selection_metric=t.selection_metric,
                        optimizer=lambda ps: torch.optim.AdamW(
                            param_groups(t.weight_decay_groups, model, ps, t.weight_decay), lr=t.lr,
                            weight_decay=t.weight_decay))
    r.fit_settings = flat_run.fit_settings_of(s)
    r.ident = run_common.run_identity(r.trainer, r.fit_settings, r.ms, r.art, s)
    return r
