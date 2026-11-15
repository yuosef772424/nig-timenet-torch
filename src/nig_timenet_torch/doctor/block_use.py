"""PURPOSE: Card A, transformer part: is each transformer block used, and by which sublayer (attention or
    feed-forward)? Update size, attention entropy, and the change of mu and of the label rank IC when a part is off.
TAGS: doctor, card a, transformer block, attention, swiglu, feed forward, ablation, update ratio, attention entropy
PITFALLS: forward hooks on ``block.attn`` / ``block.ffn`` in eval mode, so the weights and the training are untouched;
    a switched-off sublayer is a ZERO update on a model that was trained WITH it: it shows what the trained model uses,
    not what a model trained without the sublayer would reach (that is a training arm: ``ffn="none"``). Rows are the
    rows the caller passes (card A: validation rows).

``block_use(model, x, y, targets) -> (checks, rows)``: one row per (block, part) with part in attn / ffn / block:
``update_ratio`` = RMS(sublayer output) / RMS(the stream it is added to), ``attn_entropy`` = mean entropy of the
attention rows / log(number of tokens) (1 = uniform attention: the block averages, it does not select),
``mu_change`` = max over heads of 1 - |corr(mu, mu without the part)|, ``d_ic`` = worst change of the label rank IC.
The last row (``all blocks``) switches every block off at once: what the whole attention stack is worth to the outputs.
"""

from contextlib import ExitStack, contextmanager

import numpy as np
import torch

from nig_timenet_torch.doctor.common import Check
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.layers import TransformerBlock
from nig_timenet_torch.stats.metrics import pearson, spearman

__all__ = ["block_use", "transformer_blocks", "switched_off"]

_DEAD = 0.01      # a block whose removal moves every mu by less than this is not used by the outputs


def transformer_blocks(model) -> dict[str, TransformerBlock]:
    """{module name: block} of every TransformerBlock, in forward (registration) order."""
    return {n: m for n, m in model.named_modules() if isinstance(m, TransformerBlock)}


@contextmanager
def switched_off(block: TransformerBlock, parts):
    """The named sublayers ("attn", "ffn") of ``block`` return zeros while the context is open."""
    hooks = [getattr(block, p).register_forward_hook(lambda m, i, o: torch.zeros_like(o))
             for p in parts if getattr(block, p) is not None]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def _mus(model, x, targets, chunk: int = 2048) -> dict[str, np.ndarray]:
    keys = {t: head_spec("nig_regression", t).output_keys["mu"] for t in targets}
    dev = next(model.parameters()).device
    parts = [model(x[i:i + chunk].to(dev)) for i in range(0, len(x), chunk)]
    return {t: torch.cat([p[k].float().reshape(-1) for p in parts]).cpu().numpy() for t, k in keys.items()}


@torch.no_grad()
def _sizes(model, x, blocks, chunk: int = 2048) -> dict[str, dict]:
    """Per block: update ratio of attn / ffn and the attention entropy share, on the first chunk of rows."""
    seen: dict[str, dict] = {n: {} for n in blocks}

    def rms(t):
        return float(t.detach().float().square().mean().sqrt())

    hooks = []
    for n, b in blocks.items():
        hooks.append(b.register_forward_hook(lambda m, i, o, n=n: seen[n].update(stream=rms(i[0]))))
        hooks.append(b.attn.register_forward_hook(lambda m, i, o, n=n: seen[n].update(attn=rms(o))))
        if b.ffn is not None:
            hooks.append(b.ffn.register_forward_hook(lambda m, i, o, n=n: seen[n].update(ffn=rms(o))))
    try:
        model(x[:chunk].to(next(model.parameters()).device))
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for n, b in blocks.items():
        w = b.attn.last_attn_weights.float().clamp_min(1e-12)                 # (B, H, T, T)
        ent = float(-(w * w.log()).sum(-1).mean() / np.log(max(w.shape[-1], 2)))
        s = seen[n]["stream"] + 1e-12
        out[n] = {"attn": seen[n]["attn"] / s, "ffn": seen[n].get("ffn", float("nan")) / s, "attn_entropy": ent}
    return out


def block_use(model, x: torch.Tensor, y: dict[str, np.ndarray], targets) -> tuple[list[Check], list[dict]]:
    """``y``: {target: label array} aligned with ``x`` (for the rank IC). Returns no check for a model without
    transformer blocks."""
    blocks = transformer_blocks(model)
    if not blocks:
        return [], []
    was = model.training
    model.eval()
    try:
        base = _mus(model, x, targets)
        base_ic = {t: float(spearman(y[t], base[t])) for t in targets}
        sizes = _sizes(model, x, blocks)
        rows, checks = [], []
        cases = [(name, block, part, off) for name, block in blocks.items()
                 for part, off in (("attn", ("attn",)), ("ffn", ("ffn",)), ("block", ("attn", "ffn")))
                 if not (part == "ffn" and block.ffn is None)]
        for name, block, part, off in [*cases, ("all blocks", None, "block", ("attn", "ffn"))]:
            with ExitStack() as stack:                       # the last case switches every block off at once
                for b in ([block] if block is not None else blocks.values()):
                    stack.enter_context(switched_off(b, off))
                cut = _mus(model, x, targets)
            change = {t: 1.0 - abs(float(np.nan_to_num(pearson(base[t], cut[t])))) for t in targets}
            d_ic = {t: float(spearman(y[t], cut[t])) - base_ic[t] for t in targets}
            size = sizes.get(name, {"attn_entropy": float("nan")})
            row = {"block": name, "part": part, "update_ratio": size.get(part, float("nan")),
                   "attn_entropy": size["attn_entropy"], "mu_change": max(change.values()),
                   "d_ic": min(d_ic.values()), "mu_change_by_head": change, "d_ic_by_head": d_ic}
            rows.append(row)
            dead = part == "block" and row["mu_change"] < _DEAD
            checks.append(Check(
                "A", f"model/{name}", f"{part} use: 1 - |corr(mu, mu with it switched off)|; worst label IC change",
                f"{row['mu_change']:.3f}; {row['d_ic']:+.3f}",
                f"update / stream {row['update_ratio']:.2f}, attention entropy {row['attn_entropy']:.2f} of uniform"
                if part != "block" else "0 = the outputs ignore the block", "warn" if dead else "info",
                "the outputs do not use this block: the information goes around it (linear_path, stats) or the "
                "block is not trained (card B: gradient share, update ratio); fewer blocks cost nothing"
                if dead else "", {k: v for k, v in row.items() if k.endswith("_by_head")}))
        return checks, rows
    finally:
        model.train(was)
