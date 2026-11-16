"""PURPOSE: The direction-balance loss term (D-032, off by default; owner idea 2026-11-16): penalise a batch whose
    predictions lean long or short, so the model's direction stays balanced instead of carrying a level.
TAGS: direction balance, long short balance, direction bias, level bias, auxiliary loss, unbiased direction, close head
PITFALLS: only for TWO-SIDED targets (the close heads; a high mean is positive by nature): the caller names them
    (``direction_balance_targets``). The term pushes the batch's NET direction to zero, not each row: on shuffled
    batches it removes a global lean, on whole-timestamp batches it also removes the lean of a timestamp (a soft form
    of the model option ``cross_center``); it cannot remove a lean per COIN (a batch does not know its coins). A batch
    whose labels really lean (a falling market) is pushed to neutral too: the term says "no market bet".

Per named target: a NIG mean -> (mean over the batch of tanh(mu / rms(mu)))^2, the squared soft share of long minus
short rows (the scale is detached, so the term cannot be met by shrinking every mean); a class probability ->
(2 mean(p) - 1)^2. The loss term is the mean over the named targets.
"""

import torch

__all__ = ["direction_balance_loss"]


def direction_balance_loss(outputs: dict, targets_cfg: dict, names) -> torch.Tensor:
    """The balance term of the RESOLVED ``targets_cfg`` entries ``names`` (each needs a ``mu`` or a ``logits``
    output key)."""
    terms = []
    for name in names:
        keys = targets_cfg[name]["output_keys"]
        if "mu" in keys:
            mu = outputs[keys["mu"]].reshape(-1).to(torch.float32)
            scale = mu.detach().square().mean().sqrt() + 1e-12
            terms.append(torch.tanh(mu / scale).mean().square())
        else:
            p = outputs[keys["logits"]].reshape(-1).to(torch.float32)
            terms.append((2.0 * p.mean() - 1.0).square())
    if not terms:
        raise ValueError("direction_balance needs at least one target (direction_balance_targets)")
    return torch.stack(terms).mean()
