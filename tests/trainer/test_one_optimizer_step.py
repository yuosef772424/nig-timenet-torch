"""P19: one real optimizer step through compute_losses.

Post-update weights are not and must not be compared to the TF oracle (BUILD_PLAN.md section 0 tolerance table:
the optimizer step is "grads only"). Do not add that comparison or reimplement TF's AdamW to chase a bit-match.
Hyperparameters lr=5e-5, weight_decay=1e-4 = TRAINER_DEFAULT_CONFIG["optimizer"] on the TF side, cited for
traceability only.
"""

import torch

from tests.oracle import oracle_loader as ol
from tests.trainer.test_losses_end_to_end import PRIMARY, grads_vs_oracle, named_trainable, run


def test_optimizer_step_runs_and_grads_match_oracle():
    trainer, _, _ = run(PRIMARY)
    named = named_trainable(trainer)
    optimizer = torch.optim.AdamW(list(named.values()), lr=5e-5, weight_decay=1e-4)
    optimizer.zero_grad()
    total, *_ = trainer.compute_losses(ol.load_inputs(), ol.load_labels())
    total.backward()
    grads_vs_oracle(trainer, {n: p.grad for n, p in named.items()})  # the grads the step actually uses
    old = {n: p.detach().clone() for n, p in named.items()}
    optimizer.step()
    for n, p in named.items():
        assert torch.isfinite(p).all(), n
        assert not torch.equal(old[n], p.detach()), f"{n} did not change"

