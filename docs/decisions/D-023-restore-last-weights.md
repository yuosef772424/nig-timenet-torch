# D-023: `workflow.run.restore(weights=...)`: read the last finished epoch instead of the best one

Date: 2026-11-10
Owner decision: owner message of 2026-11-10 (Arabic, translated): "I want the full settings with the ability to load the
weights of the last version, not the best version".
Why: `restore` only loaded `best.pt`. The last-epoch weights existed (`checkpoint.pt`) but no step could read them, and
flipping `train.early_stop.restore_best_weights` on a finished run folder changes the RunIdentity (it is part of the fit
settings), so resuming such a folder with the flag flipped is refused.

## What changes behaviour
- Nothing at the default: `restore(run)` still loads the best-epoch weights (`weights="best"`).
- New: `restore(run, weights="last")` loads the weights after the last finished epoch (`checkpoint.pt` `model`; the
  EMA copy when the run trained with `ema_decay`, which is what `fit` leaves in the model without
  `restore_best_weights`). `FitResult.restored_best` is False then. An unknown name raises `ValueError`.
- For a NEW run the existing switch `train.early_stop.restore_best_weights = False` keeps the last epoch's weights in
  the model for every report step (tests/workflow/test_run.py); `best.pt` is still written.

## New defaults
- `nig_timenet_torch.workflow.run.restore(weights)`: `<new>` -> `'best'`
