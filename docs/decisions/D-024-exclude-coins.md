# D-024: `split.exclude_coins`: drop named coins from train, validation and test

Date: 2026-11-10
Owner decision: owner message of 2026-11-10 (Arabic, translated): "can you create code that filters the coins on the
ready data and removes the chosen coins from the training and test data".
Why: no way to leave a coin out of a run without re-exporting the artifact (its hashes and the D1 coin list are checked).

## What changes behaviour
- Nothing at the default (`()`: every coin kept, row indices byte-equal to before).
- With names: the rows of those coins leave the train, val AND test row lists. The cut is applied AFTER the time split,
  so the split dates and the purge do not move; the smoke stride (`max_rows`) is applied after the removal. The
  artifact on disk, its manifest and its data fingerprint are untouched.
- Names are matched like the artifact's coin list: upper case, with or without the `USDT` suffix. A name that is not a
  coin of the artifact raises (with a did-you-mean); removing every coin raises.
- `prepare` prints the excluded names.

## Changed default reprs (new field at its default; same behaviour)
- `nig_timenet_torch.workflow.settings.RunSettings(split)`: `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None)` -> `SplitSettings(train_end='', val_end='', train_share=0.8, val_share=0.1, embargo_hours=0.0, max_rows=None, exclude_coins=())`

## New defaults
- `nig_timenet_torch.workflow.settings.SplitSettings(exclude_coins)`: `<new>` -> `()`
