# NIG label-scale default (MG2, B-071)

Clash: #51 `nig_scale_fit="train_std"` (`model/label_scale.py`, `label_scale` buffer) vs #56 `nig_scale="train_std"`
(`model/scalers.py`, `FittedScale`); exclusive, the merged default carried #51's. Exploratory (rule C5), CPU.

Arms (E1 synthetic env, 3 epochs, batch 512, seeds 0, 1, 2, all on commit 3083d1c, `workflow/e1_arms.py`, default
preset otherwise identical): (a) default = `nig_scale_fit` train_std; (b) `nig_scale` train_std + `nig_scale_fit` fixed;
(c) both fixed (control).

| Arm (mean of 3 seeds) | blocking gate fails (per seed) | best val_raw_loss | T-sharp high / low | close corr with oracle | slope high / low |
|---|---|---|---|---|---|
| (a) nig_scale_fit | 2.0 (2, 2, 2) | -5.6294 | 0.67 / 0.92 | 0.766 | 1.29 |
| (b) nig_scale | 2.3 (2, 3, 2) | -5.6294 | 0.69 / 0.91 | 0.764 | 1.30 |
| (c) fixed (control) | 6.0 (3, 6, 9) | -5.286 | 0.43 / 0.18 | 0.632 | 1.02 |

A5 / A6 end values of (a) and (b) agree to the second decimal on every seed.

Decision: a tie between (a) and (b) (per-seed val loss within 0.0005, T-sharp within 0.02, one gate check differs on
one seed); both beat the control by a wide margin (it fails 6 of the gates, a fitted scale is worth keeping). Keep the
simpler change: the default stays (a), no pin key moves. Both code paths stay: removing one changes state_dict keys,
the attention head and #56's tests, and the evidence does not ask for it.

Files changed: `notebooks/main.ipynb` (`FX1_BUNDLE` is None / False / True, False now turns the two switches off),
`tests/notebooks/test_main_notebook.py`, `docs/decisions/D-026-*.md`, `docs/bottlenecks.csv` (B-071 closed),
`docs/friction_log.csv`, `docs/STATUS.md`, comments in `workflow/settings.py` and NOTES N-02b.

Not measured: 10 epochs, real data, GPU. Revert: set `nig_scale: "train_std"` and `nig_scale_fit: "fixed"` in
`RUN_DEFAULT_SWITCHES` / `MODEL_FIRST_FIXES` (workflow/settings.py) and regenerate the pins.
