# INFLUENCE — the effect of each training batch and of each input feature (owner task 2026-11-17)

Status (2026-11-22): done: the code, the known-answer tests and the first look on E1 (IF1, merged #63 via MG3 #69); the `batch_by` walk-forward comparison against `rows` was run in `docs/results/batch-by-wf-001.md` and `batch-by-wf-002.md`. Open, no ROADMAP step and no results note: influence diagnostics on the real artifact on the GPU (smoke job first, rule E5) and the drop-column confirmation of the top features (end of section 4).

Owner's request (Arabic, 2026-11-17): see the effect of each batch of data on the model during training (find the
best methods, pick the best and most efficient), then measure the effect of each feature on the model, with results
that are reliable when tested. This page records the candidates, the choice and why, how to run it, and what the
known-answer tests and a first look on E1 show. Code: `trainer/batch_influence.py`, `eval/influence_report.py`,
`doctor/feature_importance.py`, `doctor/influence.py` (entry).

## 1. Effect of each batch during training

| Candidate | What it measures | Cost | Reliability for us | Verdict |
|---|---|---|---|---|
| Leave-one-batch-out retraining / Data Shapley | the true counterfactual | one retraining per batch (or thousands for Shapley) | exact but impossible at 2.5 M rows | no |
| Influence functions (Koh & Liang 2017) | effect of up-weighting a sample at the END of training, via an inverse Hessian | Hessian-vector products, many iterations | known to be fragile for deep nets (non-convex, Basu et al. 2021); answers "the final model", not "during training" | no |
| TracIn-CP (Pruthi et al. 2020) | sum over saved checkpoints of lr x g_train . g_test | a few checkpoints x per-sample gradients | depends heavily on which checkpoints are kept; uses the raw gradient, not the Adam step | no |
| First-order In-Run Data Shapley = TracIn-Ideal (Wang et al., ICLR 2025) | at EVERY step, g_val . (theta_{t+1} - theta_t) | one validation gradient per step | the right object; first order only | kept, as the decomposition (every k steps) |
| **Exact one-step change of a fixed probe loss** | L_probe(theta_{t+1}) - L_probe(theta_t), every step | ONE probe forward per step (the value before step t+1 is the value after step t) | exact for the probe; includes curvature, Adam and BatchNorm statistics; checked by split halves | **chosen as the headline** |

Choice: the exact probe delta after every step (cheapest and exact), with the first-order term on every k-th step
to split the effect by module and to give the gradient agreement between the batch and the probe. Using the REAL
update (theta_{t+1} - theta_t) instead of lr x gradient makes the first-order term correct for AdamW (the 2026 note
"In-Run Data Shapley for Adam" points at exactly this gap in the SGD form).

How it is made reliable:

- **Training is unchanged**: the monitor uses no torch RNG and no optimizer state; a fit with and without it gives
  bitwise the same weights and history on CPU (test).
- **One definition of the probe loss for the whole run**: the trainer's raw loss with the loss schedules frozen at
  their start values; the step deltas then telescope exactly to the total change (test).
- **Split-half reliability**: the probe is two random halves; the correlation of their step deltas, after removing
  the rolling median (the drift every step shares early in training), with Spearman-Brown, says how much of a step's
  delta is the batch's own effect and how much is probe noise. Below 0.5: raise `probe_rows`.
- **First-order check**: r(first-order, exact) is reported; a low value means curvature dominates (large steps).
- **Groups (coins, months)**: the share-weighted mean EXCESS delta of the batches that held the group, against
  shuffles of the excess deltas within each epoch, Bonferroni over groups. With random batches the shares vary
  little, so this is conservative: "not flagged" means "not shown", not "harmless".
- **A second probe on training rows** shows memorisation steps (helped the training rows, hurt validation).
- **Group-level TracIn** (`group_every`, IF1 suggestion applied): every k steps the gradient of each row group (a coin
  or a month of training rows, `group_rows` rows each) is compared with the probe gradient; the mean of
  g_probe . g_group over the sampled steps (> 0 = training on that group lowers the validation loss) with its t and
  Bonferroni. It evaluates EVERY group at every sampled step, so it has far more power than the batch-composition
  attribution: on the planted test |t| 10-12 for the bad group against a composition z of ~4.4 (seeds 11, 22, 33),
  the bad group the only one flagged each time.

Known-answer tests (`tests/trainer/test_batch_influence.py`): on a linear model, label-shuffled batches raise the
validation loss and clean ones lower it (>= 80 % each); one coin of 16 with sign-flipped labels is the ONLY group
flagged (harmful; z 4.2-9.3 on seeds 0, 11, 22, 33); with nothing planted no group is flagged; first-order vs exact
r > 0.95.

Found while building it: on the NIG heads of the tiny model the per-step delta is dominated by the NIG
uncertainty-parameter layer (`fo_nig` carries ~90 % of the first-order variance), and an outlier label barely moves
the evidential loss (Student-t NLL has a bounded gradient for large residuals). Both are properties of the heads, not
of the monitor; read the per-head and per-module columns before the total.

## 1b. Training switch `batch_by` (owner 2026-11-17, D-034)

`train.batch_by` says what one batch is: `rows` (default, unchanged) = shuffled rows in `batch_size` chunks;
`timestamp` = all training rows of one timestamp form one batch; `day` = all rows of one UTC day. The ORDER of the
batches is shuffled every epoch (the same seeded RNG as the row shuffle), never the rows inside a batch; `batch_size`
is not used. It is for the single-coin (flat) model; a cross-section run (`split.cross_section`) already batches whole
timestamps and staged training has its own loop, so both refuse it. A non-default value changes the run identity
(a new run); the default keeps it. The model never sees the group id. Notebook: `"train.batch_by"` in `CHANGES`.
On E1 (16 coins): `timestamp` = 799 batches of 16 rows per epoch, `day` = 267 batches. With the monitor on, the batch
report then reads as "which DATES help or hurt", one step per date. First walk-forward look (20 real coins, CPU):
`docs/results/batch-by-wf-001.md` (no arm better on the test loss; timestamp batches raise the low-head rank IC in
7 / 8 pairs, a hypothesis for a larger study). On 60 coins (`docs/results/batch-by-wf-002.md`) that gain did not
repeat: timestamp batches only recover what small batches lose (vs a 34-row control +0.035), equal the default on the
low IC and have a worse test loss in 12 / 12 pairs. `rows` stays the default.

## 2. Effect of each input feature

| Candidate | Reliability problem | Verdict |
|---|---|---|
| Gradient saliency / gradient x input | fails the model-randomisation sanity checks (Adebayo et al. 2018); local, not "what the model loses" | no |
| Attention weights | not an explanation of the output | no |
| SHAP (Kernel / Deep) | costly for 32 x 155 inputs; assumes independent features; approximation on a deep net | no |
| Integrated gradients | baseline choice decides the answer; local | no |
| Drop-column retraining (LOCO) | the gold standard, one retraining x seeds per feature | only to CONFIRM the top few (next step) |
| **Grouped permutation importance** on held-out rows | correlated features share credit (fixed by permuting correlation clusters together) | **chosen** |

The method (`doctor/feature_importance.py`): for each group of input channels (a feature, a cluster of features with
|Spearman| >= 0.9 on the last bar, a block of bars across all features, or a named family) the same random row
permutation replaces those channels over the whole window; the model predicts again on VAL rows and the trainer's own
raw loss (also per head) and the rank IC per regression head are compared with the unpermuted run on the SAME rows.
Rows are cut into 40 time blocks (whole timestamps); per-block differences averaged over the repeats give a
block-bootstrap CI at 1 - 0.05 / n_groups. Verdict: `uses` (CI above 0), `hurts` (below 0: the model does better
without it), `none`. Split-half: Spearman of the importances on even vs odd blocks. Mode `within_time` permutes only
across coins of one timestamp: it measures the COIN-SPECIFIC use (a market-wide feature scores 0 there).

Known-answer tests (`tests/doctor/test_feature_importance.py`; signal f0 with a near-copy f1, market-wide f2, last
bars only, f3's link reversed between training and validation, f4 / f5 noise): `uses` is exactly {f0, f1, f2,
f0+f1, last bars}, f3 is `hurts`, noise and early bars `none`; the pair f0+f1 scores > 2 x either alone; f2 scores
exactly 0 within time; the same answer on seeds 0, 11, 22, 33 (split-half 0.89-0.93).

What it does NOT tell: importance is what THIS model loses, not what the data hold (an ignored predictive feature
scores 0); a feature with NO link to the label on these rows scores ~0 even if the model relies on it (only a
REVERSED link scores below 0).

## 3. How to run

```
python -m nig_timenet_torch.doctor.influence <artifact | e1> --out influence_out --epochs 3 --max-rows 60000 \
    --probe-rows 1024 --first-order-every 10 --group-by month --group-every 20 --within-time
python -m nig_timenet_torch.doctor.influence <artifact> --train '{"batch_by": "day"}' --out by_day   # one batch per day
python -m nig_timenet_torch.doctor.influence <artifact> --run-dir <finished run> --trained --out fi_out   # importance only
```

In a notebook or job: `run.train(run, batch_monitor=doctor.influence.monitor_for(run, InfluenceSettings(...)))`, then
`doctor.influence.batch_reports(out, run, monitor)` and `importance_reports(out, run, ImportanceSettings(...))`.
Cost: one probe forward per step per probe (default 2 probes of 1024 rows) plus a backward every 10 steps; the
importance is groups x repeats forward passes over the VAL rows.

## 4. First look on E1 (CPU, 16 coins x 2000 bars, 3 epochs, seed 0; a look, not a result)

- Batch influence: 150 steps, probe loss -4.80 -> -5.68; harmful steps 2 % in epoch 1, ~40 % in epochs 2-3 (the
  model reaches its floor after one epoch); split-half reliability 0.95; first-order vs exact r 0.99; no coin and no
  month flagged (nothing was planted).
- Feature importance (VAL, 1328 rows): the last 8 bars carry almost all of it (delta 0.91), bars older than 16 are
  `none`; by channel close 0.55 > low 0.27 > high 0.10, all `uses`; split-half 0.89. This matches E1's construction
  (the close drift comes from the last closes, the volatility from the 1 / 6 / 32-bar ranges).

Not yet done: the real artifact on the GPU (the smoke job first, rule E5), drop-column confirmation of the top
features, a walk-forward comparison of `batch_by` arms against `rows` (2+ seeds, rule C4).
