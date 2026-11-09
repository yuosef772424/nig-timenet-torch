# Gate fixes — why A5 / A6 / the beta floor fail on E1, the switches, and the new run default (FX2)

Recorded 2026-11-09. Exploratory (rule C5): CPU only, synthetic E1 (D-018 generator, the same 64,000-window env as
FX1), batch 512, test split, 3 epochs (300 steps), 3 seeds per arm. Builds on `docs/results/e1-fixes-001.md` (FX1;
the "bundle" = `bn_warmup` + `order_detach_close`). Every row: `docs/results/gates-fixes-001-arms.csv`. Tools:
`workflow/e1_arms.py` (now also records A5 / A6 start / end, `a6_label_probe`, class AUCs), `trainer.grad_report.a5_a6`.
Numbers are never edited; a correction goes in a new file that names this one.

Owner decision (2026-11-09, during the step): the switches the evidence shows useful are the run default (D-023):
`bn_warmup`, `order_detach_close`, `nig_scale_fit: train_std`, `nig_trunk_grad: 0.3`
(`workflow.settings.RUN_DEFAULT_SWITCHES`). The bundle reproduced FX1 exactly (slope 1.06 / 0.92, close corr +0.65,
7.3 fails), so this commit's numbers compare with FX1's.

## 1. Gate A5 (B-060): class heads < 0.02 of the trunk gradient

Per-loss gradient norms on the trunk (bundle, seed 0, `diag`): step 0 class BCE 0.16-0.25 (Kendall factor 0.5
included) vs NIG NLL 1.6-2.8, of which the mu path 0.87-2.6. The NIG mu gradient at step 0 comes from the ordered
offsets: softplus(0) = 0.69 with labels of sd ~0.02 gives mean errors -0.82 (high) / +0.60 (low), ~35 label sd.
End of run: class shares per head (bundle, 3 seeds) high 0.025-0.12, low 0.04-0.19, **close 0.003-0.010**: the
starved head at the end is close_class only.

Reference (rule C1): the oracle-normal class probability (true mean / sd of E1) scores AUC **0.778 / 0.821 / 0.519**
(high / low / close; base rates 0.62 / 0.40 / 0.50). The trained class heads score 0.766-0.769 / 0.806-0.811 /
0.512-0.515 in every arm that was measured, starved or not. So close_class has almost no learnable signal (planted
close IC 0.05): at convergence its gradient is ~0 by construction, and the high / low class heads are already near
the oracle while "starved" (they read the regression-shaped trunk).

**Cause (verified):** start = offset init error + Kendall's 0.5; end = a head without learnable signal (an A5
definition issue: it assumes every head has one). Arms (bundle + one change; 3 seeds; min over seeds):

| arm | A5 min start / end | high slope / corr | low slope / corr | close corr | sd scale h / l | fails | verdict |
|---|---|---|---|---|---|---|---|
| bundle | 0.015 / 0.003 | 1.06 / 0.963 | 0.92 / 0.966 | +0.65 | 1.26 / 1.26 | 7.3 | reference |
| loss_balance equal_grad_norm (Kendall off) | 0.032 / **0.107** | 1.19 / **0.744** | 1.13 / **0.837** | +0.49 | 2.33 / 2.31 | 17.0 | rejected: A5 passes, regression breaks |
| uncertainty_weighting off | 0.028 / 0.006 | 1.02 / 0.964 | 0.95 / 0.966 | +0.72 | 1.26 / 1.26 | 6.3 | start fixed only |
| kendall_init_logvar -2.3 | 0.080 / 0.018 | 1.07 / 0.964 | 0.99 / 0.964 | +0.73 | 1.26 / 1.26 | 5.3 | end fails on 1 seed |
| nig_trunk_grad 0.1 | 0.080 / **0.032** | 1.01 / 0.957 | 1.01 / 0.960 | +0.71 | 1.26 / 1.26 | 5.3 | A5 passes, corr -0.006 |
| class_head_hidden 128 | 0.014 / 0.004 | 0.99 / 0.966 | 0.88 / 0.966 | +0.57 | 1.26 / 1.27 | 7.7 | rejected |
| label_smoothing 0 | 0.016 / 0.005 | 1.01 / 0.964 | 0.96 / 0.966 | +0.66 | 1.26 / 1.26 | 7.0 | rejected |

`kendall_init_logvar -2.3` and `nig_trunk_grad 0.1` give identical start shares (both scale the class / NIG ratio by
10; a check of the measurement). Forcing equal norms injects the close_class noise into the trunk: rejected.

## 2. Gate A6: high | low cosine -0.75 to -0.88

`e1_arms.a6_label_probe`: the same cosine with labels shifted per batch so each regression head's batch-mean error is
0 ("centred"), and with the oracle mean as the label.

| run | raw | centred | oracle | oracle centred | mean error high / low / close |
|---|---|---|---|---|---|
| baseline, step 0, seed 0 | -0.75 | +0.35 | -0.75 | +0.35 | -0.82 / +0.60 / -0.15 |
| bundle, step 0, seed 1 | -0.28 | +0.08 | -0.28 | +0.08 | -0.63 / +0.79 / +0.08 |
| bundle, end, seed 0 | +0.20 | +0.34 | +0.90 | +0.96 | +0.003 / -0.003 / -0.0005 |
| bundle, end, seed 1 | -0.06 | +0.59 | +0.90 | +0.95 | +0.003 / -0.004 / +0.001 |
| bundle, end, seed 2 | **-0.80** | **+0.49** | -0.64 | **+0.94** | -0.006 / -0.014 / -0.011 |

**Verdict: artefact of the A6 definition (verified).** The negative cosine is the heads' common mean error mapped
through the mirrored high / low readouts (in the baseline also through the ordered means into mu_close, which
`order_detach_close` removes: step-0 cosine -0.75 / -0.87 -> -0.05 / -0.28). The heads' own biases absorb it; with
the error centred the heads agree (+0.35 to +0.59; +0.94-0.96 on noise-free labels). No projection (PCGrad) switch
was added. Protocol change proposed in `docs/suggestions.csv` (judge A6 on centred errors); threshold unchanged.
With `nig_scale_fit` the offsets start at s x 0.69 (~label scale), so step-0 A6 is milder (min -0.16).

## 3. Beta floor (B-052): `nig_scale_fit: train_std`

A registered learned-state component (`model/label_scale.py`, rule B4): fit on the TRAINING labels only (s = mean
std of the NIG labels), stored as the heads' `label_scale` buffers (state_dict / checkpoint), read by every forward;
beta floor = beta_min x s^2 (proportional to the label variance).

| arm (3 seeds) | A5 min s / e | high slope / corr | low slope / corr | close corr | sd scale h / l / c | coverage h / l / c | T-sharp h / l | fails |
|---|---|---|---|---|---|---|---|---|
| bundle + nig_scale_fit | 0.038 / 0.015 | 1.27 / 0.966 | 1.25 / 0.966 | +0.79 | 1.04 / 1.04 / 1.03 | 0.87 / 0.86 / 0.79 | 0.77 / 0.89 | 3.3 |
| + nig_trunk_grad 0.1 | 0.108 / 0.059 | 1.25 / 0.963 | 1.30 / 0.961 | +0.75 | 1.04 / 1.04 / 1.03 | 0.86 / 0.86 / 0.79 | 0.67 / 0.87 | 3.0 |
| **+ nig_trunk_grad 0.3 (= new default)** | 0.077 / **0.035** | 1.27 / 0.966 | 1.27 / 0.965 | **+0.79** | **1.04 / 1.04 / 1.03** | 0.86 / 0.86 / 0.79 | 0.68 / 0.93 | **2.7** |
| + uncertainty_weighting off | 0.060 / 0.021 | 1.27 / 0.966 | 1.25 / 0.965 | +0.78 | 1.04 / 1.04 / 1.03 | 0.87 / 0.86 / 0.79 | 0.75 / 0.90 | 2.3 |

New default vs bundle, seed-paired (seeds 0 / 1 / 2): close corr +0.08 / +0.22 / +0.11, blocking fails -2 / -3 / -9,
A5 end +0.025 / +0.031 / +0.046, |sd scale - 1| high -0.20 / -0.24 / -0.22 and low -0.22 / -0.21 / -0.22, high / low
corr within +-0.01, T-sharp low +0.68 / +0.64 / +0.96, T-sharp high +0.66 / +0.51 / -0.59; **cost: |slope - 1| low
+0.18 / +0.19 / +0.17, high +0.29 / +0.07 / -0.04** (slope 1.27 vs 1.06 / 0.92, the same residual as FX1's
`nig_scale 0.01`, unexplained). Remaining E1 fails of the new default: A6 (start / end, 2 of 3 seeds; does not block
because A6 blocks only on every seed) and T-sharp high (3 seeds) / low (2 seeds).

## 4. Gates, 10 epochs

- Gate run (`run_gates`, E1 + E2, 3 seeds, no e3_source; **E3 NOT run**: no real artifact here) of the new default:
  see section 6 (it was still running when the owner stopped the step; result recorded there only if it finished).
- The 3- and 10-epoch confirmation round against the baseline (C4) was **not run**: the owner stopped new work
  (usage limit). The baseline 3-epoch comparison is FX1's (same code path at the defaults, reproduced by the bundle).

## 5. Verdicts

| hypothesis | verdict |
|---|---|
| A5 start: offset init error + Kendall 0.5 | verified (Kendall off / nig_scale_fit raise the start share; per-loss norms) |
| A5 end: close_class has no learnable signal | verified (oracle AUC 0.519; share 0.003-0.015; class AUC near oracle in every arm) |
| equal-gradient-norm balancing | rejected (A5 passes, high / low corr -0.22 / -0.13, sd scale 2.3) |
| Kendall start value, class head width, label smoothing 0 | no robust A5 gain |
| nig_trunk_grad 0.3 | A5 passes on 3 of 3 seeds with corr unchanged (adopted) |
| A6 = task conflict | rejected: mean-error artefact (centred +0.35 to +0.59, oracle-centred +0.95) |
| beta floor at label scale 1 (B-052) | fixed on E1 by nig_scale_fit (sd scale 1.04, coverage in band); slope cost 1.27 |
| B-062 BN warm-up on real data | absent (checkpoint moving variances are data-scale; parent session check) |

## 6. Not verified

- No GPU, no real data, no E3; 3-epoch runs only; the 10-epoch and baseline-paired confirmation round was not run.
- The E2 gate result of the new default (the FX1 `nig_scale 0.01` alarm on E2 gate B) — see the PR description for
  the outcome if the run finished before the push.
- The residual high / low slope 1.27 with nig_scale_fit (FX1 cause 3, dropout dilution, is the main suspect).
- Real-data effect of the new defaults: earlier checkpoints were trained without them.
