# Health report — /content/drive/MyDrive/nig_timenet_runs/baseline-001-s0-r2/train

Rows: 100000

## Targets (validation)

| head | rank IC (ts) | per-coin rho | AUC (const / shuffled) | T-conf rho | gap CI95 | conf ok | std<move share | coverage80 | sharp ok |
|---|---|---|---|---|---|---|---|---|---|
| high | 0.400 | 0.225 | 0.750 (0.50 / 0.498) info only (direction suspended) | -0.527 | [-0.008, 0.072] | False | 0.878 | 0.895 | False |
| low | 0.433 | 0.260 | 0.763 (0.50 / 0.497) info only (direction suspended) | -1.000 | [0.621, 0.691] | True | 0.920 | 0.925 | False |
| close | 0.063 | 0.073 | 0.524 (0.50 / 0.496) | -0.988 | [0.495, 0.575] | True | 0.000 | 0.862 | False (informational, O-1) |

Baselines: constant confidence scores 0; a zero std scores 1.0 on std<move share (read it with coverage); constant AUC 0.5.

## NIG floor-hit shares

| head | nu | alpha | beta |
|---|---|---|---|
| high | 0.00 | 0.00 | 1.00 |
| low | 0.00 | 0.00 | 1.00 |
| close | 0.00 | 0.00 | 1.00 |

## Layer probe — y_close_reg

| point | metric | mean | raw | delta vs raw | features |
|---|---|---|---|---|---|
| raw_input | r2 | -0.166 | -0.166 | 0.000 | 96 |
| trunk_norm | r2 | -0.055 | -0.166 | 0.110 | 64 |

## Layer probe — y_close_class

| point | metric | mean | raw | delta vs raw | features |
|---|---|---|---|---|---|
| raw_input | auc | 0.540 | 0.540 | 0.000 | 96 |
| trunk_norm | auc | 0.550 | 0.540 | 0.010 | 64 |

## Trainer trace

Epochs: 5; best: {'best': -5.062315464019775, 'best_epoch': 1, 'metric': 'val_raw_loss'}; trace columns: yes

- train_total_loss: -6.1154
- train_raw_loss: -5.0356
- train_high_reg_loss: -2.3956
- train_high_class_loss: 0.5906
- train_low_reg_loss: -2.4077
- train_low_class_loss: 0.5881
- train_close_reg_loss: -2.0994
- train_close_class_loss: 0.6782
- train_high_reg_calib_loss: 0.0116
- train_low_reg_calib_loss: 0.0106
- train_close_reg_calib_loss: 0.0132
- val_total_loss: -6.0344
- val_raw_loss: -4.9467
- val_high_reg_loss: -2.3111
- val_high_class_loss: 0.6081
- val_low_reg_loss: -2.4378
- val_low_class_loss: 0.5926
- val_close_reg_loss: -2.1046
- val_close_class_loss: 0.6956
- val_high_reg_calib_loss: 0.0106
- val_low_reg_calib_loss: 0.0123
- val_close_reg_calib_loss: 0.0126
- val_selection_metric: -4.9467
- trace_sched_lambda_reg: 0.0400
- trace_sched_lambda_calib: 0.0905
- trace_kendall_weight_high_class: 0.3454
- trace_kendall_weight_low_class: 0.3500
- trace_kendall_weight_close_class: 0.3046
- trace_kendall_logvar_high_class: -0.5202
- trace_kendall_logvar_low_class: -0.5333
- trace_kendall_logvar_close_class: -0.3942
- trace_gradshare_high_reg: 0.2853
- trace_gradshare_high_class: 0.0075
- trace_gradshare_low_reg: 0.3899
- trace_gradshare_low_class: 0.0093
- trace_gradshare_close_reg: 0.3024
- trace_gradshare_close_class: 0.0057
- trace_epoch_seconds: 175.5945
