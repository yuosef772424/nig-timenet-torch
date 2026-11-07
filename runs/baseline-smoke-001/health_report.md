# Health report — /content/drive/MyDrive/nig_timenet_runs/baseline-smoke-001/train

Rows: 20000

## Targets (validation)

| head | rank IC (ts) | per-coin rho | AUC (const / shuffled) | T-conf rho | gap CI95 | conf ok | std<move share | coverage80 | sharp ok |
|---|---|---|---|---|---|---|---|---|---|
| high | -0.046 | -0.039 | 0.627 (0.50 / 0.506) info only (direction suspended) | 0.758 | [-0.229, -0.115] | False | 0.010 | 0.958 | False |
| low | 0.052 | -0.001 | 0.637 (0.50 / 0.499) info only (direction suspended) | -0.915 | [0.033, 0.118] | True | 0.124 | 0.991 | False |
| close | 0.020 | -0.003 | 0.498 (0.50 / 0.505) | -0.261 | [0.061, 0.171] | False | 0.000 | 0.997 | False (informational, O-1) |

Baselines: constant confidence scores 0; a zero std scores 1.0 on std<move share (read it with coverage); constant AUC 0.5.

## NIG floor-hit shares

| head | nu | alpha | beta |
|---|---|---|---|
| high | 0.00 | 0.00 | 0.00 |
| low | 0.00 | 0.00 | 0.00 |
| close | 0.00 | 0.00 | 0.00 |

## Layer probe — y_close_reg

| point | metric | mean | raw | delta vs raw | features |
|---|---|---|---|---|---|
| raw_input | r2 | -0.193 | -0.193 | 0.000 | 96 |
| trunk_norm | r2 | -0.044 | -0.193 | 0.149 | 64 |

## Layer probe — y_close_class

| point | metric | mean | raw | delta vs raw | features |
|---|---|---|---|---|---|
| raw_input | auc | 0.510 | 0.510 | 0.000 | 96 |
| trunk_norm | auc | 0.506 | 0.510 | -0.004 | 64 |

## Trainer trace

Epochs: 1; best: {'best': -1.4734371900558472, 'best_epoch': 1, 'metric': 'val_raw_loss'}; trace columns: yes

- train_total_loss: 1.0798
- train_raw_loss: 2.1197
- train_high_reg_loss: 0.0295
- train_high_class_loss: 0.6883
- train_low_reg_loss: -0.0310
- train_low_class_loss: 0.6753
- train_close_reg_loss: 0.0390
- train_close_class_loss: 0.6985
- train_high_reg_calib_loss: 0.0742
- train_low_reg_calib_loss: 0.0772
- train_close_reg_calib_loss: 0.0699
- val_total_loss: -2.5149
- val_raw_loss: -1.4734
- val_high_reg_loss: -1.3683
- val_high_class_loss: 0.6843
- val_low_reg_loss: -1.4952
- val_low_class_loss: 0.6691
- val_close_reg_loss: -0.6744
- val_close_class_loss: 0.6930
- val_high_reg_calib_loss: 0.0827
- val_low_reg_calib_loss: 0.0793
- val_close_reg_calib_loss: 0.0817
- val_selection_metric: -1.4734
- trace_sched_lambda_reg: 0.0000
- trace_sched_lambda_calib: 0.0000
- trace_kendall_weight_high_class: 0.3334
- trace_kendall_weight_low_class: 0.3333
- trace_kendall_weight_close_class: 0.3333
- trace_kendall_logvar_high_class: -0.0402
- trace_kendall_logvar_low_class: -0.0399
- trace_kendall_logvar_close_class: -0.0397
- trace_gradshare_high_reg: 0.5974
- trace_gradshare_high_class: 0.0003
- trace_gradshare_low_reg: 0.3130
- trace_gradshare_low_class: 0.0003
- trace_gradshare_close_reg: 0.0889
- trace_gradshare_close_class: 0.0001
- trace_epoch_seconds: 8.7276
