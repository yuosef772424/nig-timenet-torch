# Health report — /content/drive/MyDrive/nig_timenet_runs/trial-002/train

Rows: 100000

## Targets (validation)

| head | rank IC (ts) | per-coin rho | AUC (const / shuffled) | T-conf rho | gap CI95 | conf ok | range-share | coverage80 | sharp ok |
|---|---|---|---|---|---|---|---|---|---|
| high | 0.400 | 0.225 | 0.750 (0.50 / 0.498) | -0.527 | [-0.008, 0.072] | False | 1.000 | 0.895 | False |
| low | 0.433 | 0.260 | 0.763 (0.50 / 0.497) | -1.000 | [0.621, 0.691] | True | 1.000 | 0.925 | False |
| close | 0.063 | 0.073 | 0.524 (0.50 / 0.496) | -0.988 | [0.495, 0.575] | True | 0.999 | 0.862 | False |

Baselines: constant confidence scores 0; a zero-width interval scores 1.0 on range-share (read it with coverage); constant AUC 0.5.

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

Epochs: 2; best: {'best': -5.062315464019775, 'best_epoch': 1, 'metric': 'val_raw_loss'}; trace columns: yes

- train_total_loss: -5.9944
- train_raw_loss: -4.9100
- train_high_reg_loss: -2.3777
- train_high_class_loss: 0.5966
- train_low_reg_loss: -2.3371
- train_low_class_loss: 0.5968
- train_close_reg_loss: -2.0843
- train_close_class_loss: 0.6867
- train_high_reg_calib_loss: 0.0125
- train_low_reg_calib_loss: 0.0124
- train_close_reg_calib_loss: 0.0143
- val_total_loss: -6.1108
- val_raw_loss: -5.0240
- val_high_reg_loss: -2.3381
- val_high_class_loss: 0.6069
- val_low_reg_loss: -2.4593
- val_low_class_loss: 0.5921
- val_close_reg_loss: -2.1268
- val_close_class_loss: 0.6923
- val_high_reg_calib_loss: 0.0106
- val_low_reg_calib_loss: 0.0119
- val_close_reg_calib_loss: 0.0133
- val_selection_metric: -5.0240
- trace_sched_lambda_reg: 0.0100
- trace_sched_lambda_calib: 0.0095
- trace_kendall_weight_high_class: 0.3489
- trace_kendall_weight_low_class: 0.3480
- trace_kendall_weight_close_class: 0.3032
- trace_kendall_logvar_high_class: -0.5172
- trace_kendall_logvar_low_class: -0.5147
- trace_kendall_logvar_close_class: -0.3769
- trace_gradshare_high_reg: 0.2189
- trace_gradshare_high_class: 0.0036
- trace_gradshare_low_reg: 0.4422
- trace_gradshare_low_class: 0.0028
- trace_gradshare_close_reg: 0.3313
- trace_gradshare_close_class: 0.0012
- trace_epoch_seconds: 166.7178
