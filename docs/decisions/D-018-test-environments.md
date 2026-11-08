# D-018: test environments E1 / E2 / E3 and the E1 generator defaults (ROADMAP G1)

Date: 2026-11-08
Owner decision: step G1 (manager message of 2026-11-08: "ROADMAP step G1 (test environments E1 / E2 / E3)", "A new
decision file for the generator defaults", "the E1 generator must make the label scale a parameter (default = the real
artifact's scale and magnitude, so E1 shows the same pathology) ... do NOT change any model default").
Why: MODEL_DEV_PROTOCOL section 3 needs the three environments before the gates (G2) can run.

What changes behaviour: nothing that existed. No model, trainer, data-pipeline or evaluation default changes. New:

- `data_io/walk_max.py`: exact mean / variance / MGF of the running maximum of a Gaussian random walk (Spitzer).
- `data_io/synthetic.py` (E1): `SyntheticSettings` defaults below. Realistic by measurement against the real artifact
  `c572b1b36960` (median last-bar range 0.026 vs 0.029, lag-1 autocorrelation of the log bar range 0.54 vs 0.55, close
  label sd 0.032 vs 0.039, high label mean 0.022 vs 0.025), window 32 x 3 channels (log high / low / close vs the
  previous close), 4h, horizon 2, label scale 1.0 and the real artifact's target declaration (`reg_target_mode`
  "return" + `price_norm_mode` "logret_std" -> codec `logret_close`), so E1 shows B-052 (y^2 ~ 5e-4 << beta_min 0.01).
  One row per horizon block (dataset `stride` = 2, the real artifact has 1): drift and volatility are constant over a
  row's horizon, which makes the oracle moments exact. Planted close IC 0.05 (the protocol's 0.10 variant is
  `close_ic=0.10`); volatility = coin scale^0.3 x (HAR / kappa)^0.7 x exp(-0.2 z_lev) (HAR = mean of the last 1 / 6 / 32
  bar ranges; z_lev = the standardised move of the last 6 bars: the known component beyond HAR).
- `data_io/environments.py`: E2 `permute_labels(seed=0)` (one global permutation of every label row; the future columns
  of last_candles follow the labels; oracle columns dropped) and E3 `real_slice` (20 coins x 6 months ending at the full
  run's train end, coverage >= 0.95, coins at evenly spaced ranks of their median bar range, own split 70 / 15 / 15 in
  `split_dates`); console script `nig-envs` (pyproject.toml) writes E1 / E2 / E3 as contract artifacts.
- `data_io/target_codec.future_column(target)`: the last_candles future column of a head (no default).

- `nig_timenet_torch.data_io.environments.permute_labels(seed)`: `<new>` -> `0`
- `nig_timenet_torch.data_io.environments.real_slice(min_coverage)`: `<new>` -> `0.95`
- `nig_timenet_torch.data_io.environments.real_slice(months)`: `<new>` -> `6`
- `nig_timenet_torch.data_io.environments.real_slice(n_coins)`: `<new>` -> `20`
- `nig_timenet_torch.data_io.environments.real_slice(train_share)`: `<new>` -> `0.7`
- `nig_timenet_torch.data_io.environments.real_slice(val_share)`: `<new>` -> `0.15`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(base_tf)`: `<new>` -> `'4h'`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(burn_in_bars)`: `<new>` -> `128`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(close_ic)`: `<new>` -> `0.05`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(coin_sigma_dispersion)`: `<new>` -> `0.25`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(coin_sigma_median)`: `<new>` -> `0.02`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(har_weight)`: `<new>` -> `0.7`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(horizon)`: `<new>` -> `2`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(label_codec)`: `<new>` -> `'logret_close'`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(leverage)`: `<new>` -> `0.2`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(leverage_bars)`: `<new>` -> `6`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(n_bars)`: `<new>` -> `4000`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(n_coins)`: `<new>` -> `32`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(reg_target_scale)`: `<new>` -> `1.0`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(seed)`: `<new>` -> `0`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(signal_bars)`: `<new>` -> `3`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(start)`: `<new>` -> `'2023-01-01T00:00:00+00:00'`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(substeps)`: `<new>` -> `48`
- `nig_timenet_torch.data_io.synthetic.SyntheticSettings(window)`: `<new>` -> `32`
- `nig_timenet_torch.data_io.synthetic.build_synthetic(settings)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.synthetic.write_synthetic(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.synthetic.write_synthetic(settings)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.synthetic.write_synthetic(source_commit)`: `<new>` -> `''`
