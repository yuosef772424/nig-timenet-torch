# Recovered from a past session (scratch file, not a repo module). Ran on the Colab runtime (sent through cbx.py): prints the shape of the full artifact.
# Original name: scratchpad/explore.py. Kept verbatim except this header and ruff-noqa.
# ruff: noqa
import numpy as np, pandas as pd
from nig_timenet_torch.data_io.loading import LoaderSettings, load_dataset
from nig_timenet_torch.core.schema import LAST_COLUMNS, TS_COL
from nig_timenet_torch.workflow.run_common import split_bounds
from nig_timenet_torch.workflow.settings import SplitSettings
src = load_dataset(LoaderSettings(name="/content/art1", local_dir=""))
ds = src.dataset
print({k: (v.shape if hasattr(v, "shape") else (type(v).__name__, (v if not isinstance(v, (list, dict)) else len(v)))) for k, v in ds.items()})
print(LAST_COLUMNS, TS_COL)
last = np.asarray(ds["last_candles"]); ts = last[:, TS_COL].astype(np.int64)
print(pd.Timestamp(ts.min()), pd.Timestamp(ts.max()))
print("split_bounds", split_bounds(ds, SplitSettings()))
u, c = np.unique(ts, return_counts=True)
print("timestamps", len(u), "coins/ts quantiles", np.quantile(c, [0, .1, .5, .9, 1]))
s = pd.Series(c, index=pd.to_datetime(u)); print(s.resample("YS").mean().round(0).to_dict())
b = ds["asset_bounds"]; print(b[0], len(b))
print("feature_order", ds.get("feature_order"), "targets", ds.get("targets"), ds.get("forecast_horizon"), ds.get("reg_target_mode"), ds.get("price_norm_mode"), ds.get("stride"))
print(last[:3]); print(np.asarray(ds["X_4h"][:1, -3:]), [np.asarray(ds[k][:3]) for k in ("y_close_reg","y_high_reg","y_low_reg","y_close_class")])
