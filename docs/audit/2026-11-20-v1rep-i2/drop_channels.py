"""Write a contract artifact with a SUBSET of the input channels of another one (hypothesis I2 of i1-001: channel
ablation of the market-residual artifact). Rows, labels, coins and splits are unchanged; the kept channels keep
their order.

    python drop_channels.py SRC DST --drop res_rsi_14,res_stoch_k_14_3,res_stoch_d_3,volume_rel_30

A study script (rule G6): reads and writes artifacts with the package's writer, no training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import read_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop", required=True)
    a = ap.parse_args(argv)
    src, dst = Path(a.src), Path(a.dst)
    man = read_manifest(src)
    ds = npy_dir.load(src, mmap=False)
    names = list(ds["feature_order"])
    drop = a.drop.split(",")
    missing = sorted(set(drop) - set(names))
    if missing:
        raise SystemExit(f"{src} has no channels {missing}; it has {names}")
    keep = [i for i, n in enumerate(names) if n not in drop]
    key = f"X_{ds['base_tf']}"
    ds = {**ds, key: np.ascontiguousarray(np.asarray(ds[key])[:, :, keep]),
          "feature_order": [names[i] for i in keep]}
    out = write_artifact(ds, dst, source_commit=man.source_commit,
                         producer=f"drop_channels.py (-{a.drop} on {man.data_fingerprint})")
    rec = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
           "fingerprint": out["export"]["data_sha256"], "dropped": drop, "kept": ds["feature_order"]}
    (dst / "derivation_drop_channels.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
