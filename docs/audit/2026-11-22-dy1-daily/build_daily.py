"""DY1: build the daily contract artifact (1d bars, window 32, horizon 2 days) from the public Binance archive.

Owner task 2026-11-22 (DY1). Universe: every USDT perpetual of data.binance.vision (delisted ones included, so the
book can hold a coin that later died), minus the sealed holdout coins, whose first daily month is <= --listed-by
(a later listing cannot be among the 100 longest histories of any fold). Bars from --start; bars at or after the time
seal are dropped by build_logret_dataset. Inputs / labels: data/vision_slice.py::build_logret_dataset (the real
artifact's definition: log high / low / close moves vs the previous close; labels log(future / last close)).

    python docs/audit/2026-11-22-dy1-daily/build_daily.py OUT --raw RAW [--start 2020-01] [--end 2026-09]
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from nig_timenet_torch.core.schema import OPEN_TIME_COL
from nig_timenet_torch.data.vision_slice import build_logret_dataset, fetch_month, load_frames
from nig_timenet_torch.data_io.artifact_writer import write_artifact
from nig_timenet_torch.data_io.lock import is_sealed, load_lock

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/klines/"


def _get(url: str, tries: int = 6):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def list_symbols() -> list[str]:
    names, marker = [], ""
    while True:
        s = _get(f"{S3}?delimiter=/&prefix={PREFIX}" + (f"&marker={PREFIX}{marker}/" if marker else "")).decode()
        p = re.findall(rf"<Prefix>{PREFIX}([^/<]+)/</Prefix>", s)
        names += p
        if len(p) < 1000:
            return [n for n in names if n.endswith("USDT")]
        marker = p[-1]


def months_of(symbol: str, interval: str) -> list[str]:
    s = _get(f"{S3}?prefix={PREFIX}{symbol}/{interval}/").decode()
    return sorted(set(re.findall(rf"{symbol}-{interval}-(\d{{4}}-\d{{2}})\.zip<", s)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--listed-by", default="2023-06")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    raw, lock = Path(a.raw), load_lock()
    raw.mkdir(parents=True, exist_ok=True)
    plan_path = raw / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
    else:
        syms = [s for s in list_symbols() if not is_sealed(s, lock)]
        with ThreadPoolExecutor(a.workers) as pool:
            got = list(pool.map(lambda s: months_of(s, a.interval), syms))
        plan = {s: [m for m in ms if a.start <= m <= a.end] for s, ms in zip(syms, got) if ms and ms[0] <= a.listed_by}
        plan = {s: ms for s, ms in plan.items() if ms}
        plan_path.write_text(json.dumps(plan))
    print(f"{len(plan)} coins, {sum(map(len, plan.values()))} monthly files", flush=True)

    def fetch(s):
        for m in plan[s]:
            fetch_month(s, a.interval, pd.Period(m, "M"), raw, get=_get)
        return s

    with ThreadPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(fetch, sorted(plan))):
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(plan)} coins fetched", flush=True)
    frames = load_frames(raw, sorted(plan), a.interval)
    t0 = pd.Timestamp(a.start + "-01", tz="UTC").value
    frames = {c: f[f[OPEN_TIME_COL] >= t0].reset_index(drop=True) for c, f in frames.items()}
    frames = {c: f for c, f in frames.items() if len(f) > a.window + a.horizon + 1}
    ds = build_logret_dataset(frames, interval=a.interval, window=a.window, horizon=a.horizon, stride=1,
                              time_seal=lock.time_seal)
    ds["real_slice"]["universe"] = (f"every non-sealed USDT perpetual of data.binance.vision listed by {a.listed_by}, "
                                    f"bars from {a.start}")
    man = write_artifact(ds, a.out, source_commit="", overwrite=True, producer="nig_timenet_torch.data_io.vision_slice")
    print(f"wrote {a.out}: {man['export']['rows']} rows, {len(man['export']['coins'])} coins, "
          f"max_bar_time {man['export']['max_bar_time']}", flush=True)


if __name__ == "__main__":
    main()
