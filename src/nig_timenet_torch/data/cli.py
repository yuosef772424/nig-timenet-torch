"""nig-data command line

PURPOSE: Build, fetch or show the resolved settings of the native data pipeline from the command line (python -m
    nig_timenet_torch.data.cli).
TAGS: cli, nig-data, console script, build, fetch, settings override
PITFALLS: an entry point declared in pyproject.toml: nothing imports it, so the G8 orphan scan reads the script
    table; unknown dotted keys raise.

``python -m nig_timenet_torch.data.cli``: build or download data from the command line.

    build  [--preset NAME ...] [--config FILE.json] [--set dotted.key=VALUE ...]
           builds the dataset and writes it when output.path is set
    fetch  the same options; downloads the candles of the configured coins (SourceSettings.fetch) and stops
    show   print the resolved settings as JSON
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from nig_timenet_torch.data.build import feature_order
from nig_timenet_torch.data.fetch import fetch_all
from nig_timenet_torch.data.lock import is_sealed, load_lock
from nig_timenet_torch.data.pipeline import build_artifact
from nig_timenet_torch.data.presets import PRESETS, preset
from nig_timenet_torch.data.settings import DataSettings
from nig_timenet_torch.data.sources import resolve_coins

__all__ = ["main", "settings_from_args"]


def _value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def settings_from_args(presets: list[str], config: str = "", sets: list[str] = ()) -> DataSettings:
    """presets, then a JSON file, then ``key=value`` overrides, applied in that order."""
    base = DataSettings.from_dict(json.loads(Path(config).read_text(encoding="utf-8"))) if config else None
    s = preset(*presets, base=base)
    patch = {}
    for item in sets:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects dotted.key=VALUE, got {item!r}")
        patch[key] = _value(val)
    return s.updated(patch) if patch else s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "fetch", "show"))
    ap.add_argument("--preset", action="append", default=[], choices=sorted(PRESETS))
    ap.add_argument("--config", default="")
    ap.add_argument("--set", action="append", default=[], dest="sets")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = settings_from_args(a.preset, a.config, a.sets)
    if a.command == "show":
        print(json.dumps(s.to_dict(), indent=2, default=list))
        _ = feature_order(s)
        return 0
    if a.command == "fetch":
        lock = load_lock()
        names = [n for n in resolve_coins(s.source) if not is_sealed(n, lock)]
        added = fetch_all(s.source, names)
        print(f"{sum(added.values())} rows added for {len(added)} coins")
        return 0
    _, manifest = build_artifact(s)
    if manifest:
        print(f"written to {s.output.directory or s.output.path} ({manifest['export']['rows']} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
