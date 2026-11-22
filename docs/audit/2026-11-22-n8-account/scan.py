"""N8 account scan: the ACCOUNT section (eval/account.py) of every run folder on Drive that holds battery tables.

Usage (Colab, after `colab_ops setup <branch> --no-artifact`):
    . /content/nig_env.sh; python scan.py /content/drive/MyDrive/nig_timenet_runs /content/n8_scan.json
A group = the fold folders (`fold_<k>_seed_<s>`) of one run folder and ONE seed, or a single run folder; its folds go
into ONE account, each with its own VAL threshold (`workflow.battery_run.account_from_folders`). A table saved without
`last_close` cannot show its horizon: those groups run with 2 sub-books (`--books 2`: every run here used the 4 h
artifact whose label is `future_close(i) = last_close(i + 2)`, docs/DATA_ARTIFACT_CONTRACT.md) and are marked
`books_from = "artifact"`. Writes one JSON entry per group and prints one line per group.
"""
import json
import re
import sys
import time
from pathlib import Path

from nig_timenet_torch.workflow.battery_run import account_from_folders


def main(root: Path, out: Path) -> None:
    groups: dict[str, list[Path]] = {}
    for t in sorted(root.glob("*/**/battery_table_test.parquet")):
        f = t.parent
        if not (f / "battery_table_val.parquet").is_file():
            continue
        seed = re.search(r"seed_(\d+)$", f.name)
        key = f"{f.parent.relative_to(root)} seed {seed.group(1)}" if f.name.startswith("fold_") and seed else (
            str(f.parent.relative_to(root)) if f.name.startswith("fold_") else str(f.relative_to(root)))
        groups.setdefault(key, []).append(f)
    res = {}
    for key, folders in groups.items():
        rel, t0, books_from = key, time.time(), "table"
        try:
            sec = account_from_folders(sorted(folders), n_placebo=20)
            if "SKIPPED" in sec["title"] and "last_close" in " ".join(sec["rows"]):
                sec, books_from = account_from_folders(sorted(folders), n_placebo=20, account_books=2), "artifact"
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the scan
            res[rel] = {"folds": len(folders), "error": repr(e)[:400]}
            print(f"{rel:<60} ERROR {repr(e)[:160]}", flush=True)
            continue
        s = sec["stats"]
        res[rel] = {"folds": len(folders), "title": sec["title"], "rows": sec["rows"], "note": sec["note"],
                    "stats": s, "seconds": round(time.time() - t0, 1),
                    "books_from": books_from}
        if "model" in s:
            m, p = s["model"], s.get("placebo", {})
            print(f"{rel:<60} {books_from[:3]} folds {len(folders):>2} CAGR {m['cagr'] * 100:+6.1f}% "
                  f"Sharpe {m['sharpe']:5.2f} "
                  f"DD {m['max_dd'] * 100:6.1f}% | placebo p95 {p.get('cagr_p95', float('nan')) * 100:+6.1f}% "
                  f"beats {p.get('model_beats', float('nan')) * 100:3.0f}% | timing "
                  f"{s['breadth timing']['cagr'] * 100:+6.1f}% | BTC "
                  f"{s.get('buy and hold', {}).get('cagr', float('nan')) * 100:+6.1f}%", flush=True)
        else:
            print(f"{rel:<60} {sec['title']} {sec['rows']}", flush=True)
        out.write_text(json.dumps(res, indent=1, default=float))
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
