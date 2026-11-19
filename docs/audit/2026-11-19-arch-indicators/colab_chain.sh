#!/usr/bin/env bash
# Colab chain for 2026-11-20 (start a fresh chat with docs/handoff/2026-11-20-start-here.md):
#   method 1 = BR1 (BTC-relative inputs + BTC-relative close label, jobs rolling-btcrel-lbl-*)
#   method 3 = M3  (market-residual inputs + residual close label, jobs rolling-mres-*)
# For each: build the artifact (skipped when it exists), run its smoke (rule E5), then seeds 0 and 1 in parallel.
#
#   bash colab_chain.sh REPO_DIR [br1|m3|both]      (REPO_DIR = a checkout PINNED at one commit, e.g. /content/nig_t)
#
# Needs: /content/nig_venv (colab_ops setup), /content/art1 (the artifact copy), Drive mounted.
# A cut main run resumes when this script is started again on the SAME commit (checkpoints under run_dir on Drive).
set -euo pipefail
REPO=${1:?usage: colab_chain.sh REPO_DIR [br1|m3|both]}
WHAT=${2:-both}
R=/content/drive/MyDrive/nig_timenet_runs
CACHE=/content/drive/MyDrive/crypto_model/candle_cache_4h_hlcqv
export PYTHONPATH=/content/nig_venv/lib/python3.13/site-packages:$REPO/src
export NIG_REPO_DIR=$REPO
cd "$REPO"
echo "commit $(git rev-parse --short HEAD)"
CFG=/content/cfg_chain
mkdir -p "$CFG"

configs() {   # job_id smoke_id artifact -> $CFG/<job>_smoke.json and <job>_s0/_s1.json
  python3 - "$1" "$2" "$3" "$CFG" <<'EOF'
import json, sys
job, smoke, art, cfg = sys.argv[1:]
c = json.load(open(f"docs/jobs/{smoke}.json"))["config"]; c["artifact"] = art
json.dump(c, open(f"{cfg}/{job}_smoke.json", "w"))
c = json.load(open(f"docs/jobs/{job}.json"))["config"]; c["artifact"] = art
for s in (0, 1):
    d = dict(c, rolling=dict(c["rolling"], seeds=[s]), run_dir=c["run_dir"] + f"_s{s}")
    json.dump(d, open(f"{cfg}/{job}_s{s}.json", "w"))
EOF
}

run_arm() {   # job_id smoke_id artifact
  local job=$1 smoke=$2 art=$3
  configs "$job" "$smoke" "$art"
  if [ ! -f "$R/$smoke/out/rolling_summary.json" ]; then
    python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/${job}_smoke.json" --out "$R/$smoke/out" \
      --max-minutes 20 | tail -n 2
  fi
  for s in 0 1; do
    if [ ! -f "$R/$job/out_s$s/rolling_summary.json" ]; then
      nohup python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/${job}_s$s.json" --out "$R/$job/out_s$s" \
        --max-minutes 120 > "/content/${job}_s$s.log" 2>&1 &
      echo "started $job seed $s (log /content/${job}_s$s.log)"
    fi
  done
}

if [ "$WHAT" = br1 ] || [ "$WHAT" = both ]; then
  [ -f /content/art_btcrel/manifest.json ] || \
    python -u -m nig_timenet_torch.data_io.btc_relative /content/art1 /content/art_btcrel --cache "$CACHE" | tail -n 4
  [ -f /content/art_btcrel_lbl/manifest.json ] || \
    python -u -m nig_timenet_torch.data_io.btc_relative /content/art_btcrel /content/art_btcrel_lbl --label-only | tail -n 4
  run_arm rolling-btcrel-lbl-001 rolling-btcrel-lbl-smoke-001 /content/art_btcrel_lbl
fi
if [ "$WHAT" = m3 ] || [ "$WHAT" = both ]; then
  [ -f /content/art_mres/manifest.json ] || \
    python -u -m nig_timenet_torch.data_io.market_residual /content/art1 /content/art_mres --cache "$CACHE" | \
      grep -v per_coin | tail -n 12
  run_arm rolling-mres-001 rolling-mres-smoke-001 /content/art_mres
fi
wait
echo CHAIN-DONE
