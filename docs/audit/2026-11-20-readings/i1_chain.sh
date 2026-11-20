#!/usr/bin/env bash
# Arm I1 (PREREGISTERED_I1.md): art_ind with the R configuration. Copy the artifact (skipped when present), run the
# smoke (rule E5, skipped when done), then seeds 0 and 1 in parallel. Resumes on the SAME commit.
#
#   bash i1_chain.sh REPO_DIR      (REPO_DIR = a checkout PINNED at one commit)
set -euo pipefail
REPO=${1:?usage: i1_chain.sh REPO_DIR}
R=/content/drive/MyDrive/nig_timenet_runs
export PYTHONPATH=/content/nig_venv/lib/python3.13/site-packages:$REPO/src
export NIG_REPO_DIR=$REPO
cd "$REPO"
echo "commit $(git rev-parse --short HEAD)"
[ -f /content/art_ind/manifest.json ] || \
  cp -r /content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1_plus_ind /content/art_ind
CFG=/content/cfg_i1
mkdir -p "$CFG"
python3 - "$CFG" <<'EOF'
import json, sys
cfg = sys.argv[1]
c = json.load(open("docs/jobs/rolling-indr-smoke-001.json"))["config"]
json.dump(c, open(f"{cfg}/smoke.json", "w"))
c = json.load(open("docs/jobs/rolling-indr-001.json"))["config"]
for s in (0, 1):
    json.dump(dict(c, rolling=dict(c["rolling"], seeds=[s]), run_dir=c["run_dir"] + f"_s{s}"),
              open(f"{cfg}/s{s}.json", "w"))
EOF
if [ ! -f "$R/rolling-indr-smoke-001/out/rolling_summary.json" ]; then
  python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/smoke.json" --out "$R/rolling-indr-smoke-001/out" \
    --max-minutes 20 | tail -n 2
fi
for s in 0 1; do
  if [ ! -f "$R/rolling-indr-001/out_s$s/rolling_summary.json" ]; then
    nohup python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/s$s.json" --out "$R/rolling-indr-001/out_s$s" \
      --max-minutes 120 > "/content/rolling-indr-001_s$s.log" 2>&1 &
    echo "started rolling-indr-001 seed $s (log /content/rolling-indr-001_s$s.log)"
  fi
done
wait
echo I1-DONE
