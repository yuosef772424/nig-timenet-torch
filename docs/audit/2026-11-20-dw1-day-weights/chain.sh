#!/usr/bin/env bash
# DW1 (PREREGISTERED.md here): the smoke of arm DW as two fold parts (rule E5, the GPU smoke of this commit), then
# every seed of the three arms split into fold groups, all at once. The part that completes a seed writes its
# rolling_summary.json. Resumes on the SAME commit (start the same command again).
#
#   [JOBS="job ..."] bash chain.sh REPO_DIR ["0,1,2,3 4,5,6"]
set -euo pipefail
REPO=${1:?usage: chain.sh REPO_DIR [GROUPS]}
GROUPS_=${2:-"0,1,2,3 4,5,6"}
JOBS_=${JOBS:-"rolling-dw1-r-001 rolling-dw1-dw-001 rolling-dw1-bal-001"}
R=/content/drive/MyDrive/nig_timenet_runs
export PYTHONPATH=/content/nig_venv/lib/python3.13/site-packages:$REPO/src
export NIG_REPO_DIR=$REPO
cd "$REPO"
echo "commit $(git rev-parse --short HEAD) groups $GROUPS_ start $(date -u +%T)"
[ -f /content/art1/manifest.json ] || { echo "/content/art1 missing: run colab_ops setup first"; exit 1; }
CFG=/content/cfg_dw1
mkdir -p "$CFG"
python3 - "$CFG" "$GROUPS_" "$JOBS_" <<'PY'
import json, sys
cfg, groups = sys.argv[1], [[int(i) for i in g.split(",")] for g in sys.argv[2].split()]
c = json.load(open("docs/jobs/rolling-dw1-smoke-001.json"))["config"]
for f in (0, 1):
    json.dump(dict(c, rolling=dict(c["rolling"], fold_ids=[f])), open(f"{cfg}/smoke_f{f}.json", "w"))
for job in sys.argv[3].split():
    c = json.load(open(f"docs/jobs/{job}.json"))["config"]
    for s in c["rolling"]["seeds"]:
        for g in groups:
            tag = "-".join(map(str, g))
            json.dump(dict(c, rolling=dict(c["rolling"], seeds=[s], fold_ids=g), run_dir=c["run_dir"] + f"_s{s}"),
                      open(f"{cfg}/{job}_s{s}_f{tag}.json", "w"))
PY
if [ ! -f "$R/rolling-dw1-smoke-001/out/rolling_summary.json" ]; then
  for f in 1 0; do
    python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/smoke_f$f.json" \
      --out "$R/rolling-dw1-smoke-001/out" --max-minutes 20 | tail -n 3
  done
fi
for job in $JOBS_; do
  for cfgf in "$CFG/${job}"_s*_f*.json; do
    b=$(basename "$cfgf" .json); s=${b#${job}_s}; s=${s%%_*}
    [ -f "$R/$job/out_s$s/rolling_summary.json" ] && continue
    nohup python -u -m nig_timenet_torch.jobs.rolling --config "$cfgf" --out "$R/$job/out_s$s" --max-minutes 180 \
      > "/content/$b.log" 2>&1 &
    echo "started $b"
  done
done
wait
echo "CHAIN-DONE $(date -u +%T)"
