#!/usr/bin/env bash
# V1-rep + I2 (PREREGISTERED.md here). Builds art_mres_noind (art_mres must exist: colab_chain.sh m3 builds it),
# runs the I2 smoke as two fold parts (rule E5, the GPU smoke of this commit), then every seed of the three jobs split
# into fold groups, all at once. The part that completes a seed writes its rolling_summary.json. Resumes on the SAME
# commit.
#
#   [JOBS="job ..."] bash chain.sh REPO_DIR ["0,1,2,3 4,5,6"]   (JOBS: default the three jobs of PREREGISTERED.md)
set -euo pipefail
REPO=${1:?usage: chain.sh REPO_DIR [GROUPS]}
GROUPS_=${2:-"0,1,2,3 4,5,6"}
R=/content/drive/MyDrive/nig_timenet_runs
export PYTHONPATH=/content/nig_venv/lib/python3.13/site-packages:$REPO/src
export NIG_REPO_DIR=$REPO
cd "$REPO"
D=docs/audit/2026-11-20-v1rep-i2
echo "commit $(git rev-parse --short HEAD) groups $GROUPS_ start $(date -u +%T)"
[ -f /content/art_mres/manifest.json ] || { echo "art_mres missing: run colab_chain.sh m3 first"; exit 1; }
[ -f /content/art_mres_noind/manifest.json ] || python -u $D/drop_channels.py /content/art_mres /content/art_mres_noind \
  --drop res_rsi_14,res_stoch_k_14_3,res_stoch_d_3,volume_rel_30 | tail -n 12
CFG=/content/cfg_v1i2
mkdir -p "$CFG"
python3 - "$CFG" "$GROUPS_" "${JOBS:-rolling-r23-001 rolling-mres23-001 rolling-mres-noind-001}" <<'PY'
import json, sys
cfg, groups = sys.argv[1], [[int(i) for i in g.split(",")] for g in sys.argv[2].split()]
c = json.load(open("docs/jobs/rolling-mres-noind-smoke-001.json"))["config"]
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
if [ ! -f "$R/rolling-mres-noind-smoke-001/out/rolling_summary.json" ]; then
  for f in 1 0; do
    python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/smoke_f$f.json" \
      --out "$R/rolling-mres-noind-smoke-001/out" --max-minutes 20 | tail -n 1
  done
fi
for job in ${JOBS:-rolling-r23-001 rolling-mres23-001 rolling-mres-noind-001}; do
  for cfgf in "$CFG/${job}"_s*_f*.json; do
    b=$(basename "$cfgf" .json); s=${b#${job}_s}; s=${s%%_*}
    [ -f "$R/$job/out_s$s/rolling_summary.json" ] && continue
    nohup python -u -m nig_timenet_torch.jobs.rolling --config "$cfgf" --out "$R/$job/out_s$s" --max-minutes 120 \
      > "/content/$b.log" 2>&1 &
    echo "started $b"
  done
done
wait
echo "CHAIN-DONE $(date -u +%T)"
