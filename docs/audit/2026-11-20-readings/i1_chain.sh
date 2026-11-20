#!/usr/bin/env bash
# Arm I1 (PREREGISTERED_I1.md): art_ind with the R configuration, its folds SPLIT over several processes
# (RollingSettings.fold_ids, owner 2026-11-20: "every fold trains an independent model"). Copy the artifact (skipped
# when present), run the smoke as two parts (fold 1, then fold 0: it exercises the split on the GPU, rule E5), then one
# process per seed x fold group, all at once. The part that completes a seed's folds writes its rolling_summary.json.
# Resumes on the SAME commit (each fold x seed has its checkpoint under run_dir).
#
#   bash i1_chain.sh REPO_DIR ["0,1,2 3,4 5,6"]      (REPO_DIR = a checkout PINNED at one commit; fold groups)
set -euo pipefail
REPO=${1:?usage: i1_chain.sh REPO_DIR [GROUPS]}
GROUPS_=${2:-"0,1,2 3,4 5,6"}
R=/content/drive/MyDrive/nig_timenet_runs
export PYTHONPATH=/content/nig_venv/lib/python3.13/site-packages:$REPO/src
export NIG_REPO_DIR=$REPO
cd "$REPO"
echo "commit $(git rev-parse --short HEAD) groups $GROUPS_ start $(date -u +%T)"
[ -f /content/art_ind/manifest.json ] || \
  cp -r /content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1_plus_ind /content/art_ind
CFG=/content/cfg_i1
mkdir -p "$CFG"
python3 - "$CFG" "$GROUPS_" <<'EOF'
import json, sys
cfg, groups = sys.argv[1], [[int(i) for i in g.split(",")] for g in sys.argv[2].split()]
c = json.load(open("docs/jobs/rolling-indr-smoke-001.json"))["config"]
for f in (0, 1):
    json.dump(dict(c, rolling=dict(c["rolling"], fold_ids=[f])), open(f"{cfg}/smoke_f{f}.json", "w"))
c = json.load(open("docs/jobs/rolling-indr-001.json"))["config"]
for s in (0, 1):
    for g in groups:
        tag = "-".join(map(str, g))
        json.dump(dict(c, rolling=dict(c["rolling"], seeds=[s], fold_ids=g), run_dir=c["run_dir"] + f"_s{s}"),
                  open(f"{cfg}/s{s}_f{tag}.json", "w"))
EOF
if [ ! -f "$R/rolling-indr-smoke-001/out/rolling_summary.json" ]; then
  for f in 1 0; do
    python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/smoke_f$f.json" --out "$R/rolling-indr-smoke-001/out" \
      --max-minutes 20 | tail -n 1
  done
fi
for s in 0 1; do
  if [ ! -f "$R/rolling-indr-001/out_s$s/rolling_summary.json" ]; then
    for g in $GROUPS_; do
      tag=${g//,/-}
      nohup python -u -m nig_timenet_torch.jobs.rolling --config "$CFG/s${s}_f$tag.json" \
        --out "$R/rolling-indr-001/out_s$s" --max-minutes 120 > "/content/rolling-indr-001_s${s}_f$tag.log" 2>&1 &
      echo "started rolling-indr-001 seed $s folds $g"
    done
  fi
done
wait
echo "I1-DONE $(date -u +%T)"
