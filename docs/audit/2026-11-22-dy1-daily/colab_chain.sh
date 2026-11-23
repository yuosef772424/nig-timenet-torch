#!/usr/bin/env bash
# DY1 amendment 1 on Colab: build the daily artifacts, the smoke (rule E5), then the six dy1-vm-* arms in parallel.
#   bash colab_chain.sh <pinned checkout, e.g. /content/nig_t>
# Started again on the same commit it skips what is built and resumes every arm from its run_dir.
set -euo pipefail
R=$1; A=$R/docs/audit/2026-11-22-dy1-daily; RUNS=/content/drive/MyDrive/nig_timenet_runs
. /content/nig_env.sh
export PYTHONPATH=$(echo "$PYTHONPATH" | sed "s#/content/nig-timenet-torch/src#$R/src#") NIG_REPO_DIR=$R
cd "$R"
[ -f /content/art_vm/manifest.json ] || python -u "$A/build_daily.py" /content/art_vm --raw /content/raw_1d \
  --coins-file "$A/coins_volmatched.txt" --workers 12
[ -f /content/art_vm_mres/manifest.json ] || python -u -m nig_timenet_torch.data.market_residual /content/art_vm \
  /content/art_vm_mres --raw /content/raw_1d --beta-bars 90 | tail -12
mkdir -p /content/cfg
for j in dy1-vm-smoke-001 dy1-vm-s-def-001 dy1-vm-s-std-001 dy1-vm-s-mres-001 dy1-vm-p-def-001 dy1-vm-p-std-001 \
         dy1-vm-p-mres-001; do
  python -c "import json; json.dump(json.load(open('docs/jobs/$j.json'))['config'], open('/content/cfg/$j.json','w'))"
done
if [ ! -f $RUNS/dy1-vm-smoke-001/out/rolling_summary.json ]; then
  python -u -m nig_timenet_torch.jobs.rolling --config /content/cfg/dy1-vm-smoke-001.json \
    --out $RUNS/dy1-vm-smoke-001/out --max-minutes 20 > /content/dy1-vm-smoke-001.log 2>&1
fi
tail -2 /content/dy1-vm-smoke-001.log || true
for a in s-def s-std s-mres p-def p-std p-mres; do
  j=dy1-vm-$a-001
  [ -f $RUNS/$j/out/rolling_summary.json ] && continue
  python -u -m nig_timenet_torch.jobs.rolling --config /content/cfg/$j.json --out $RUNS/$j/out --max-minutes 170 \
    > /content/$j.log 2>&1 &
done
wait
echo "chain done"
