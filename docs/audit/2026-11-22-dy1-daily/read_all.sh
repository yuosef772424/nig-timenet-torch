#!/usr/bin/env bash
# DY1: the pre-registered reading of every finished dy1-vm arm (Colab; tables on Drive). bash read_all.sh <checkout>
set -uo pipefail
R=$1; A=$R/docs/audit/2026-11-22-dy1-daily; RUNS=/content/drive/MyDrive/nig_timenet_runs; O=$RUNS/_dy1_read
mkdir -p $O
. /content/nig_env.sh
export PYTHONPATH=$(echo "$PYTHONPATH" | sed "s#/content/nig-timenet-torch/src#$R/src#")
cd $R
args=()
for d in $RUNS/dy1-vm-*-001; do
  a=$(basename $d | sed 's/dy1-vm-//; s/-001//'); [ "$a" = smoke ] && continue
  [ -f $d/out/rolling_summary.json ] || continue
  for s in 0 1; do
    if [[ $a == *mres ]]; then
      python $A/own_returns.py /content/art_vm $O/own/$a/seed_$s $d/out/fold_*_seed_$s > /dev/null || exit 1
      args+=("$a/own/s$s=$O/own/$a/seed_$s/fold_*_seed_$s")
    fi
    args+=("$a/s$s=$d/out/fold_*_seed_$s")
  done
done
python $A/read_dy1.py $O/read_dy1.json "${args[@]}" 2>&1 | grep -v Warning > $O/read_dy1.txt
python $A/failure_read.py "${args[@]}" 2>&1 | grep -v Warning > $O/failure_read.txt
echo "read_all done"
