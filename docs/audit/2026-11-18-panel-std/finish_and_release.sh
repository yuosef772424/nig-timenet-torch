#!/usr/bin/env bash
# Wait until no training job runs on this Colab runtime, write the readings of the finished walk-forwards to Drive,
# flush Drive and release the runtime (owner 2026-11-18 / 19: "end the Colab session after the training").
# Usage: nohup bash finish_and_release.sh > /content/drive/MyDrive/nig_timenet_runs/finish_and_release.log 2>&1 &
set -uo pipefail
RUNS=/content/drive/MyDrive/nig_timenet_runs
CODE=${CODE:-/content/nig_u}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
while pgrep -f "nig_timenet_torch.jobs.(rolling|panel)|nig_timenet_torch.data_io.lower_timeframe" > /dev/null; do
  sleep 120
done
say "no job left"
. /content/nig_env.sh
export PYTHONPATH=${PYTHONPATH//\/content\/nig-timenet-torch\/src/$CODE\/src}
for pair in "panel-rolling-2tf-001 panel-rolling-ref-003" "panel-rolling-ls-001 panel-rolling-ref-002" \
            "panel-rolling-ref-003 panel-rolling-ref-002" "panel-rolling-usdt-001 panel-rolling-ref-002" \
            "panel-rolling-std-order-001 panel-rolling-ref-002"; do
  set -- $pair
  say "reading $1 vs $2"
  python -u $CODE/docs/audit/2026-11-18-panel-std/read_std_rolling.py --cand $RUNS/$1/out --ref $RUNS/$2/out \
    --out $RUNS/$1/reading_vs_$2 > $RUNS/$1/reading_vs_$2.log 2>&1 || say "reading $1 failed"
done
say "BTC-hedged book on the later walk-forwards"
python -u /content/btc_book.py panel-rolling-2tf-001 panel-rolling-ref-003 panel-rolling-usdt-001 \
  panel-rolling-std-order-001 > $RUNS/memorisation-check-001/btc_book_later.log 2>&1 || say "btc book failed"
say "flush Drive and release the runtime"
sleep 60
python - <<'PY'
import glob
import jupyter_client
f = sorted(glob.glob("/root/.local/share/jupyter/runtime/kernel-*.json"))[-1]
kc = jupyter_client.BlockingKernelClient(connection_file=f)
kc.load_connection_file()
kc.start_channels()
kc.execute("from google.colab import drive; drive.flush_and_unmount()\nfrom google.colab import runtime; runtime.unassign()")
print("sent to", f)
PY
