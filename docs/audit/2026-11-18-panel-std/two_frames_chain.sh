#!/usr/bin/env bash
# The two-timeframe chain of owner task 2026-11-18, run on a Colab GPU runtime (no secret in this file; the GitHub
# token is read from Drive by the setup recipe, colab_ops/recipes.py). Steps, each resumable / idempotent:
#   0. code + cached env + the 4h artifact copy (colab_ops setup recipe)
#   1. the reading of panel-rolling-std-001 vs panel-rolling-ref-002 (CPU, in the background)
#   2. wait until no other rolling / panel job runs on this runtime
#   3. the 4h + 1h artifact: built once (1h candles from the Binance archive, cached per coin on Drive), kept on Drive
#   4. smoke panel-rolling-2tf-smoke-001 (rule E5), then panel-rolling-2tf-001 + panel-rolling-ref-003 side by side,
#      and the resume of panel-rolling-ls-001 (cut with the runtime on 2026-11-18)
#   5. the readings of 2tf vs ref-003 and of ls vs ref-002, on Drive
#   6. SHUTDOWN=1: release the runtime (google.colab runtime.unassign through the notebook kernel; unverified)
# Usage: SHUTDOWN=1 bash two_frames_chain.sh   (log: MyDrive/nig_timenet_runs/two_frames_chain.log)
set -uo pipefail
BRANCH=${BRANCH:-ccr-acb890db-b2ekab}
CODE=/content/nig-timenet-torch
RUNS=/content/drive/MyDrive/nig_timenet_runs
ART4=/content/art1
ART_DRIVE=/content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1_plus_1h
ART2=/content/art_1h4h
CACHE=/content/drive/MyDrive/crypto_model/candles_1h_cache
AUDIT=$CODE/docs/audit/2026-11-18-panel-std
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "0. setup $BRANCH"
cd /content
PYTHONPATH=$CODE/src python -c "from nig_timenet_torch.colab_ops.recipes import setup_script; \
print(setup_script('$BRANCH'))" > /content/setup_2tf.sh 2>/dev/null || {
  say "the checkout is missing: clone it first (see the run command in docs/handoff/2026-11-18-panel-std-rolling.md)"; exit 1; }
bash /content/setup_2tf.sh || { say "setup failed"; exit 1; }
. /content/nig_env.sh
export NIG_REPO_DIR=$CODE
cd $CODE

cfg() {  # cfg <job> <artifact>: the job file's config pointed at a local artifact copy
  mkdir -p /content/cfg
  python -c "import json,sys; c=json.load(open('docs/jobs/$1.json'))['config']; c['artifact']='$2'; \
json.dump(c, open('/content/cfg/$1.json','w'))"
}
job() {  # job <job> <max minutes> [checkout]: the rolling entry, outputs on Drive (resumes from run_dir); a
  # checkout other than $CODE runs that checkout's code and records its commit (a resume must keep its commit)
  local repo=${3:-$CODE}
  PYTHONPATH=${PYTHONPATH//$CODE\/src/$repo\/src} NIG_REPO_DIR=$repo python -u -m nig_timenet_torch.jobs.rolling \
    --config /content/cfg/$1.json --out $RUNS/$1/out --max-minutes $2 > $RUNS/$1.log 2>&1
}
reading() {  # reading <cand> <ref> <out>
  python -u $AUDIT/read_std_rolling.py --cand $RUNS/$1/out --ref $RUNS/$2/out --out $RUNS/$1/reading_vs_$2 \
    > $RUNS/$1/reading_vs_$2.log 2>&1
}

say "1. reading std vs ref-002 (background)"
if [ ! -f $RUNS/panel-rolling-std-001/reading_vs_panel-rolling-ref-002/reading.json ]; then
  reading panel-rolling-std-001 panel-rolling-ref-002 &
fi

say "2. wait for other jobs on this runtime"
while pgrep -f "nig_timenet_torch.jobs.(rolling|panel)" > /dev/null; do sleep 60; done

say "3. the 4h + 1h artifact"
if [ ! -f $ART_DRIVE/manifest.json ]; then
  python -u -m nig_timenet_torch.data_io.lower_timeframe $ART4 $ART2 --tf 1h --window 32 --cache $CACHE \
    --workers 16 --overwrite || { say "artifact build failed"; exit 1; }
  rm -rf $ART_DRIVE.tmp && cp -r $ART2 $ART_DRIVE.tmp && mv $ART_DRIVE.tmp $ART_DRIVE
elif [ ! -f $ART2/manifest.json ]; then
  cp -r $ART_DRIVE $ART2.tmp && mv $ART2.tmp $ART2
fi
python -c "import json; d=json.load(open('$ART2/derivation.json')); print('missing share', d['missing_share'], \
'fingerprint', d['fingerprint'][:12])"

say "4. smoke, then the jobs"
cfg panel-rolling-2tf-smoke-001 $ART2
job panel-rolling-2tf-smoke-001 15 || { say "smoke failed: see $RUNS/panel-rolling-2tf-smoke-001.log"; exit 1; }
cfg panel-rolling-2tf-001 $ART2; cfg panel-rolling-ref-003 $ART2; cfg panel-rolling-ls-001 $ART4
job panel-rolling-2tf-001 400 & P1=$!
job panel-rolling-ref-003 400 & P2=$!
# panel-rolling-ls-001 started on 6c928bb: its checkpoints only resume on that commit (run identity)
[ -d /content/nig_ls/.git ] || git -C $CODE worktree add -f /content/nig_ls 6c928bb > /dev/null
job panel-rolling-ls-001 400 /content/nig_ls & P3=$!
wait $P1; say "2tf exit $?"; wait $P2; say "ref-003 exit $?"; wait $P3; say "ls exit $?"

say "5. readings"
reading panel-rolling-2tf-001 panel-rolling-ref-003
reading panel-rolling-ls-001 panel-rolling-ref-002
wait
say "chain done"

if [ "${SHUTDOWN:-0}" = "1" ]; then
  say "6. releasing the runtime"
  python - <<'PY'
import glob
import jupyter_client
f = sorted(glob.glob("/root/.local/share/jupyter/runtime/kernel-*.json"))[-1]
kc = jupyter_client.BlockingKernelClient(connection_file=f)
kc.load_connection_file()
kc.start_channels()
kc.execute("from google.colab import runtime; runtime.unassign()")
PY
fi
