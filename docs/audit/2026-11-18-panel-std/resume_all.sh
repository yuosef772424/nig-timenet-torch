#!/usr/bin/env bash
# Resume everything of the 2026-11-18/19 panel session on a NEW Colab GPU runtime (the old one stopped at about 22:58
# UTC on 2026-11-18). No secret in this file: the GitHub token is read from Drive by the setup recipe.
#   1. main checkout (branch head) + cached env + the 4h artifact copy; pinned checkouts for the jobs that resume
#      (a cut job resumes only on the commit it started on: docs/COLAB_OPS.md)
#   2. local copies of the derived artifacts (4h + 1h, + USDT channel) from Drive
#   3. resume panel-rolling-2tf-001 / -ref-003 (commit 03bafba) and -usdt-001 / -std-order-001 (commit df4580b);
#      smoke then the flat-split pair panel-std-order-full-001 / -400k-001 (branch head)
#   4. RELEASE=1 (default): wait for every job, write the readings and the BTC-hedged book to Drive, flush Drive,
#      release the runtime (finish_and_release.sh)
# Run (one Colab cell, after mounting Drive and cloning the branch to /content/nig-timenet-torch):
#   !nohup bash /content/nig-timenet-torch/docs/audit/2026-11-18-panel-std/resume_all.sh \
#        > /content/drive/MyDrive/nig_timenet_runs/resume_all.log 2>&1 &
set -uo pipefail
BRANCH=${BRANCH:-ccr-acb890db-b2ekab}
CODE=/content/nig-timenet-torch
RUNS=/content/drive/MyDrive/nig_timenet_runs
ART=/content/drive/MyDrive/crypto_model/torch_artifacts
AUDIT=$CODE/docs/audit/2026-11-18-panel-std
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "1. setup $BRANCH"
cd /content
PYTHONPATH=$CODE/src python -c "from nig_timenet_torch.colab_ops.recipes import setup_script; \
print(setup_script('$BRANCH'))" > /content/setup_resume.sh || { say "clone the branch first"; exit 1; }
bash /content/setup_resume.sh || { say "setup failed"; exit 1; }
. /content/nig_env.sh
for pin in "03bafba /content/nig_03b" "df4580b /content/nig_df4"; do
  set -- $pin
  [ -d $2/.git ] || [ -f $2/.git ] || git -C $CODE worktree add -f $2 $1 > /dev/null || { say "worktree $1 failed"; exit 1; }
  say "checkout $2 at $(git -C $2 log --oneline -1 | cut -c1-60)"
done

say "2. artifacts"
for pair in "preprocessing_output_1_plus_1h /content/art_1h4h" "preprocessing_output_1_plus_usdt /content/art_usdt"; do
  set -- $pair
  [ -f $2/manifest.json ] || { cp -r $ART/$1 $2.tmp && mv $2.tmp $2; } || { say "copy $1 failed"; exit 1; }
done

cfg() {  # cfg <job> <artifact>
  mkdir -p /content/cfg
  python -c "import json; c=json.load(open('$CODE/docs/jobs/$1.json'))['config']; c['artifact']='$2'; \
json.dump(c, open('/content/cfg/$1.json','w'))"
}
run() {  # run <entry> <job> <checkout> <max minutes>: outputs and resume points on Drive
  local repo=$3
  PYTHONPATH=${PYTHONPATH//$CODE\/src/$repo\/src} NIG_REPO_DIR=$repo python -u -m nig_timenet_torch.jobs.$1 \
    --config /content/cfg/$2.json --out $RUNS/$2/out --max-minutes $4 >> $RUNS/$2.log 2>&1
}

say "3. resume the walk-forwards"
cfg panel-rolling-2tf-001 /content/art_1h4h; cfg panel-rolling-ref-003 /content/art_1h4h
cfg panel-rolling-usdt-001 /content/art_usdt; cfg panel-rolling-std-order-001 /content/art1
run rolling panel-rolling-2tf-001 /content/nig_03b 600 &
run rolling panel-rolling-ref-003 /content/nig_03b 600 &
run rolling panel-rolling-usdt-001 /content/nig_df4 600 &
run rolling panel-rolling-std-order-001 /content/nig_df4 600 &
sleep 60
say "smoke of the flat-split pair"
cfg panel-std-order-smoke-001 /content/art1
if run panel panel-std-order-smoke-001 $CODE 15; then
  cfg panel-std-order-full-001 /content/art1; cfg panel-std-order-400k-001 /content/art1
  run panel panel-std-order-full-001 $CODE 600 &
  run panel panel-std-order-400k-001 $CODE 600 &
else
  say "smoke failed: see $RUNS/panel-std-order-smoke-001.log (the flat-split pair is not started)"
fi
say "jobs started"

if [ "${RELEASE:-1}" = "1" ]; then
  cp $CODE/docs/audit/2026-11-19-rolling-memorisation/btc_book.py /content/btc_book.py
  CODE=$CODE bash $AUDIT/finish_and_release.sh
fi
