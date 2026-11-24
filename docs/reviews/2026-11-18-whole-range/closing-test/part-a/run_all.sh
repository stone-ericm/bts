#!/bin/zsh
# Closing test part a: re-run the four slice harnesses on FINAL (bc143131) in this clean copy.
# Harnesses are copies of docs/reviews/2026-11-18-whole-range/slice-*/ placed at .codex-review/<slice>/,
# their original location relative to the repo root (their path constants assume it). One change:
# slice-1-carbon/rehearsal_v3_harness.py also pins bts.health.runner.datetime (Carbon's C1 note 2).
# Each step: own log, exit code recorded, bounded by timeout. Steps run one after another.
cd /Users/eric/projects/bts-work/wr-closing-harness || exit 9
OUT=$PWD/closing-a/out; mkdir -p $OUT
H=$PWD/.codex-review
export PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York
PY=$PWD/.venv/bin/python
step() { local name=$1 secs=$2; shift 2; echo "$(date '+%F %T') start $name" >> $OUT/steps.txt
  timeout $secs "$@" > $OUT/$name.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $name rc=$rc" >> $OUT/steps.txt; }
HEADPP=$PWD/src
B2=$H/slice-2-codex/base/src

# slice 1 Codex
step s1-rehearsal 3600 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_rehearsal.py --logs $OUT/s1-rehearsal-cases
step s1-edges 2400 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_edges.py
cp $H/slice-1-codex/opening_edges.json $OUT/s1-edges.json 2>/dev/null
step s1-edges-fixed 2400 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_edges.py --expect-fixed
cp $H/slice-1-codex/opening_edges.json $OUT/s1-edges-fixed.json 2>/dev/null
step s1-ops 1200 env PYTHONPATH=$HEADPP $PY $H/slice-1-codex/opening_ops.py
cp $H/slice-1-codex/opening_ops.json $OUT/s1-ops.json 2>/dev/null

# slice 1 Carbon (pytest files)
step c1-carbon 2400 env PYTHONPATH=$HEADPP:$H/slice-1-carbon $PY -m pytest -p no:cacheprovider -q $H/slice-1-carbon/test_slice1_rehearsal_final.py $H/slice-1-carbon/test_slice1_probes_v3.py -rA

# slice 2 Codex
step s2-matrix-head 2400 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/offline_fixtures.py
step s2-matrix-policy-head 2400 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/offline_fixtures.py --production
step s2-matrix-policy-head-inclusive 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/offline_fixtures.py --production --inclusive-control --case last_date
step s2-matrix-policy-base-bridge 2400 env PYTHONPATH=$B2 $PY $H/slice-2-codex/offline_fixtures.py --production --bridge
for ph in startup workflow statuses; do step s2-deploy-$ph 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py $ph; done
DR=$OUT/deploy-root; rm -rf $DR; mkdir -p $DR
step s2-deploy-prepare 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py prepare --root $DR
step s2-deploy-rollback 1200 env PYTHONPATH=$B2 $PY $H/slice-2-codex/deploy_probe.py rollback --root $DR
step s2-deploy-forwarded 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py forwarded --root $DR
step s2-deploy-old-fetch 1200 env PYTHONPATH=$B2 $PY $H/slice-2-codex/deploy_probe.py old-fetch --root $DR
step s2-deploy-new-read 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py new-read --root $DR
step s2-cron 2400 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/cron_probe.py

# slice 3 Codex
for s in interactions refusals clocks_and_scoring cold_features; do step s3-$s 2400 env PYTHONPATH=$HEADPP $PY $H/slice-3-codex/$s.py; done
step s3-mutate 6000 env PYTHONPATH=$HEADPP $PY $H/slice-3-codex/mutate.py

# the adopted regression tests (the probes as the change sets took them over)
step adopted-tests 2400 $PY -m pytest -p no:cacheprovider -q tests/test_wr_*.py -rfE
echo "$(date '+%F %T') ALL STEPS DONE" >> $OUT/steps.txt
