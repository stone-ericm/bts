#!/bin/zsh
# Closing test RE-RUN, part a: the four slice harnesses on FINAL2 (d0bdcdba) in this clean clone.
# Harnesses are copies of docs/reviews/2026-11-18-whole-range/slice-*/ placed at .codex-review/<slice>/,
# their original location relative to the repo root (their path constants assume it), with
# `git archive 619abda7 src scripts` as slice-2's base/. One harness change (as in the first run):
# slice-1-carbon/rehearsal_v3_harness.py also pins bts.health.runner.datetime (Carbon's C1 note 2).
# Invocations follow the first run's corrected forms (gap2.sh): slice-2 day matrices through
# offline_fixtures.day_matrix(); the statuses phase after startup/prepare, with their logs beside it, under BASE;
# slice-3 mutate.py with the 33 cases whose anchors exist (fresh-identity, fresh-policy, fallback-reload,
# fallback-fail have no anchor on FINAL2, as on FINAL). Each step: own log, exit code recorded, bounded by timeout.
cd /Users/eric/projects/bts-work/wr2-harness || exit 9
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

# slice 2 Codex: day matrices as slice 2 ran them
M="import sys; sys.argv=['day_matrix']+sys.argv[1:]; import offline_fixtures as f; f.day_matrix()"
cd $H/slice-2-codex
step s2-matrix-head 2400 env PYTHONPATH=$HEADPP:$H/slice-2-codex $PY -c "$M"
step s2-matrix-policy-head 2400 env PYTHONPATH=$HEADPP:$H/slice-2-codex $PY -c "$M" --production
step s2-matrix-policy-head-inclusive 1200 env PYTHONPATH=$HEADPP:$H/slice-2-codex $PY -c "$M" --production --inclusive-control --case last_date
step s2-matrix-policy-base-bridge 2400 env PYTHONPATH=$B2:$H/slice-2-codex $PY -c "$M" --production --bridge
cd /Users/eric/projects/bts-work/wr2-harness
for ph in startup workflow; do step s2-deploy-$ph 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py $ph; done
DR=$OUT/deploy-root; rm -rf $DR; mkdir -p $DR
step s2-deploy-prepare 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py prepare --root $DR
step s2-deploy-rollback 1200 env PYTHONPATH=$B2 $PY $H/slice-2-codex/deploy_probe.py rollback --root $DR
step s2-deploy-forwarded 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py forwarded --root $DR
step s2-deploy-old-fetch 1200 env PYTHONPATH=$B2 $PY $H/slice-2-codex/deploy_probe.py old-fetch --root $DR
step s2-deploy-new-read 1200 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/deploy_probe.py new-read --root $DR
cd $H/slice-2-codex
cp $OUT/s2-deploy-startup.log deploy-startup.log
cp $OUT/s2-deploy-prepare.log deploy-prepare.log
step s2-deploy-statuses 1200 env PYTHONPATH=$B2 $PY $H/slice-2-codex/deploy_probe.py statuses
cd /Users/eric/projects/bts-work/wr2-harness
step s2-cron 2400 env PYTHONPATH=$HEADPP $PY $H/slice-2-codex/cron_probe.py

# slice 3 Codex
for s in interactions refusals clocks_and_scoring cold_features; do step s3-$s 2400 env PYTHONPATH=$HEADPP $PY $H/slice-3-codex/$s.py; done
step s3-mutate 6000 env PYTHONPATH=$HEADPP $PY $H/slice-3-codex/mutate.py explicit-required snapshot-copy optional-refused dated-identity automatic-match exact-int dated-canonical saver-consistency non-dated-identity automatic-replacement cli-warmup-refusal cli-catchup-refusal orchestrate-catchup-refusal orchestrate-warmup-refusal preview-stamp run-stamp cascade-stamp cascade-reuse-stamp earliest-refusal-copy dated-clears-reading dm-identifier public-identifier age-14h future-five-min recorded-at-forwarded startup-threshold startup-scoreable startup-delivered morning-dedup-date morning-cross-date void-slate morning-dedup-date-control morning-warn-date-control

# the adopted regression tests (the probes as the change sets took them over), and the season walk's tests
step adopted-tests 3000 $PY -m pytest -p no:cacheprovider -q tests/test_wr_*.py tests/scripts/season_walk -rfE
echo "$(date '+%F %T') ALL STEPS DONE" >> $OUT/steps.txt
